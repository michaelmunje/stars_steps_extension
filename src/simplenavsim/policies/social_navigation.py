"""Socially aware local trajectory-scoring controller (batched, backend independent).

At every control step, for every environment:

1. Candidate commands (arcs): a grid of ``num_speeds x num_turn_rates`` constant ``(v, w)`` pairs, plus
   slow reverse. The default 11 x 31 grid (0.1 m/s, 0.1 rad/s steps; 372 arcs) includes driving
   straight and gentle corrections; the earlier 6 x 11 grid (0.3 rad/s steps, 77 arcs) forced
   zig-zags between coarse turns (decision 94).
2. Each candidate is rolled out over ``prediction_horizon`` seconds (``plan_dt`` steps) with the
   robot's acceleration limits; pedestrians are forecast by a ``PedestrianPredictor`` (constant
   velocity by default; the STARS forecaster can be plugged in).
3. Costs: goal progress, soft predicted collisions, pedestrian proximity (personal space),
   disruption (entering the space in front of moving pedestrians), static-obstacle clearance,
   speed tracking, turning and command smoothness, each weighted by a tunable parameter.
4. **Hard safety constraint**: candidates whose predicted clearance to a pedestrian falls below
   ``safety_margin`` within ``safety_horizon`` seconds, or that touch a static obstacle within
   the horizon, are discarded. If none survive, the robot performs an emergency stop.
5. The best command is applied and the controller replans at the next step.

``GoalSeekingPolicy`` (``goal_seeking.py``) is the same controller blind to pedestrians: it only
avoids static obstacles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from simplenavsim.core.geometry import obstacle_clearance, wrap_angle
from simplenavsim.core.state import Observation
from simplenavsim.policies.base import ConstantVelocityPredictor, ParameterSpec, PedestrianPredictor, Policy

PARAMETERS = (
    ParameterSpec("goal_weight", 2.0, 0.5, 50.0, log=True, description="Weight of progress toward the goal"),
    ParameterSpec("collision_weight", 5.0, 0.1, 200.0, log=True, description="Penalty for predicted collisions beyond the safety horizon"),
    ParameterSpec("pedestrian_weight", 1.0, 0.01, 200.0, log=True, description="Penalty for entering pedestrians' personal space"),
    ParameterSpec("disruption_weight", 1.0, 0.01, 200.0, log=True, description="Penalty for blocking the path ahead of moving pedestrians"),
    ParameterSpec("obstacle_weight", 1.0, 0.01, 200.0, log=True, description="Penalty for small static-obstacle clearance"),
    ParameterSpec("preferred_speed", 0.8, 0.3, 1.0, description="Preferred forward speed (m/s)"),
    ParameterSpec("personal_space_radius", 0.8, 0.3, 1.5, description="Desired clearance around pedestrians (m)"),
    ParameterSpec("prediction_horizon", 2.0, 1.0, 4.0, description="Local trajectory prediction duration (s)"),
    ParameterSpec("turn_penalty", 0.1, 0.0, 2.0, description="Penalty for turning rate"),
    ParameterSpec("smoothness_weight", 0.2, 0.0, 2.0, description="Penalty for command changes"),
    # Scales on the controller's fixed waiting-related costs (decision 69); 1 keeps the config value.
    ParameterSpec("speed_scale", 1.0, 0.1, 20.0, log=True, description="Multiplier on speed_cost"),
    ParameterSpec("retreat_scale", 1.0, 0.1, 10.0, log=True, description="Multiplier on retreat_cost"),
    ParameterSpec("alignment_weight", 0.1, 0.0, 1.0, description="Reward for facing the goal at the plan's end"),
    ParameterSpec("pass_side_weight", 0.0, 0.0, 5.0, description="Penalty for switching the side on which the robot passes the blocking pedestrian (decision 82)"),
    ParameterSpec("side_commit_weight", 0.0, 0.0, 5.0, description="Penalty for reversing turn direction within commit_steps of the last turn (decision 81)"),
    ParameterSpec("idle_turn_weight", 0.0, 0.0, 5.0, description="Penalty for turning away from the goal without progress (loops; decision 78)"),
)


@dataclass
class ControllerConfig:
    num_speeds: int = 11
    reverse_speed: float = 0.3     # also consider backing off at this speed (0 disables)
    num_turn_rates: int = 31        # odd, so w = 0 (straight) is a candidate
    plan_dt: float = 0.2
    max_horizon: float = 4.0
    safety_horizon: float = 1.0
    safety_margin: float = 0.05
    obstacle_clearance: float = 0.6
    speed_cost: float = 0.5
    retreat_cost: float = 2.0      # penalty on moving away from the goal (no orbiting instead of waiting)
    retreat_along_path: bool = False  # retreat measured at the worst point of the plan, not only its end
    progress_along_path: bool = False # progress from the mean goal distance over the plan, not its end (decision 89)
    two_stage: bool = False        # also score "wait for t, then steer to the goal", t in stage1_times (decision 90)
    stage1_times: tuple = (0.5, 1.0, 1.5)
    stage2_gain: float = 2.5       # heading-error gain (1/s) of the goal-steering second stage
    disruption_lookahead: float = 1.5
    commit_steps: int = 20         # control steps a chosen turn direction stays committed (2 s at dt 0.1)
    commit_min_turn: float = 0.3   # |w| (rad/s) that counts as choosing a side
    pass_range: float = 2.0        # m: the nearest pedestrian ahead within this range is "the blocker"
    max_speed: float = 1.0
    max_angular_speed: float = 1.5
    max_acceleration: float = 1.0
    max_angular_acceleration: float = 3.0
    robot_radius: float = 0.3


class SocialNavigationPolicy(Policy):
    name = "social"
    parameters = PARAMETERS
    consider_pedestrians = True

    def __init__(self, config: ControllerConfig | None = None, predictor: PedestrianPredictor | None = None) -> None:
        self.config = config or ControllerConfig()
        self.predictor = predictor or ConstantVelocityPredictor()
        self.last_info: dict[str, torch.Tensor] = {}
        self._side = self._side_left = None
        self._pass = None          # (blocker index, pass side, steps left), each [B]

    def reset(self, batch_size: int) -> None:
        if hasattr(self.predictor, "reset"):
            self.predictor.reset()
        self._side = None          # [B] committed turn direction (-1, 0, +1) and steps left
        self._side_left = None
        self._pass = None

    def _p(self, params: torch.Tensor, name: str) -> torch.Tensor:
        return params[:, self.parameter_names.index(name)]

    def candidates(self, device) -> torch.Tensor:
        c = self.config
        v = torch.linspace(0.0, c.max_speed, c.num_speeds, device=device)
        if c.reverse_speed > 0:
            v = torch.cat([v.new_tensor([-c.reverse_speed]), v])
        w = torch.linspace(-c.max_angular_speed, c.max_angular_speed, c.num_turn_rates, device=device)
        grid = torch.cartesian_prod(v, w)                                          # [M, 2]
        if not c.two_stage:
            return grid
        # Third column: 1 = after a stage-1 hold the plan steers straight for the goal. Only *waiting*
        # gets a second stage: giving every command one let any first move (even spinning) be
        # "rescued" by the goal-steering stage, so the robot spun (decision 90). One wait-then-go
        # candidate per hold time in stage1_times.
        waits = torch.tensor([[0.0, 0.0, t] for t in c.stage1_times], device=device)
        return torch.cat([torch.cat([grid, torch.zeros(grid.shape[0], 1, device=device)], 1), waits])

    def rollout_robot(self, obs: Observation, commands: torch.Tensor, steps: int) -> torch.Tensor:
        """Predicted robot positions ``[B, M, H, 2]`` and headings ``[B, M, H]`` for commands ``[M, 2]``."""
        c, dt = self.config, self.config.plan_dt
        B, M = obs.robot_pose.shape[0], commands.shape[0]
        x = obs.robot_pose[:, None, 0].expand(B, M).clone()
        y = obs.robot_pose[:, None, 1].expand(B, M).clone()
        theta = obs.robot_pose[:, None, 2].expand(B, M).clone()
        v, w = obs.robot_v[:, None].expand(B, M).clone(), obs.robot_w[:, None].expand(B, M).clone()
        positions, headings = [], []
        target_v, target_w = commands[None, :, 0].expand(B, M), commands[None, :, 1].expand(B, M)
        hold = commands[None, :, 2] if commands.shape[1] > 2 else None             # 0 = single-stage
        for step in range(steps):
            staged = None if hold is None else (hold > 0) & ((step + 1) * dt > hold + 1e-6)
            if staged is not None and bool(staged.any()):
                bearing = torch.atan2(obs.goal[:, None, 1] - y, obs.goal[:, None, 0] - x)
                error = wrap_angle(bearing - theta)
                to_goal_w = (c.stage2_gain * error).clamp(-c.max_angular_speed, c.max_angular_speed)
                to_goal_v = c.max_speed * torch.cos(error).clamp(min=0.0)       # slow while turning toward the goal
                target_w = torch.where(staged, to_goal_w, target_w)
                target_v = torch.where(staged, to_goal_v, target_v)
            v = v + (target_v - v).clamp(-c.max_acceleration * dt, c.max_acceleration * dt)
            w = w + (target_w - w).clamp(-c.max_angular_acceleration * dt, c.max_angular_acceleration * dt)
            mid = theta + 0.5 * w * dt
            x, y, theta = x + v * torch.cos(mid) * dt, y + v * torch.sin(mid) * dt, theta + w * dt
            positions.append(torch.stack([x, y], -1))
            headings.append(theta)
        return torch.stack(positions, 2), torch.stack(headings, 2)

    def act(self, obs: Observation, params: torch.Tensor) -> torch.Tensor:
        c = self.config
        device = obs.robot_pose.device
        commands = self.candidates(device)
        M = commands.shape[0]
        H = int(math.ceil(c.max_horizon / c.plan_dt))
        horizon = self._p(params, "prediction_horizon")
        t = torch.arange(1, H + 1, device=device) * c.plan_dt                      # [H]
        in_horizon = (t[None] <= horizon[:, None] + 1e-6).float()                 # [B, H]
        steps_used = in_horizon.sum(-1).clamp(min=1)
        robot_xy, robot_theta = self.rollout_robot(obs, commands, H)               # [B, M, H, 2]
        B = robot_xy.shape[0]
        goal = obs.goal[:, None, None]
        distance0 = torch.linalg.norm(obs.goal - obs.robot_pose[:, :2], dim=-1)    # [B]
        distance = torch.linalg.norm(robot_xy - goal, dim=-1)                      # [B, M, H]
        last = (steps_used.long() - 1)[:, None, None].expand(B, M, 1)
        final_distance = torch.gather(distance, 2, last)[..., 0]
        progress = (distance0[:, None] - final_distance) / (c.max_speed * horizon[:, None])
        if c.progress_along_path:
            # Near the goal, a plan that curls round and ends close scores like one that drives in:
            # averaging the goal distance over the plan rewards getting there sooner (decision 89).
            mean_distance = (distance * in_horizon[:, None]).sum(-1) / steps_used[:, None]
            progress = (distance0[:, None] - mean_distance) / (0.5 * c.max_speed * horizon[:, None])
        final_theta = torch.gather(robot_theta, 2, last)[..., 0]
        final_xy = torch.gather(robot_xy, 2, last[..., None].expand(B, M, 1, 2))[:, :, 0]
        bearing = torch.atan2(obs.goal[:, None, 1] - final_xy[..., 1], obs.goal[:, None, 0] - final_xy[..., 0])
        alignment = torch.cos(wrap_angle(bearing - final_theta))
        # Arriving: no reward for overshooting; progress saturates at the goal.
        cost = -self._p(params, "goal_weight")[:, None] * (progress + self._p(params, "alignment_weight")[:, None] * alignment)
        # Retreating from the goal is only worth it to avoid danger: penalise it outright, so the
        # controller waits rather than circling when pedestrians block the way.
        retreat_cost = c.retreat_cost * self._p(params, "retreat_scale")[:, None]
        if c.retreat_along_path:
            # A loop that comes back to the same distance by the horizon is still a retreat.
            worst = torch.where(in_horizon[:, None] > 0, distance, torch.zeros_like(distance)).amax(-1)
            retreat = (worst - distance0[:, None]) / (c.max_speed * horizon[:, None])
            cost = cost + retreat_cost * torch.relu(torch.maximum(retreat, -progress))
        else:
            cost = cost + retreat_cost * torch.relu(-progress)
        crowding = torch.zeros(B, device=device)
        valid = torch.ones(B, M, dtype=torch.bool, device=device)
        future = None
        pass_side = torch.zeros(B, M, device=device)                              # side of the blocker each plan ends on
        blocker = torch.zeros(B, dtype=torch.long, device=device)
        has_blocker = torch.zeros(B, dtype=torch.bool, device=device)

        # Static obstacles (hard within the whole horizon, soft clearance cost).
        clearance_o, _ = obstacle_clearance(robot_xy.reshape(B, M * H, 2), obs.circles, obs.circle_mask,
                                            obs.segments, obs.segment_mask)
        clearance_o = clearance_o.view(B, M, H) - c.robot_radius
        valid &= ~((clearance_o < c.safety_margin) & (in_horizon[:, None] > 0)).any(-1)
        obstacle = (torch.relu(1.0 - clearance_o / c.obstacle_clearance) ** 2 * in_horizon[:, None]).sum(-1) / steps_used[:, None]
        cost = cost + self._p(params, "obstacle_weight")[:, None] * obstacle

        if self.consider_pedestrians and obs.ped_pos.shape[1]:
            future = self.predictor.predict(obs, H, c.plan_dt)                     # [B, P, H, 2]
            visible = obs.ped_visible[:, None, None, :]                            # [B, 1, 1, P]
            rel = robot_xy[:, :, :, None] - future.permute(0, 2, 1, 3)[:, None]    # [B, M, H, P, 2]
            gap = torch.linalg.norm(rel, dim=-1) - obs.ped_radius[:, None, None] - c.robot_radius
            gap = torch.where(visible, gap, torch.full_like(gap, 1e3))
            safety = (t <= c.safety_horizon + 1e-6)[None, None, :, None]
            valid &= ~((gap < c.safety_margin) & safety).any(dim=(-1, -2))
            # Current overlap check too (a pedestrian already too close: only moving away stays valid).
            weight = in_horizon[:, None, :, None]
            collision = ((gap < 0).float() * weight).amax(-1).sum(-1) / steps_used[:, None]
            personal = self._p(params, "personal_space_radius")[:, None, None, None]
            proximity = (torch.relu(1.0 - gap / personal) ** 2 * weight).sum(dim=(-1, -2)) / steps_used[:, None]
            speed = torch.linalg.norm(obs.ped_vel, dim=-1)                          # [B, P]
            heading = obs.ped_vel / speed.clamp(min=1e-6)[..., None]
            ahead = (rel * heading[:, None, None]).sum(-1) / torch.linalg.norm(rel, dim=-1).clamp(min=1e-6)
            zone = personal + speed[:, None, None] * c.disruption_lookahead
            moving = (speed > 0.2).float()[:, None, None]
            disruption = (0.5 * (1 + ahead) * torch.relu(1.0 - gap / zone) * moving * (speed[:, None, None] / 1.3)
                          * weight * visible).sum(dim=(-1, -2)) / steps_used[:, None]
            # How close the nearest visible pedestrian is now, relative to personal space (0 = far).
            now_gap = torch.where(obs.ped_visible, torch.linalg.norm(obs.ped_pos - obs.robot_pose[:, None, :2], dim=-1)
                                  - obs.ped_radius - c.robot_radius, torch.full_like(obs.ped_radius, 1e3))
            crowding = torch.relu(1.0 - now_gap.min(dim=-1).values / (2.0 * personal[:, 0, 0, 0])).clamp(max=1.0)
            # Pass-side memory: the nearest visible pedestrian ahead (toward the goal, within
            # pass_range) is the blocker; each plan ends left (+1) or right (-1) of it in the goal
            # frame. Switching sides on the same blocker within commit_steps costs extra.
            goal_dir = obs.goal - obs.robot_pose[:, :2]
            goal_dir = goal_dir / torch.linalg.norm(goal_dir, dim=-1, keepdim=True).clamp(min=1e-6)
            rel_now = obs.ped_pos - obs.robot_pose[:, None, :2]
            dist_now = torch.linalg.norm(rel_now, dim=-1)
            ahead = (rel_now * goal_dir[:, None]).sum(-1) > 0
            candidate = obs.ped_visible & ahead & (dist_now < c.pass_range)
            has_blocker = candidate.any(-1)
            blocker = torch.where(candidate, dist_now, torch.full_like(dist_now, 1e9)).argmin(-1)
            last_step = (steps_used.long() - 1).clamp(min=0)
            blocker_end = future[torch.arange(B, device=device), blocker, last_step]          # [B, 2]
            offset = final_xy - blocker_end[:, None]
            pass_side = torch.sign(goal_dir[:, None, 0] * offset[..., 1] - goal_dir[:, None, 1] * offset[..., 0])
            cost = (cost + self._p(params, "collision_weight")[:, None] * collision
                    + self._p(params, "pedestrian_weight")[:, None] * proximity
                    + self._p(params, "disruption_weight")[:, None] * disruption)

        v, w = commands[None, :, 0], commands[None, :, 1]
        # Speed tracking fades out when a pedestrian is close: slowing down / waiting is then fine.
        cost = cost + c.speed_cost * self._p(params, "speed_scale")[:, None] * (1.0 - crowding)[:, None] * ((v - self._p(params, "preferred_speed")[:, None]) / c.max_speed) ** 2
        cost = cost + self._p(params, "turn_penalty")[:, None] * w.abs() / c.max_angular_speed
        # Turning away from the goal without making progress starts a loop: penalise the loss of
        # goal alignment on arcs that get nowhere. Turning back toward the goal stays free (a
        # penalty on all idle turning blocked that recovery and made loops wider, decision 78).
        heading_now = torch.cos(wrap_angle(torch.atan2(obs.goal[:, 1] - obs.robot_pose[:, 1],
                                                       obs.goal[:, 0] - obs.robot_pose[:, 0]) - obs.robot_pose[:, 2]))
        idle = 1.0 - progress.clamp(0.0, 1.0)
        turned_away = torch.relu(heading_now[:, None] - alignment) / 2.0
        cost = cost + self._p(params, "idle_turn_weight")[:, None] * idle * turned_away
        change = ((v - obs.last_command[:, None, 0]).abs() / c.max_speed
                  + (w - obs.last_command[:, None, 1]).abs() / c.max_angular_speed)
        cost = cost + self._p(params, "smoothness_weight")[:, None] * change
        # Slow down near the goal instead of orbiting it.
        near = (distance0 < 1.0)[:, None]
        cost = cost + near * (v - distance0[:, None].clamp(max=1.0) * c.max_speed).abs()

        # Memory across control steps: once the robot turns one way to pass, reversing the turn
        # within commit_steps costs extra, so successive plans do not alternate sides (loops).
        if self._side is None or self._side.shape[0] != B:
            self._side = torch.zeros(B, device=device)
            self._side_left = torch.zeros(B, dtype=torch.long, device=device)
        turning = w.abs() > c.commit_min_turn
        reverses = turning & (torch.sign(w) * self._side[:, None] < 0) & (self._side_left > 0)[:, None]
        cost = cost + self._p(params, "side_commit_weight")[:, None] * reverses.float()
        if self._pass is None or self._pass[0].shape[0] != B:
            self._pass = (torch.full((B,), -1, dtype=torch.long, device=device), torch.zeros(B, device=device),
                          torch.zeros(B, dtype=torch.long, device=device))
        pass_ped, pass_kept, pass_left = self._pass
        same_blocker = has_blocker & (pass_ped == blocker) & (pass_left > 0)
        switches = same_blocker[:, None] & (pass_side * pass_kept[:, None] < 0)
        cost = cost + self._p(params, "pass_side_weight")[:, None] * switches.float()

        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        best = cost.argmin(dim=1)
        command = commands[best][:, :2]
        chosen = command[:, 1]
        commit = chosen.abs() > c.commit_min_turn
        self._side = torch.where(commit, torch.sign(chosen), self._side)
        self._side_left = torch.where(commit, torch.full_like(self._side_left, c.commit_steps),
                                      (self._side_left - 1).clamp(min=0))
        chosen_side = pass_side.gather(1, best[:, None])[:, 0]
        new_blocker = has_blocker & ~same_blocker
        self._pass = (torch.where(has_blocker, blocker, pass_ped),
                      torch.where(new_blocker, chosen_side, pass_kept),
                      torch.where(new_blocker, torch.full_like(pass_left, c.commit_steps), (pass_left - 1).clamp(min=0)))
        emergency = ~valid.any(dim=1)
        command = torch.where(emergency[:, None], torch.zeros_like(command), command)
        stop = (commands[:, 0].abs() + (commands[:, 0] != 0) * 1e3 + commands[:, 1].abs()).argmin()[None]  # v = 0, w ~ 0
        self.last_info = {"emergency_stop": emergency, "valid_fraction": valid.float().mean(1),
                          "stop_valid": valid[:, stop].any(1),
                          # For visualisation: the chosen plan [B, H, 2] (planning horizon only) and the
                          # pedestrian forecasts it was scored against [B, P, H, 2] (None without pedestrians).
                          "plan": robot_xy[torch.arange(B, device=device), best], "plan_steps": steps_used,
                          "predicted": future}
        return command
