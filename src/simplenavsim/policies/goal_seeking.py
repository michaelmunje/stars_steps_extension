"""Baseline 1: goal-seeking controller that avoids static obstacles but ignores pedestrians.

It is ``SocialNavigationPolicy`` with every pedestrian term (costs and the hard pedestrian
constraint) switched off, so differences between the two isolate the social terms.
"""
from __future__ import annotations

from simplenavsim.policies.social_navigation import PARAMETERS, SocialNavigationPolicy


class GoalSeekingPolicy(SocialNavigationPolicy):
    name = "goal_seeking"
    consider_pedestrians = False
    # Same parameter layout as the social controller (pedestrian entries are ignored).
    parameters = PARAMETERS


POLICIES = {"social": SocialNavigationPolicy, "goal_seeking": GoalSeekingPolicy}


def make_policy(name: str, **kwargs):
    try:
        return POLICIES[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
