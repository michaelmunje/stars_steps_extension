from __future__ import annotations

import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    pred,gt=np.asarray(pred),np.asarray(gt)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError("ade expects primary trajectories shaped (T, 2)")
    if pred.shape[0] == 0:
        raise ValueError(f"ade received an empty trajectory (pred.shape={pred.shape})")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError(f"ade received non-finite values (pred.shape={pred.shape}, gt.shape={gt.shape})")
    return float(np.mean(np.linalg.norm(pred-gt,axis=-1)))


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    pred,gt=np.asarray(pred),np.asarray(gt)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError("fde expects primary trajectories shaped (T, 2)")
    if pred.shape[0] == 0:
        raise ValueError(f"fde received an empty trajectory (pred.shape={pred.shape})")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError(f"fde received non-finite values (pred.shape={pred.shape}, gt.shape={gt.shape})")
    return float(np.linalg.norm(gt[-1]-pred[-1]))


def topk_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    pred,gt=np.asarray(pred),np.asarray(gt)
    if pred.ndim!=3 or gt.ndim!=2 or pred.shape[1:]!=gt.shape or pred.shape[-1] != 2:
        raise ValueError("topk_ade expects pred (K, T, 2) and gt (T, 2)")
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError(f"topk_ade received an empty trajectory (pred.shape={pred.shape})")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError(f"topk_ade received non-finite values (pred.shape={pred.shape}, gt.shape={gt.shape})")
    return float(np.min(np.mean(np.linalg.norm(pred-gt[None],axis=-1),axis=1)))


def topk_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    pred,gt=np.asarray(pred),np.asarray(gt)
    if pred.ndim!=3 or gt.ndim!=2 or pred.shape[1:]!=gt.shape or pred.shape[-1] != 2:
        raise ValueError("topk_fde expects pred (K, T, 2) and gt (T, 2)")
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError(f"topk_fde received an empty trajectory (pred.shape={pred.shape})")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError(f"topk_fde received non-finite values (pred.shape={pred.shape}, gt.shape={gt.shape})")
    return float(np.min(np.linalg.norm(pred[:,-1]-gt[-1],axis=-1)))


def collision(
    path1: np.ndarray,
    path2: np.ndarray,
    person_radius: float = 0.1,
    inter_parts: int = 2,
) -> bool:
    if len(path1) != len(path2):
        raise ValueError("collision paths must have equal lengths")
    valid = np.isfinite(path1).all(axis=1) & np.isfinite(path2).all(axis=1)
    for index in range(len(path1) - 1):
        if not valid[index : index + 2].all():
            continue
        first = np.linspace(path1[index], path1[index + 1], inter_parts + 1)
        second = np.linspace(path2[index], path2[index + 1], inter_parts + 1)
        if np.min(np.linalg.norm(first - second, axis=1)) <= 2 * person_radius:
            return True
    return False


def collision_metrics(
    primary_prediction: np.ndarray,
    predicted_neighbors: np.ndarray,
    ground_truth_neighbors: np.ndarray,
) -> tuple[bool, bool]:
    def any_collision(neighbors:np.ndarray,person_radius:float=.1,inter_parts:int=2)->bool:
        neighbors=np.asarray(neighbors)
        if neighbors.size==0:return False
        if neighbors.ndim!=3 or neighbors.shape[1:]!=primary_prediction.shape:
            raise ValueError("neighbor paths must be shaped (N, T, 2)")
        primary_start=primary_prediction[:-1];primary_delta=np.diff(primary_prediction,axis=0)
        neighbor_start=neighbors[:,:-1];neighbor_delta=np.diff(neighbors,axis=1)
        valid=np.isfinite(neighbor_start).all(axis=-1)&np.isfinite(neighbors[:,1:]).all(axis=-1)
        valid&=np.isfinite(primary_start).all(axis=-1)[None]&np.isfinite(primary_prediction[1:]).all(axis=-1)[None]
        alpha=np.linspace(0.0,1.0,inter_parts+1)
        primary_points=primary_start[None,:,None,:]+primary_delta[None,:,None,:]*alpha[None,None,:,None]
        neighbor_points=neighbor_start[:,:,None,:]+neighbor_delta[:,:,None,:]*alpha[None,None,:,None]
        distance=np.linalg.norm(primary_points-neighbor_points,axis=-1)
        return bool(np.any((distance<=2*person_radius)&valid[:,:,None]))
    pred_col=any_collision(predicted_neighbors)
    gt_col=any_collision(ground_truth_neighbors)
    return pred_col, gt_col
