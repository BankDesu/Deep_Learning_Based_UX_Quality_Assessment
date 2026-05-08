from __future__ import annotations

import torch

try:
    from scipy import stats as _stats
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().float().numpy().flatten()


def kendall_tau(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not _SCIPY:
        raise ImportError("scipy is required. pip install scipy")
    tau, _ = _stats.kendalltau(_to_numpy(pred), _to_numpy(target))
    return float(tau)


def spearman_rho(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not _SCIPY:
        raise ImportError("scipy is required. pip install scipy")
    rho, _ = _stats.spearmanr(_to_numpy(pred), _to_numpy(target))
    return float(rho)


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not _SCIPY:
        raise ImportError("scipy is required. pip install scipy")
    r, _ = _stats.pearsonr(_to_numpy(pred), _to_numpy(target))
    return float(r)


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.float() - target.float()).abs().mean().item()


def rule_f1(
    pred_scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    rule_names: list[str] | None = None,
) -> dict[str, float | list]:
    """
    Per-rule and macro Precision / Recall / F1.

    pred_scores : [N, R] — model rule scores (1 = good UX, 0 = violation)
    labels      : [N, R] — ground-truth (1 = violated, 0 = ok)
    threshold   : rule_score < threshold → predicted violation
    """
    pred_viol = (pred_scores < threshold).float()
    labels = labels.float()

    tp = (pred_viol * labels).sum(0)
    fp = (pred_viol * (1 - labels)).sum(0)
    fn = ((1 - pred_viol) * labels).sum(0)

    precision = (tp / (tp + fp + 1e-8)).cpu()
    recall    = (tp / (tp + fn + 1e-8)).cpu()
    f1        = (2 * precision * recall / (precision + recall + 1e-8)).cpu()

    result: dict[str, float | list] = {
        "macro_precision": precision.mean().item(),
        "macro_recall":    recall.mean().item(),
        "macro_f1":        f1.mean().item(),
        "f1_per_rule":     f1.tolist(),
    }
    if rule_names:
        for i, name in enumerate(rule_names):
            result[f"f1_{name}"] = f1[i].item()
    return result
