from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DetectionResult:
    boxes: torch.Tensor  # [N, 4] normalized xyxy
    scores: torch.Tensor  # [N]


class YoloV8DetectorAdapter:
    """
    Placeholder detector adapter.
    Replace this with an actual YOLOv8 call while preserving output contract.
    """

    def __init__(self, max_elements: int = 16) -> None:
        self.max_elements = max_elements

    def __call__(self, visual_feature_map: torch.Tensor) -> list[DetectionResult]:
        b, _, h, w = visual_feature_map.shape
        activation = visual_feature_map.mean(dim=1)
        flat = activation.reshape(b, -1)
        k = min(self.max_elements, flat.shape[-1])
        scores, idx = torch.topk(flat, k=k, dim=-1)

        results: list[DetectionResult] = []
        for i in range(b):
            y = torch.div(idx[i], w, rounding_mode="floor")
            x = idx[i] % w
            x0 = (x.float() / w).clamp(0, 1)
            y0 = (y.float() / h).clamp(0, 1)
            box_w = torch.full_like(x0, 1.0 / max(w, 1))
            box_h = torch.full_like(y0, 1.0 / max(h, 1))
            x1 = (x0 + box_w).clamp(0, 1)
            y1 = (y0 + box_h).clamp(0, 1)
            boxes = torch.stack([x0, y0, x1, y1], dim=-1)
            results.append(DetectionResult(boxes=boxes, scores=scores[i]))

        return results
