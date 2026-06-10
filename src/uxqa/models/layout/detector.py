from dataclasses import dataclass
import warnings

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

    def __init__(
        self,
        max_elements: int = 16,
        backend: str = "placeholder",
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        self.max_elements = max_elements
        self.backend = backend.lower()
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._yolo = None

        if self.backend == "yolo":
            try:
                from ultralytics import YOLO  # type: ignore

                self._yolo = YOLO(self.model_path)
            except Exception as exc:  # pragma: no cover - runtime dependency guard
                warnings.warn(
                    f"YOLO backend unavailable ({exc}). Falling back to placeholder detector.",
                    RuntimeWarning,
                )
                self.backend = "placeholder"

    def _placeholder_detections(self, visual_feature_map: torch.Tensor) -> list[DetectionResult]:
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

    def _normalize_for_yolo(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def _yolo_detections(self, screenshot: torch.Tensor, visual_feature_map: torch.Tensor) -> list[DetectionResult]:
        assert self._yolo is not None
        x = self._normalize_for_yolo(screenshot).float().cpu()
        preds = self._yolo.predict(
            source=x,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        results: list[DetectionResult] = []
        for pred in preds:
            boxes = pred.boxes.xyxy if pred.boxes is not None else None
            conf = pred.boxes.conf if pred.boxes is not None else None

            if boxes is None or conf is None or boxes.numel() == 0:
                empty_boxes = torch.zeros((0, 4), device=visual_feature_map.device, dtype=visual_feature_map.dtype)
                empty_scores = torch.zeros((0,), device=visual_feature_map.device, dtype=visual_feature_map.dtype)
                results.append(DetectionResult(boxes=empty_boxes, scores=empty_scores))
                continue

            boxes = boxes[: self.max_elements]
            conf = conf[: self.max_elements]
            img_h = float(pred.orig_shape[0])
            img_w = float(pred.orig_shape[1])

            norm = boxes.clone()
            norm[:, [0, 2]] = norm[:, [0, 2]] / max(img_w, 1.0)
            norm[:, [1, 3]] = norm[:, [1, 3]] / max(img_h, 1.0)
            norm = norm.clamp(0, 1).to(device=visual_feature_map.device, dtype=visual_feature_map.dtype)
            conf = conf.to(device=visual_feature_map.device, dtype=visual_feature_map.dtype)
            results.append(DetectionResult(boxes=norm, scores=conf))

        return results

    def __call__(
        self,
        visual_feature_map: torch.Tensor,
        screenshot: torch.Tensor | None = None,
    ) -> list[DetectionResult]:
        if self.backend == "yolo" and self._yolo is not None and screenshot is not None:
            try:
                return self._yolo_detections(screenshot, visual_feature_map)
            except Exception as exc:  # pragma: no cover - runtime fallback
                warnings.warn(
                    f"YOLO inference failed ({exc}). Falling back to placeholder detector.",
                    RuntimeWarning,
                )
        return self._placeholder_detections(visual_feature_map)
