"""
HierarchyParser — converts UI accessibility tree / XML dict into SemanticElement list.

Expected input format (list of dicts):
    [
        {
            "bbox": [x0, y0, x1, y1],   # normalized [0,1]
            "type": "button",            # element type string
            "role": "button",            # aria role
            "label": "Sign Up",          # visible text or aria-label
            "interactive": True,         # clickable / focusable
            "heading_level": 0,          # 0 = not heading, 1-6 = H1-H6
        },
        ...
    ]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# Canonical element type vocabulary
ELEMENT_TYPES = [
    "button",
    "input",
    "text",
    "image",
    "link",
    "heading",
    "nav",
    "container",
    "icon",
    "unknown",
]
ELEMENT_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ELEMENT_TYPES)}
N_ELEMENT_TYPES = len(ELEMENT_TYPES)


@dataclass(slots=True)
class SemanticElement:
    """Single UI element with semantic metadata."""
    bbox: list[float]       # normalized [x0, y0, x1, y1]
    element_type: str       # one of ELEMENT_TYPES
    role: str               # aria role (raw string)
    label: str              # visible text or aria-label
    interactive: bool       # True if clickable / focusable
    heading_level: int      # 0 = not a heading; 1–6 = H1–H6


class HierarchyParser:
    """
    Stateless parser: dict list → SemanticElement list.

    Usage
    -----
    parser = HierarchyParser()
    elements = parser.parse(raw_hierarchy)          # list[SemanticElement]
    tensors  = parser.to_tensors(elements, device)  # dict[str, Tensor]
    """

    # Map common synonyms to canonical types
    _TYPE_ALIASES: dict[str, str] = {
        "btn": "button",
        "img": "image",
        "a": "link",
        "anchor": "link",
        "textview": "text",
        "label": "text",
        "edittext": "input",
        "textfield": "input",
        "textinput": "input",
        "h1": "heading",
        "h2": "heading",
        "h3": "heading",
        "h4": "heading",
        "h5": "heading",
        "h6": "heading",
        "navbar": "nav",
        "div": "container",
        "view": "container",
        "frame": "container",
        "imageview": "image",
        "ic": "icon",
    }

    def _normalize_type(self, raw: str) -> str:
        key = raw.strip().lower()
        key = self._TYPE_ALIASES.get(key, key)
        return key if key in ELEMENT_TYPE_TO_IDX else "unknown"

    def _infer_heading_level(self, raw_type: str, item: dict[str, Any]) -> int:
        """Infer heading level from type string or explicit field."""
        explicit = item.get("heading_level", 0)
        if explicit:
            return int(explicit)
        t = raw_type.strip().lower()
        if t in ("h1", "heading1"):
            return 1
        if t in ("h2", "heading2"):
            return 2
        if t in ("h3", "heading3"):
            return 3
        if t in ("h4", "heading4"):
            return 4
        if t in ("h5", "heading5"):
            return 5
        if t in ("h6", "heading6"):
            return 6
        return 0

    def parse(self, hierarchy: list[dict[str, Any]]) -> list[SemanticElement]:
        """Parse raw hierarchy dicts into SemanticElement list."""
        elements: list[SemanticElement] = []
        for item in hierarchy:
            bbox = list(item.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            # Clamp to [0,1]
            bbox = [max(0.0, min(1.0, float(v))) for v in bbox[:4]]
            if len(bbox) < 4:
                bbox = bbox + [0.0] * (4 - len(bbox))

            raw_type = str(item.get("type", "unknown"))
            etype = self._normalize_type(raw_type)
            role = str(item.get("role", ""))
            label = str(item.get("label", ""))
            interactive = bool(item.get("interactive", False))
            heading_level = self._infer_heading_level(raw_type, item)

            elements.append(
                SemanticElement(
                    bbox=bbox,
                    element_type=etype,
                    role=role,
                    label=label,
                    interactive=interactive,
                    heading_level=heading_level,
                )
            )
        return elements

    def to_tensors(
        self,
        elements: list[SemanticElement],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Convert SemanticElement list into tensors.

        Returns
        -------
        bboxes          : FloatTensor [N, 4]  — normalized xyxy
        type_ids        : LongTensor  [N]     — index into ELEMENT_TYPES
        interactive     : BoolTensor  [N]     — is element interactive?
        heading_levels  : LongTensor  [N]     — 0 or 1–6
        button_mask     : BoolTensor  [N]     — is element a button/CTA?
        """
        if not elements:
            empty = lambda dtype: torch.zeros(0, dtype=dtype, device=device)
            return {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "type_ids": empty(torch.long),
                "interactive": empty(torch.bool),
                "heading_levels": empty(torch.long),
                "button_mask": empty(torch.bool),
            }

        bboxes = torch.tensor(
            [e.bbox for e in elements], dtype=torch.float32, device=device
        )
        type_ids = torch.tensor(
            [ELEMENT_TYPE_TO_IDX.get(e.element_type, ELEMENT_TYPE_TO_IDX["unknown"]) for e in elements],
            dtype=torch.long,
            device=device,
        )
        interactive = torch.tensor(
            [e.interactive for e in elements], dtype=torch.bool, device=device
        )
        heading_levels = torch.tensor(
            [e.heading_level for e in elements], dtype=torch.long, device=device
        )
        button_mask = torch.tensor(
            [e.element_type == "button" or e.role in ("button", "cta") for e in elements],
            dtype=torch.bool,
            device=device,
        )
        return {
            "bboxes": bboxes,
            "type_ids": type_ids,
            "interactive": interactive,
            "heading_levels": heading_levels,
            "button_mask": button_mask,
        }
