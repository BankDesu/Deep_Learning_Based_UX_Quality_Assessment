from dataclasses import dataclass

import torch
from torch import nn

from .detector import YoloV8DetectorAdapter
from .gnn_encoder import SimpleGNNEncoder
from .graph_builder import build_layout_graph


@dataclass(slots=True)
class LayoutBranchOutput:
    layout_embedding: torch.Tensor
    layout_token: torch.Tensor
    graph_density: torch.Tensor


class LayoutBranch(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        out_dim: int,
        max_elements: int = 16,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.detector = YoloV8DetectorAdapter(max_elements=max_elements)
        self.gnn = SimpleGNNEncoder(in_dim=visual_dim, hidden_dim=out_dim, layers=gnn_layers)
        self.token_proj = nn.Linear(out_dim, out_dim)

    def _sample_node_features(self, feature_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        _, h, w = feature_map.shape
        if boxes.numel() == 0:
            return torch.zeros((0, feature_map.shape[0]), device=feature_map.device, dtype=feature_map.dtype)

        cx = (((boxes[:, 0] + boxes[:, 2]) * 0.5) * (w - 1)).long().clamp(0, max(w - 1, 0))
        cy = (((boxes[:, 1] + boxes[:, 3]) * 0.5) * (h - 1)).long().clamp(0, max(h - 1, 0))
        return feature_map[:, cy, cx].transpose(0, 1)

    def forward(self, visual_feature_map: torch.Tensor) -> LayoutBranchOutput:
        b, c, _, _ = visual_feature_map.shape
        detections = self.detector(visual_feature_map)
        batch_embeddings: list[torch.Tensor] = []
        graph_density: list[torch.Tensor] = []

        for i in range(b):
            boxes = detections[i].boxes
            node_features = self._sample_node_features(visual_feature_map[i], boxes)
            adjacency = build_layout_graph(boxes)
            graph_density.append(adjacency.mean() if adjacency.numel() else torch.tensor(0.0, device=visual_feature_map.device))

            if node_features.numel() == 0:
                emb = torch.zeros((self.token_proj.in_features,), device=visual_feature_map.device, dtype=visual_feature_map.dtype)
            else:
                encoded_nodes = self.gnn(node_features, adjacency)
                emb = encoded_nodes.mean(dim=0)
            batch_embeddings.append(emb)

        layout_embedding = torch.stack(batch_embeddings, dim=0).reshape(b, -1)
        layout_token = self.token_proj(layout_embedding).unsqueeze(1)
        density = torch.stack(graph_density, dim=0)
        return LayoutBranchOutput(layout_embedding=layout_embedding, layout_token=layout_token, graph_density=density)
