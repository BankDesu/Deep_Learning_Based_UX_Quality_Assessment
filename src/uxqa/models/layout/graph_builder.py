import torch


def build_layout_graph(boxes: torch.Tensor) -> torch.Tensor:
    """
    Build a dense adjacency matrix from UI element centers.
    boxes: [N, 4] normalized xyxy
    returns adjacency: [N, N]
    """
    n = boxes.shape[0]
    if n == 0:
        return torch.zeros((0, 0), device=boxes.device, dtype=boxes.dtype)

    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    dist = torch.cdist(centers, centers, p=2)
    adjacency = 1.0 / (1.0 + dist)
    adjacency.fill_diagonal_(1.0)
    return adjacency
