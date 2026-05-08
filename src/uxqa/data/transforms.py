from __future__ import annotations

try:
    from torchvision import transforms as T
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False

# ImageNet stats — Swin backbone pretrained on ImageNet
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _check() -> None:
    if not _TV_AVAILABLE:
        raise ImportError("torchvision is required. pip install torchvision")


def train_transforms(image_size: int = 224) -> "T.Compose":
    _check()
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def val_transforms(image_size: int = 224) -> "T.Compose":
    _check()
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
