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


def train_transforms_strong(image_size: int = 224) -> "T.Compose":
    """Stronger augmentation for small datasets (800 samples).

    Avoids aggressive spatial crops (UI layout matters) but adds
    affine jitter, stronger color, and random erasing.
    """
    _check()
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=0, translate=(0.04, 0.04), shear=3),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0),
    ])


def val_transforms(image_size: int = 224) -> "T.Compose":
    _check()
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
