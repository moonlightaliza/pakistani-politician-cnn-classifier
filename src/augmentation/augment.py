"""
Augmentation Pipeline — Phase 1
Defines train/val/test transforms for politician image classification.

Run from project root:
    python3 src/augmentation/augment.py   ← to preview augmented samples
"""

from torchvision import transforms

# ImageNet stats (ResNet + EfficientNet both pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224


def get_transforms(split: str = "train"):
    """
    Args:
        split: 'train' → augmented | 'val' / 'test' → clean only
    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    print("Train transforms:")
    print(get_transforms("train"))
    print("\nVal/Test transforms:")
    print(get_transforms("val"))
