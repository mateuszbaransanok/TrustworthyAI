from torchvision import transforms

AUGMENTATION: dict[str, transforms.Compose] = {
    'cifar10': transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    ),
}
AUGMENTATION['cifar100'] = AUGMENTATION['cifar10']

TRANSFORMS: dict[str, transforms.Compose] = {
    'mnist': transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    'cifar10': transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    ),
}
TRANSFORMS['cifar100'] = TRANSFORMS['cifar10']
TRANSFORMS['textures'] = TRANSFORMS['cifar10']
TRANSFORMS['gaussian'] = TRANSFORMS['cifar10']
TRANSFORMS['uniform'] = TRANSFORMS['cifar10']
