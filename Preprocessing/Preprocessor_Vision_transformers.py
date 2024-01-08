
import torchvision.transforms as transforms
import torch

def resize_data(default_inputsize):

    data_transforms = transforms.Compose([
    transforms.Resize(default_inputsize),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.Lambda(lambda x: torch.clamp(x, min=0, max=1))
    ])

    return data_transforms


