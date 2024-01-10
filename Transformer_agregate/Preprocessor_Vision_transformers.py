
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

default_inputsize = (224, 224)

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

# expert 1 list of transformations
sketch_transform_1 = [
    transforms.RandomResizedCrop(size=default_inputsize, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
] 

# expert 2 list of transformations
sketch_transform_2 = [
    transforms.RandomRotation(0),
    transforms.RandomRotation(360*torch.rand(1).item())
] 

# expert 3 list of transformations
sketch_transform_3 = sketch_transform_1 + sketch_transform_2 

# expert 4: no transformation

# expert 5 list of transformations
sketch_transform_5 = [transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(0),
                      transforms.RandomHorizontalFlip()] 



class AugmentedCustomImageDataset(Dataset):
    def __init__(self, data_path, labels, image_processor,transform=None):
        """
        Args:
            data_path (list): Liste des chemins vers les images.
            labels (list): Liste des étiquettes correspondant aux images.
            transform (callable, optional): Transformation à appliquer aux images (par exemple, redimensionner, normaliser).
        """
        self.data_path = data_path
        self.labels = labels
        self.transform = transform
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        img_path = self.data_path[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        # depending on the expert that we want to train, a tranformation list will be choose: just keep on line uncomment
        img = sketch_transform_1[np.random.randint(len(sketch_transform_1))](img) 
        # img = sketch_transform_2[np.random.randint(len(sketch_transform_2))](img) 
        # img = sketch_transform_3[np.random.randint(len(sketch_transform_3))](img)
        # no transformation for expert 4
        # img = sketch_transform_5[np.random.randint(len(sketch_transform_5))](img) 

        if self.transform:
          img = self.transform(img)
          inputs = self.image_processor(img, return_tensors="pt", do_rescale=False)
          inputs['pixel_values'] = inputs['pixel_values'].squeeze()
          inputs['label'] = torch.tensor(label, dtype=torch.long)

          return inputs 


class CustomImageDataset(Dataset):
    """
    A customized image dataset object that will be used to prepare model's inputs

    Args:
        data_path (list): List of paths to images.
        labels (list): List of labels corresponding to the images.
        transform (callable, optional): Transformation to apply to images (e.g., resize, normalize).
    """
    
    def __init__(self, data_path, labels, image_processor, transform=None):
        
        self.data_path = data_path
        self.labels = labels
        self.transform = transform
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img_path = self.data_path[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
            inputs = self.image_processor(img, return_tensors="pt", do_rescale=False)
            inputs['pixel_values'] = inputs['pixel_values'].squeeze()
            inputs['label'] = torch.tensor(label, dtype=torch.long)

        return inputs
    

