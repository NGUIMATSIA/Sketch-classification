from torch.utils.data import Dataset
from PIL import Image
from torch


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
