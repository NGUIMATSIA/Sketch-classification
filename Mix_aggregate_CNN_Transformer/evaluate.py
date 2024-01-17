import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from data import get_datasets, get_transforms
from torchvision import models
from model import get_pretrained_model, CustomModel
import timm
from torchvision import datasets
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class EnsembleModel(nn.Module):
    def __init__(self, model_paths, num_classes, device):
        super(EnsembleModel, self).__init__()
        self.models = []

        for path in model_paths:
            model_name = os.path.basename(path).split('_')[0]

            if model_name == "resnet50":
                base_model = getattr(models, model_name)(weights=False)
                model = CustomModel(base_model, num_classes)
            elif model_name == "resnext50":
              base_model = getattr(models, "resnext50_32x4d")(weights=False)
              model = CustomModel(base_model, num_classes)
            elif model_name in ["efficientnet"]:
                model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
            elif model_name == "mobilenet":
                base_model = models.mobilenet_v2(pretrained=False)
                model = CustomModel(base_model, num_classes)
            elif model_name.startswith("vit"):
                model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
            elif model_name.startswith("deit"):
                model = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=num_classes)
            elif model_name.startswith("swin"):
                model = timm.create_model("swin_small_patch4_window7_224", pretrained=False, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            # Load the model state
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.models.append(model)

    def forward(self, x):
      preds_list = [model(x) for model in self.models]

      softmaxed_preds = [F.softmax(pred, dim=1) for pred in preds_list]

      mean_preds = torch.mean(torch.stack(softmaxed_preds), dim=0)

      return mean_preds

def plot_misclassified_images(images, true_labels, predicted_labels, test_loader):

    save_folder = "/content/drive/My Drive/experiment"
    os.makedirs(save_folder, exist_ok=True)

    for i in range(30):
        img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        plt.imshow(img)
    
        true_class_name = test_loader.dataset.classes[true_labels[i]]
        predicted_class_name = test_loader.dataset.classes[predicted_labels[i]]

        plt.title(f'True: {true_class_name}, Predicted: {predicted_class_name}')
        plt.axis('off')

    
        save_path = os.path.join(save_folder, f'image_{i}.png') 
        plt.savefig(save_path)
        plt.clf()

def opts():
    parser = argparse.ArgumentParser(
        description="Advenced Machine Learning"
        )
    parser.add_argument("--data",
                         type=str, 
                         default="test", 
                         help="folder where data is located"
                         )
    parser.add_argument("--model_names",
                         nargs='+',
                           help="list of model names for the ensemble"
                           )
    parser.add_argument("--experiment", type=str, default="/content/drive/My Drive/experiment")
  
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for training and validation"
        )
    parser.add_argument("--num_workers", 
                        type=int,
                          default=4, 
                          help="Number of workers for data loading"
                          )
    return parser.parse_args()

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    args = opts()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_paths = [os.path.join(args.experiment, f"{name}_best.pth") for name in args.model_names]
    ensemble_model = EnsembleModel(model_paths, num_classes=250, device=device)
    ensemble_model.eval()

    _, test_transforms = get_transforms(input_size=224)

    test_dataset = datasets.ImageFolder(os.path.join('/content/drive/My Drive/TU_berlin', args.data), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_accuracy = {}

    correct = 0
    total = 0
    misclassified_images = []
    true_labels_list = []
    predicted_labels_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = ensemble_model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for label, prediction in zip(labels, predicted):
                class_correct[label.item()] += (prediction == label).item()
                class_total[label.item()] += 1

                if len(args.model_names) >= 2 and prediction != label:
                    misclassified_images.append(inputs[0].cpu())
                    true_labels_list.append(label.item())
                    predicted_labels_list.append(prediction.item())

    global_accuracy = correct / total
    print(f"Global Accuracy: {global_accuracy}")

    for class_idx in class_correct:
        class_accuracy[test_loader.dataset.classes[class_idx]] = class_correct[class_idx] / class_total[class_idx]

    sorted_class_accuracy = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)

    top_5_classes = sorted_class_accuracy[:5]
    bottom_5_classes = sorted_class_accuracy[-5:]

    print(f"Top 5 Performing Classes: {top_5_classes}")
    print(f"Bottom 5 Performing Classes: {bottom_5_classes}")

    if len(args.model_names) >= 2:
        misclassified_images = torch.stack(misclassified_images)[:30]
        true_labels_list = true_labels_list[:30]
        predicted_labels_list = predicted_labels_list[:30]
        plot_misclassified_images(misclassified_images, true_labels_list, predicted_labels_list, test_loader)

if __name__ == "__main__":
    main()
