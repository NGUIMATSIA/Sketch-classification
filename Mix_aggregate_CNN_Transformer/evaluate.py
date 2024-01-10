import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model_factory import ModelFactory 
import csv
import torch
from torchvision import models
from collections import OrderedDict
from model import get_pretrained_model, CustomModel
import timm
import torch.nn.functional as F
from transformers import ViTForImageClassification
from collections import Counter

class EnsembleModel(nn.Module):
    def __init__(self, model_paths, num_classes, device):
        super(EnsembleModel, self).__init__()
        self.models = []

        for path in model_paths:
            model_name = os.path.basename(path).split('_')[0]

            if model_name == "resnet50":
                base_model = getattr(models, model_name)(pretrained=False)
                model = CustomModel(base_model, num_classes)
            elif model_name == "resnext50":
              base_model = getattr(models, "resnext50_32x4d")(pretrained=False)
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
    # Gather the predictions from all models
      preds_list = [model(x) for model in self.models]

    # Convert each model's logits to probabilities
      softmaxed_preds = [F.softmax(pred, dim=1) for pred in preds_list]

    # Calculate the average of the probabilities
      mean_preds = torch.mean(torch.stack(softmaxed_preds), dim=0)

    # Determine the final predictions by taking the argmax of the averaged probabilities
      final_preds = torch.argmax(mean_preds, dim=1)

      return final_preds




def opts():
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument("--data", type=str, default="data_sketches", help="folder where data is located")
    parser.add_argument("--model_names", nargs='+', help="list of model names for the ensemble")
    parser.add_argument("--outfile", type=str, default="experiment/kaggle.csv", help="output CSV file name")
    return parser.parse_args()

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    args = opts()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_paths = [os.path.join('experiment', f"{name}_best.pth") for name in args.model_names]
    ensemble_model = EnsembleModel(model_paths, num_classes= 250, device=device)
    ensemble_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dir = os.path.join(args.data, "test_images/mistery_category")
    with open(args.outfile, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label"])

        for f in tqdm(sorted(os.listdir(test_dir))):
            if f.lower().endswith('.png'):
                img_path = os.path.join(test_dir, f)
                img = pil_loader(img_path)
                img = transform(img).unsqueeze(0).to(device)
                
                output = ensemble_model(img)
                pred = output.item()
                
                filename_without_extension = os.path.splitext(f)[0]
                writer.writerow([filename_without_extension, pred])

    print(f"Successfully wrote {args.outfile}, you can upload this file to the Kaggle competition website.")

if __name__ == "__main__":
    main()
