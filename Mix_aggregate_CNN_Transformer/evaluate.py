import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
from torchvision import models
from model import get_pretrained_model, CustomModel
import timm
import torch.nn.functional as F
from collections import Counter

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

      final_preds = torch.argmax(mean_preds, dim=1)

      return final_preds




def opts():
    parser = argparse.ArgumentParser(description="Advenced Machine Learning")
    parser.add_argument("--data", type=str, default="data_sketches", help="folder where data is located")
    parser.add_argument("--model_names", nargs='+', help="list of model names for the ensemble")
    #parser.add_argument("--outfile", type=str, default="experiment/test_predit.CSV", help="output CSV file name")
    return parser.parse_args()

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main():
    args = opts()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_paths = [os.path.join('C:/Users/frank/Sketch-classification/experiment', f"{name}_best.pth") for name in args.model_names]

    ensemble_model = EnsembleModel(model_paths, num_classes=250, device=device)
    ensemble_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    #This part is just design to get the names of labels
    train_dir = "C:/Users/frank/Desktop/TU_berlin/train"
    class_names = sorted(os.listdir(train_dir))
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    test_dir = args.data

    class_correct = Counter()
    class_total = Counter()

    for root, dirs, files in tqdm(os.walk(test_dir)):
        for f in files:
            if f.lower().endswith('.png'):
                class_label = os.path.basename(root)
                img_path = os.path.join(root, f)
                img = pil_loader(img_path)
                img = transform(img).unsqueeze(0).to(device)
                output = ensemble_model(img)
                pred = torch.argmax(output).item()

                class_index = class_to_idx[class_label]

                class_total[class_label] += 1
                if pred == class_index:
                    class_correct[class_label] += 1

    
    class_performance = {cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0) for cls in class_total}
    
    overall_performance = sum(class_performance.values()) / len(class_performance) if class_performance else 0

    # Identifying top 5 and bottom 5 classes
    best_performing_classes = sorted(class_performance.items(), key=lambda x: x[1], reverse=True)[:5]
    worst_performing_classes = sorted(class_performance.items(), key=lambda x: x[1])[:5]

    # Displaying the results
    print(f"Global Accuracy: {overall_performance:.2f}")

    print("\nTop 5 Best Performing Classes:")
    for cls, performance in best_performing_classes:
        print(f"Class {cls}: Accuracy {performance:.2f}")

    print("\nTop 5 Worst Performing Classes:")
    for cls, performance in worst_performing_classes:
        print(f"Class {cls}: Accuracy {performance:.2f}")

if __name__ == "__main__":
    main()
