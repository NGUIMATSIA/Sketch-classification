import torch.nn as nn
import torchvision.models as models
import timm  
from transformers import ViTForImageClassification  

class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes, dropout_rate=0.4):
        super(CustomModel, self).__init__()
        self.base_model = base_model

        # For ResNet, ResNeXt
        if hasattr(base_model, 'fc'):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()

        # For MobileNet, DenseNet
        elif hasattr(base_model, 'classifier'):
            if isinstance(base_model.classifier, nn.Sequential):
                if isinstance(base_model.classifier[-1], nn.Linear):
                    in_features = base_model.classifier[-1].in_features
                else:
                    raise AttributeError("Last layer of classifier is not Linear.")
            elif isinstance(base_model.classifier, nn.Linear):
                in_features = base_model.classifier.in_features
            else:
                raise AttributeError("Classifier attribute is not recognized.")
            base_model.classifier = nn.Identity()


       
        elif isinstance(base_model, ViTForImageClassification):
            in_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()

        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
  

def get_pretrained_model(model_name, num_classes=250):
    if model_name in ["resnet50", "resnext50_32x4d"]:
        base_model = models.__dict__[model_name](pretrained=True)
        model = CustomModel(base_model, num_classes)
    elif model_name.startswith("vit"):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "efficientnet_b3":
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    elif model_name.startswith("deit"):
        model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "mobilenet_v2":
        base_model = models.mobilenet_v2(pretrained=True)
        model = CustomModel(base_model, num_classes)
    elif model_name.startswith("swin"):
        model = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

