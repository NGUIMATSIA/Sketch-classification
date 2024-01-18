from model import get_pretrained_model
import Simple_CNN

class ModelFactory:
    def __init__(self, model_names, num_classes=250):
    
        if isinstance(model_names, str):
            model_names = [model_names]

        self.models = {name: get_pretrained_model(name, num_classes) for name in model_names}   
        print("Model names:", model_names)

    def get_models(self):
        return self.models
    
    def get_models_advanced(self):
        return self.models