import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import get_datasets
from model_factory import ModelFactory
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Model Training for Sketch Classification")
    parser.add_argument("--data_train", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--data_val", type=str, required=True, help="Path to the validation data directory")
    parser.add_argument("--model_names", nargs='+', required=True, help="List of model names for the ensemble")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--experiment", type=str, default="experiment", help="Folder where experiment outputs are located")
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    return total_loss / len(val_loader), correct / len(val_loader.dataset)



def main():
    args = parse_args()

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = get_datasets(args.data_train, args.data_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    factory = ModelFactory(args.model_names, num_classes=250)
    models = factory.get_models()

    for name, model in models.items():
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        best_val_accuracy = 0.0
        patience = 5
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
            val_loss, val_accuracy = validate_model(model, val_loader, device)
            print(f"Model: {name}, Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(args.experiment, f'{name}_best.pth'))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered for {name} at epoch {epoch}")
                    break

        print(f"Best model for {name} saved with accuracy: {best_val_accuracy}")
          
    # Load best models for ensemble prediction
    for name, _ in models.items():
        checkpoint = torch.load(os.path.join(args.experiment, f'{name}_best.pth'))
        models[name].load_state_dict(checkpoint)


if __name__ == "__main__":
    main()
