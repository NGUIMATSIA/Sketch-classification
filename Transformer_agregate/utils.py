from prettytable import PrettyTable
import torch
from tqdm import tqdm
import numpy as np

def count_parameters(model, plot_table=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if plot_table:
      print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def predict_classes(model, loader, device):
  """
  Function to make predictions on test data
  """
  model.eval()

  all_predictions = []

  with torch.no_grad():

      for batch in tqdm(loader):
          inputs = batch["pixel_values"].to(device)

          outputs = model(inputs)
          predictions = torch.argmax(outputs.logits, dim=1)

          all_predictions.extend(predictions.cpu().numpy())

  return all_predictions

def experts_agregator_n5(alls, experts_acc):

    """
    Function to agregate all predictions of the five experts to make one final predicition
    
    Args:
        alls: a numpy array of size (number of experts, number of test images)
        experts_acc: a list of accuracy of all experts
    """

    agreement_distrib = np.zeros((alls.shape[1]))
    for sample in range(alls.shape[1]):
        unique = np.unique(alls[:, sample], return_counts=True)[1].max()
        agreement_distrib[sample] = unique

    prediction = np.zeros((alls.shape[1],))
    for idx, unique in enumerate(agreement_distrib):
        if unique == 5:
            prediction[idx] = alls[0, idx]
        else:
            unique_preds = np.unique(alls[:, idx], return_counts=True)
            if unique == 1:
                prediction[idx] = alls[0, idx]
            elif unique in [3,4]:
                majority_class = unique_preds[0][unique_preds[1].argmax()]
                prediction[idx] = majority_class
            else:
                if len(unique_preds[0]) == 3:# vote [2,2,1]
                    ttc = np.argsort(-unique_preds[1])[0:2] #top_two_classes
                    acc_group0 = sum([experts_acc[i] for i in range(5) if alls[i, idx] == unique_preds[0][ttc[0]]])/2
                    acc_group1 = sum([experts_acc[i] for i in range(5) if alls[i, idx] == unique_preds[0][ttc[1]]])/2
                    if acc_group0 > acc_group1:
                        prediction[idx] = unique_preds[0][ttc[0]]
                    else:
                        prediction[idx] = unique_preds[0][ttc[1]]
                else: # vote [2,1,1,1]
                    majority_class = unique_preds[0][unique_preds[1].argmax()]
                    prediction[idx] = majority_class



    return agreement_distrib, prediction




def experts_agregator_n4(alls, experts_acc):
    """
    same function with four experts
    """

    agreement_distrib = np.zeros((alls.shape[1]))
    for sample in range(alls.shape[1]):
        unique = np.unique(alls[:, sample], return_counts=True)[1].max()
        agreement_distrib[sample] = unique

    prediction = np.zeros((alls.shape[1],))
    for idx, unique in enumerate(agreement_distrib):
        if unique == 4:
            prediction[idx] = alls[0, idx]
        else:
            unique_preds = np.unique(alls[:, idx], return_counts=True)
            if unique == 1:
                prediction[idx] = alls[0, idx]
            elif unique == 3:
                majority_class = unique_preds[0][unique_preds[1].argmax()]
                prediction[idx] = majority_class
            else:
                if len(unique_preds[0]) == 2: # vote [2,2]
                    acc_group0 = sum([experts_acc[i] for i in range(4) if alls[i, idx] == unique_preds[0][0]])/2
                    acc_group1 = sum([experts_acc[i] for i in range(4) if alls[i, idx] == unique_preds[0][1]])/2
                    if acc_group0 > acc_group1:
                        prediction[idx] = unique_preds[0][0]
                    else:
                        prediction[idx] = unique_preds[0][1]
                else: #vote [2,1, 1]
                    majority_class = unique_preds[0][unique_preds[1].argmax()]
                    prediction[idx] = majority_class

    return agreement_distrib, prediction