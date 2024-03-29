# Sketch Recognition
## Overview
The goal of this project is to train a machine learning model for sketch classification using the [How Do Humans Sketch Objects?](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) database. We divided this dataset into training (70%), validation (15%), and test (15%) sets using the **Split_and_Observe** notebook. You can download this split through the following link:
1. [Dataset](https://drive.google.com/drive/folders/1BTFb0hxsmjmgRxxdzVvDOOnGLxe3qSJa?usp=sharing)

To reproduce the results of this work, make sure to modify the paths to the folders in the `validation.py` (lines 66, 130, 99) and `main.py` (line 67) modules. We have added comments on these different lines. The `experiment.ipynb` notebook calls all the other modules to train all the models. We have also saved the weights of all trained models, which you can download using the following link:
1. [Trained models](https://drive.google.com/drive/folders/1MFJMzUQZmcRCnV9m6GzbVS5WFVEl0YT9?usp=sharing)

Since we don't have enough resources, we trained all these models on Google Colab (GPU A100). Each of these models, except for the CNN trained from scratch, takes approximately 4 to 5 hours to train (maximum number of epochs set to 50 + early stopping strategies).

