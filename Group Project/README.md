# Instructions:
In order for this to work, you will have to know a couple things about this folder. The **diabetes_detection.ipynb** & **diabeted_detection.py** are both TESTING files and not working fully.

The proper python file is the **dataset_classification.py** which will go in the following order:

1. Read in a Kaggle Dataset or a local path to a csv dataset
2. Read the columns and ask for the target column you want to train the nerual network to predict
3. Mine the rules using FP-Growth (Apriori was too expensive and resource intensive)
4. Name the model you are going to train in the .keras format for loading into a predication program separately
5. It will output the generalized rules from the dataset and use that as the training data for the neural network using tensorflow
6. The model will be output at the end of training in the .keras format

**NOTE:** Rename the .csv file that gets the general rule enhancement if you want to save the input to the neural network otherwise it will be overwritten in the next run