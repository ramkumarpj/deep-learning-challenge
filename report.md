# Alphabet Soup Charity Prediction Model

## Overview
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. The goal is to create a Binary Classifier,
using the features provided in the dataset that can predict whether applications will be successful, if funded by Alphabet Soup.


## Results

### Data Preprocessing

- The target feature for the model is 'IS_SUCCESSFUL'

- Features used for creating the model:
* APPLICATION_TYPE — Alphabet Soup application type
* AFFILIATION — Affiliated sector of industry
* CLASSIFICATION — Government organization classification
* USE_CASE — Use case for funding
* ORGANIZATION — Organization type
* STATUS — Active status
* INCOME_AMT — Income classification
* SPECIAL_CONSIDERATIONS — Special considerations for application
* ASK_AMT — Funding amount requested
* IS_SUCCESSFUL — Was the money used effectively

* EIN and NAME columns were dropped from the original dataset

* The number of unique values for each column was determined. For columns that have more than 10 unique values, the number of data points for each unique value was determined. The number of data points for each unique value is used to pick a cutoff point to bin "rare" categorical variables together in a new value, "Other". Following features were binned using this method-
- APPLICATION_TYPE
- CLASSIFICATION
- INCOME_AMT
- AFFILIATION
- USE_CASE
- ORGANIZATION
* The categorical values were encoded using pandas 'get_dummies' function
* train_test_split function was used to split the training and test dataset
* Scikit Learn's StandardScaler() was used to scale the training and testing features



### Compiling, Training and Evaluating the Model

* A neural network model was created by assigning the number of input features and nodes for each layer using TensorFlow and Keras
* The first hidden layer was created using 32 input features, 60 nodes and 'relu' as the activation function.
* The second hidden layer was created using 32 input features, 24 nodes and 'relu' as the activation function.
* The output layer uses 'sigmoid' for the activation function.
* The model structure was validated and found to be using 3469 parameters in total.
* The model was compiled with loss function as 'binary_crossentropy', optimizer set to 'adam' and metrics 'accuracy'
* The compiled model was trained using training data set for 100 epochs. A callback function was used to save the weights every 5 epochs.
* The trained model reached an accuracy of 0.5324 and loss of 0.6911 after 36 epochs
* The model has an accuracy of 0.6780 and loss of .8298 for test data.
*  The model was saved to a file in HDF5 format.

* The model did not achieve the target performance of higher than 75% accuracy.

* The rare categorical values for following features were binned separately to improve the performance
  - INCOME_AMT
  - AFFILIATION
  - USE_CASE
  - ORGANIZATION
* The first hidden layer was created using 60 nodes and second hidden layer was created using 24 nodes for the optimized model.
* The other improvements tried out didn't have a favorable impact on the model performance, hence these changes were removed from the notebook.
  - dropping INCOME_AMT, USE_CASE, ORGANIZATION columns
  - Adding 3 extra hidden layers
  - Use 'tanh' as the activation function.  

## Summary

The binary classifier model created to predict selection of applications for funding has an accuracy of 67%. I would recommend using hyper parameter tuning to create a model which can provide a better performance.  
