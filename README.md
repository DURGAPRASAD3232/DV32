# Wine Quality Prediction Using Support Vector Machine (SVM)
## Overview
This project aims to predict the quality of wine based on various physicochemical properties using a Support Vector Machine (SVM) classifier. The dataset used contains several features such as acidity, sugar content, pH levels, and alcohol content, which are used as input variables to predict the wine quality. The goal is to build a machine learning model that can accurately classify the quality of wine into different categories based on these features.

# Objective
The primary objective of this project is to:

Train a machine learning model using the SVM algorithm.
Predict the quality of wine based on its physicochemical properties.
Evaluate the performance of the model using various metrics like accuracy, precision, recall, and F1-score.
# Dataset
The dataset used in this project is the Wine Quality Dataset. This dataset contains information about the properties of both red and white wines, with the following columns:

Fixed acidity: The amount of fixed acids in the wine.
Volatile acidity: The amount of volatile acids in the wine.
Citric acid: The amount of citric acid in the wine.
Residual sugar: The amount of sugar left after fermentation.
Chlorides: The amount of chlorides in the wine.
Free sulfur dioxide: The amount of free sulfur dioxide in the wine.
Total sulfur dioxide: The amount of total sulfur dioxide in the wine.
Density: The density of the wine.
pH: The pH level of the wine.
Sulphates: The amount of sulphates in the wine.
Alcohol: The alcohol content in the wine.
Quality: The quality rating of the wine (target variable), typically on a scale from 0 to 10.
The dataset is available in two parts:

winequality-red.csv: Data for red wines.
winequality-white.csv: Data for white wines.
Methodology
The project follows these main steps:

# Data Preprocessing:

Combining the red and white wine datasets into one unified dataset.
Handling missing values, if any, and normalizing or scaling the data.
Splitting the dataset into training and testing subsets.
Model Selection:

A Support Vector Machine (SVM) algorithm is chosen to build the predictive model. SVM is a powerful classification technique that works well for both linear and non-linear decision boundaries.
 # Training the Model:

The model is trained on the training data, where the SVM algorithm learns to classify wines based on their physicochemical properties.
Evaluation:

After training, the model is tested on the unseen testing data.
Performance is evaluated using accuracy, precision, recall, and F1-score to assess the effectiveness of the SVM classifier.
Model Evaluation
The evaluation of the SVM model is done using common classification metrics, such as:

Accuracy: The proportion of correct predictions made by the model.
Precision: The proportion of positive predictions that were correct.
Recall: The proportion of actual positives that were correctly identified.
F1-Score: The harmonic mean of precision and recall, providing a single score to evaluate the model's performance.
Additionally, a confusion matrix is used to visualize the performance across different wine quality categories.

# Results
After training and evaluation, the model’s performance is measured. In this project, it is expected that the SVM model will provide reasonable predictions, though the exact results may vary based on hyperparameters and dataset splits.

# Conclusion
This project demonstrates how machine learning techniques, specifically Support Vector Machines (SVM), can be applied to predict wine quality based on its chemical properties. SVM provides an effective method for classification problems like this, and with proper preprocessing and tuning, the model can be optimized for better performance.
