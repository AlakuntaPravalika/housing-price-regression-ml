Housing Price Prediction using Linear Regression

1. Project Overview

This project implements Linear Regression from scratch to predict housing prices based on various property features such as area, crime rate, school rating, garage availability, and basement size.
The project demonstrates the complete machine learning pipeline including data preprocessing, model training, evaluation, and validation without using machine learning libraries like Scikit-learn.


2. Objectives

~ Implement Simple Linear Regression using Gradient Descent
~ Implement Multiple Linear Regression
~ Perform data preprocessing (missing values & outliers)
~ Evaluate models using standard regression metrics
~ Perform feature selection
~ Validate the model using cross validation
~ Visualize model performance


3. Dataset

The dataset contains synthetic housing data with the following features: 
      Feature                Description
      area	                 Size of the house (sq ft)
      crime_rate             Crime level in the neighborhood
      school_rating    	     Rating of nearby schools
      garage	           Number of garage spaces
      basement	           Basement area
      price	                 Target variable (house price)


4. Technologies Used
Python, NumPy, Pandas, Matplotlib


5. Machine Learning Workflow

     1. Load Dataset
     2. Handle Missing Values (Mean / Median Imputation)
     3. Remove Outliers using IQR
     4. Train Simple Linear Regression
     5. Train Multiple Linear Regression
     6. Perform Feature Selection
     7. Detect Multicollinearity using VIF
     8. Evaluate Model Performance
     9. Validate using K-Fold Cross Validation


6. Evaluation Metrics

The model performance is evaluated using:
     1. R² Score
     2. RMSE (Root Mean Squared Error)
     3. MAE (Mean Absolute Error)
     4. MAPE (Mean Absolute Percentage Error)


7. Visualizations

The project generates several diagnostic plots:
     1. Regression Line Plot
     2. Confidence Interval Plot
     3. Residual Plot
     4. Cost Function Convergence Plot


8. Key Concepts Implemented

     1. Gradient Descent Optimization
     2. Linear Regression
     3. Feature Selection
     4. Variance Inflation Factor (VIF)
     5. K-Fold Cross Validation


