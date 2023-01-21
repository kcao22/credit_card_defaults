
Inspiration & Guidance Credit: https://github.com/TrainingByPackt/Data-Science-Projects-with-Python

# Predicting Defaulting Credit Card Account Payments

## Business Problem
 - Credit card company providing 30,000 of their account holders' financial data information with a historical information of 6 months. 
 - Data is presented on the credit card account level, with each row representing a single, unique account.
 - Financial data for each account represents a 6 month historical period encoded each month for the status of an account's payments. Dataset contains a column indicating if an account has defaulted on its current month's payment.

## Project Overview
 - Built Python classification tool that predicts whether a credit card account will default on its next payment. Model performance is determined with a Receiver Operating Characteristic (ROC) Area Under Curve (AUC) value, with optimal model performance of ~0.776.
 - Used Python Pandas and Matplotlib to perform data quality checks and discovered underlying data issues.
 - Explored various classification models, optimized hyperparameters with cross validation methods, and visualized runtime performances to assist in selecting optimal hyperparameters.
 - Performed financial analysis given the case study's cost of contacting each customer netted a total max net savings of NT $15,767,877.

![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/financial_analysis_net_savings_positive_threshold.png)

## Libraries and Resources Used

 - **Python Version**: 3.9
 - **Main Packages**: Matplotlib, Pandas, scikit-learn, seaborn
 - **Environment Requirements**: conda env create -f environment.yml
 - **Data Cleaning Ideas**: https://github.com/PlayingNumbers/ds_salary_proj

 ## Data Cleaning and Quality Check

 Before beginning any model building, dataset exploration was performed to understand the data at hand. Furthermore, data quality checks were performed to ensure soundness of dataset. Below is a list of different data quality issues that required cleaning.
 
  - Created a data dictionary with all possible values for each column according to company's data layout explanations. Encoded values were presented as a separate sheet within data dictionary and serves as a reference page for data exploration. See below embedded image for a snapshot of data dictionary.
  - Value counts were examined to determine duplicate accounts in dataset. Although the case study claims a dataset of 30,000 records with each record being unique, there were a total of 313 accounts that appeared twice in the dataset. These accounts contained 0 values across all columns.
  - 'PAY_1', 'EDUCATION', and 'MARRIAGE' encoded values were found to contain previously undefined values. These values were either binned into a separate, already existing value of 'Other', or binned into its own group of 'Unknown' values.
  - 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' histograms contained suspect values. For example, PAY_3 data contained a small number of accounts with one month delinquency in payments. However, in PAY_2, the next payment month, there are a few thousand accounts with payment delinquency of two months. Time series wise, this would not be possible, thus PAY_2 through PAY_6 features were not considered in this case study. See below embedded image for a visual of PAY_# distributions.
  - 'BILL_AMT_#' fields contained negative values and rows with negative bill amount values had to be ommitted. 
  - 'PAY_AMT_#' fields were visually checked via distribution 
  
![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/data_dictionary_preview.PNG)

![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/pay_distributions_qc.png)

![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/log_pay_amt_distributions.png)

## General Feature Exploration in Relation to Payment Default Status

I first begin with univariate feature exploration before considering interactions between features and more complex, non-linear patterns.

Given that the case study problem is a binary classification problem, I first explore ANOVA F-Test statistic values as surface level analysis. This will provide an idea of which features will be more significant for modeling. To confirm the below results, I also employ scikit-learn's SelectPercentile method and select the top 20% of features by feature importance. Both results show that 'LIMIT_BAL' and 'PAY_1' are the top two most important features (larger F-statistic value signifies a greater difference in the mean values for the two classes for the given field).

![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/f_test_df.PNG)


A deeper dive into the 'PAY_1' feature default rates by 'PAY_1' encodings show a slight linear trend. Accounts that have paid the minimum monthly payment exhibit a lower default rate lower than the average overall dataset default rate. As delinquency in payments increase, so do the default rates for each encoded group.

![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/pay_1_default_rates_by_code.png)

Further, a deeper dive into the 'LIMIT_BAL' feature shows that accounts with higher credit limits exhibit a lower proportion of population defaulting on next payment and vice versa with lower credit limit account populations exhibiting a higher proportion of accounts defaulting. 

![Alt Text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/credit_limit_check.png)


## Machine Learning Model Exploration & Selection

Each model will explore hyperparameter optimization using a variety of visualizations and calculations. Below is a brief description for various models explored and a few hyperparameters optimized.

### Logistic Regression
With the dataset cleaned and some surface level univariate feature exploration performed, I begin with the basic logistic regression model for binary classification.

First, I examine if an important feature such as 'PAY_1' exhibits a linear relationship with the log odds of the probability of default. From the visual below, the relationship between the feature and log odds of probability of default is not really linear. Excluding the encoded values of (-2, 0, 2), all other encoded values exhibit an inconsistent up-and-down log offs of default trend. For the sake of the model exploration, I will assume that this is a sufficient enough linear relationship and continue to perform hyperparameter optimization. A pipeline is developed to encompass this process and min max scaling is introduced before further hyperparameter optimization is performed. Min Max scaling is introduced in order to scale features and ensure that gradient descent converges faster for logistic regression. Scaling will optimize coefficient parameter fitting for the model.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/log_odds_linear_or_nonlinear.png)

As an example, the 'C' hyperparameter is optimized via cross validation methods. For each split in the cross validation process, a new 'C' hyperparameter is set for the pipeline logistic regression model and can be seen in the visual below. Overfitting begins to occur after a 'C' value of ~ 10**-1. 

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/overfitting_example_c_hyperparameter.png)

For logistic regression, a higher ROC AUC value of ~0.73 is observed after hyperparameter tuning, with previous ROC AUC values of ~0.71 prior to 'C' hyperparameter tuning. 

The variance of ROC curves for each split is displayed below for the range of 'C' hyperparameter values.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/variance_in_splits.png)

### Decision Tree

Next, a single decision tree model is explored. Cross validation with a range of 'max_depth' values ranging between 1 and 12 are explored. To prevent obvious scenarios of overfitting, the max leaves are considered relative to the maxiumum number of positive samples. For my cross validation splits, I am using 3/4 of the training dataset to fit the model and the remaining 1/4 of the training set as validation set to evaluate ROC AUC scores for the model. Given this knowledge, with an 80/20 split of the total data into training and testing datasets, the training dataset will contain approximately 21,331 records. With a 75/25 split in the stratified cross validation process, the varying fitting datasets will contain approximately 16,000 samples. A maximum overfit model would have a single leaf for each training sample. With $$2^{n}$$ leaves for 'n' levels of max depth, setting $$2^{n}$$ = 16,000 and evaluating for 'n' using $$\log_2 16000$$ yields a maximum depth of 12. Therefore for the hyperparameter optimizaation of 'max_depth', I will use a range of 1 to 12. Visualizing the results, overfitting occurs after a 'max_depth' value of 2.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/rf_max_depth_single_param_overfitting_ex.png)

Optimizing the 'max_depth' value boosts the cross validation ROC AUC values up to ~0.74, a slight increase from logistic regression.

With how easily a single Decision Tree overfits, I will further explore the use of Random Forest ensemble models.

## Random Forest 

In addition to 'max_depth', 'n_estimators' is explored using the same cross validation approaches as above. This time, however, the mean fit time of the random forest ensemble model is also considered relative to the number of trees grown versus the mean ROC AUC scores. With this evaluation, I can determine if increasing the number of base models in the ensemble is worth it given the performance to fit time ratio. Below is a visual of these results. As seen, there are diminishing returns after a certain number of estimators. Once the number of decision trees grown exceeds approximately 20, the value of the mean testing ROC AUC from cross validation fluctuates up and down. Additionally, the mean run time increases linearly. Given this, an ideal number of base trees appears to be around 20 based on the range of 20 to 100 estimators. However, this may not be the optimal hyperparameter when tuned with other hyperparameters. This merely serves the purpose of another factor to consider when exploting features individually.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/rf_runtime_num_trees.png)

Another visualization to determine the optimal combination of hyperparameters optimized is the checkerboard model, pairing 'max_depth' feature alongside 'n_estimators' along with a colorbar representing the average validation ROC AUC scores and serving as a z-axis. From the image, I can determine that a combination of 'max_depth' = 9 and 'n_estimators' of 200 yields the highest ROC AUC of ~0.78.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/rf_pair_plot_checkerboard_graph.png)

Random forest feature importance can also be viewed based on the significance of a feature in growing the base estimator decision trees and reducing node impurity. As seen in the below visual, 'PAY_1', 'LIMIT_BAL', and the various 'PAY_#' features continue to show the greatest significance in predicting if an account defaults on payment in the coming month.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/rf_feature_imp.png)

## Model Performance Confirmation

To validate the model performance, I also plotted the predicted probability of defaulting alongside default rates in a decile and binned manner. Below is a visual of predicted probability rates of the test set divided into deciles, and then plotted against actual default rates according to each decile probability bin. In doing so, I can see that as the decile of predicted probability of default increases, so does default rate.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/financial_analysis_decile_default_rate.png)

A further plot of equally binned ranges of predicted probability rather than deciles can be seen below, with the same confirmation that the model correctly predicts a higher number of defaults for actually defaulting accounts.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/financial_analysis_binned_prob_default_rate.png)

An expected calibration error (ECE) plot was also created to determine the accuracy of the model fitting process. This was done using mean probabilities from the actual test set and the predicted probabilities grouped by prediction probability deciles. The ECE can be monitored in future model deployment to determine if 

## Financial Analysis

Finally, a financial analysis is provided to the client based on the test set results with the tuned Random Forest model. 

Below is a visual produced based on the case study's costs and average success of calling to settle account payments. With the Random Forest model providing ROC AUC values as a metric of success, it is vital to determine which threshold in the ROC AUC graph would provide the highest number of true positives identified relative to the cost per call and false positives. Below, a graph of thresholds along with net savings is calculated and a threshold of 0.24 for the model would generate the highest savings for the company. This would yield savings of NT $15,767,877.  

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/financial_analysis_net_savings_positive_threshold.png)

Costs also need to be taken into account relative to savings. A visual of this is also provided below. To generate the greatest savings, an upfront investment of ~ NT $2000 is needed per account.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/financial_analysis_net_savings_vs_cost.png)
