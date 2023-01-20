
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

### Logistic Regression
With the dataset cleaned and some surface level univariate feature exploration performed, I begin with the basic logistic regression model for binary classification.

First, I examine if an important feature such as 'PAY_1' exhibits a linear relationship with the log odds of the probability of default. From the visual below, the relationship between the feature and log odds of probability of default is not really linear. Excluding the encoded values of (-2, 0, 2), all other encoded values exhibit an inconsistent up-and-down log offs of default trend. For the sake of the model exploration, I will assume that this is a sufficient enough linear relationship and continue to perform hyperparameter optimization. A pipeline is developed to encompass this process and min max scaling is introduced before further hyperparameter optimization is performed. Min Max scaling is introduced in order to scale features and ensure that gradient descent converges faster for logistic regression. Scaling will optimize coefficient parameter fitting for the model.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/log_odds_linear_or_nonlinear.png)

As an example, the 'C' hyperparameter is optimized via cross validation methods. For each split in the cross validation process, a new 'C' hyperparameter is set for the pipeline logistic regression model and can be seen in the visual below. Overfitting begins to occur after ~ 10**-1. 

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/overfitting_example_c_hyperparameter.png)

For logistic regression, a high ROC AUC value of ~0.73 is observed after hyperparameter tuning.

The variance of ROC curves for each split is displayed below for the range of 'C' hyperparameter values.

![Alt text](https://github.com/kcao22/credit_card_defaults/blob/main/Visualizations/variance_in_splits.png)

### Decision Tree & Random Forest

Next, a single decision tree model is explored. Cross validation with a range of 'max_depth' values ranging between 1 and 12 are explored. To prevent obvious scenarios of overfitting, the max leaves are considered relative to the maxiumum number of positive samples. Given that the 'max_depth' is limited to a level of 12. Given that there are  


In the 
This provides insight on ballpark ranges for optimizing each parameter and ensuring no overfitting occurs. Below is an example of overfitting for a parameter.

![alt text](https://github.com/kcao22/webscraping_ds_salaries/blob/main/images/max_depth%20overfitting.png "Outliers")

Recursive feature elimination for random forest is then used to reduce the number of features resulting from OneHotEncoding.

My best performing Random Forest model produced a MAE score of 14.58.

## Interactive Web Page Estimation Tool

As a final step, an interactive web page was produced using Gradio where a user can input job posting information. The web page then returns a predicted salary for the job posting. To see this in action, view the .gif at the top of README. If you would like to use the web interface yourself, please run gradio_gui.py in Python terminal to host the page locally.
![Alt Text](https://github.com/kcao22/webscraping_ds_salaries/blob/main/images/terminal_gradio.gif)
