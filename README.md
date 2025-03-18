# Proxy Finder: Proxy Selection in Divergent Data Sets by Methodological Prediction and Correlation Reduction (MPSA 2025)

This repository contains code for the paper:
*ProxyFinder: Proxy Selection in Divergent Data Sets by Methodological Prediction and Correlation Reduction*, Kirin Danek, James Daniels, Howard Lavine. 2025.

$$
\text{Objective function} = \arg \max\limits_{c \in C} \left( (1 - \alpha) R^2_{\hat{t} \sim c} - \alpha \frac{1}{|O|} \sum_{o \in O} R^2_{o \sim c} \right)
$$

### Summary of the Proxy Finder Algorithm  

#### Objective:  
To identify the *k* best proxy variables for a target variable from a dataset that has that variable explicitly measured (df1) in a second dataset that does not (df2). A good proxy variable accurately predicts a given target while being uncorrelated with specified control variables.   

#### Inputs:

 - **df1:** The dataset containing the target variable.
 - **df2:** The dataset we want to find a proxy for the target variable in.
 - **Target:** The target variable in df1 that we would like to find a proxy for in df2.
 - **Predictors:** The variables we will use to predict the target in df1 and df2. These variables must exist in both df1 and df2.
 - **Penalty Variables (O)** The control variables that we would like the proxy to be uncorrelated with. Optionally, this can include all predictor variables to reduce direct influence of predictor variable variance on the predicted target variable.
 - **k**: (optional) Number of proxies for the algorithm to recommend. In unspecified, defaults to 1.
 - **Candidates (C):** (optional) All candidate proxy variables we'd like to check. If unspecified, defaults to all numerical variables in df2.
 - **Penalty Weight ($\alpha$):** (optional) Penalty weight (between 0,1 inclusive). Defaults to 0.65.
   
#### Steps:
##### 1. Data Preprocessing:  
 - Rescale all numerical columns in both datasets (df1 and df2) to ensure values are between 0 and 1.
 - Remove any rows with insufficient predictive data.
##### 2. Model Training and Target Prediction:
 - Use predictive model to predict the target in df1 using the set of predictors. Save the learned weights. Confirm model performance on test (held-out) data.
 - Impute Item: Use the learned weights to impute the focal variable in df2.
##### 3. Proxy Selection:
 - **3.a: Initial Proxy Analysis:** For each variable that is a candidate proxy in df2:
   - Perform a regression analysis (OLS) regressing the predicted target column ($\hat{t}$) on the candidate proxy.
   - Store the R-squared value of the regression model.
 - **3.b: Score Penalization:** For each candidate proxy:
   - Calculate the R-squared value for the regression of each orthogonal variable on the candidate proxy.
   - Compute an penalization score as the mean of these R-squared values.
   - Adjust the initial R-squared value of the proxy by subtracting a weighted penalization score to obtain “Proxy Score”.
 - **3.c: Final Selection:** Sort the candidate proxies based on their Proxy Score and select the top proxies.

#### Output:
 - **Top Proxies:** Return a list of the top candidate proxies that best predict the focal variable while being orthogonal to the specified control variables.
 - **Proxy Scores:** Print Proxy Scores.

### Authors of the code
- [Kirin Danek](kirin2d2.github.io) - kd9132@princeton.edu, Undergraduate Student, Princeton University
- James Daniels - jamesdaniels@princeton.edu, Undergraduate Student, Princeton University
