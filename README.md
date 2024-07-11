# Summary of the Proxy Finder Algorithm
### Objective:
To identify a proxy variable from a dataset that can accurately predict a given item (e.g., status_threat) while being orthogonal (i.e., uncorrelated) to specified control variables (e.g., authoritarianism, christian_nationalism, etc.).
### Steps:
#### Data Preparation:  

 - Normalize Data: Rescale all numerical columns in both datasets (df1 and df2) to ensure values are between 0 and 1.  

#### Model Training:  

 - Train Model: Use multiple regression (Ordinary Least Squares, OLS) to predict the target item in df1 using a set of predictors. This model is used to generate predicted values of the item for df2.
 - Predict Item: Use the trained model to predict the target item in df2.  

#### Proxy Selection:  

 - Initial Proxy Analysis: For each candidate proxy in df2:
     - Perform a regression analysis to predict the item using the candidate proxy.
     - Store the R-squared value and p-value of the regression model.
     - Remove any candidate proxies that do not provide enough predictors.
     - Orthogonality Check: For each remaining candidate proxy:
         - Calculate the R-squared value for the regression of each orthogonal variable on the candidate proxy.
         - Compute an orthogonality score as the mean of these R-squared values.
     - Adjust the initial R-squared value of the proxy by subtracting a weighted orthogonality score.
     - Final Selection: Sort the candidate proxies based on their adjusted R-squared values (or scores) and select the top proxies.  

#### Output:
 - Top Proxies: Return the top candidate proxies that best predict the item while being orthogonal to the specified control variables.
