### Proxy Finder:

$$
\text{Objective function} = \arg \max\limits_{c \in C} \left( (1 - \alpha) R^2_{fv \sim c} - \alpha \frac{1}{|O|} \sum_{o \in O} R^2_{o \sim c} \right)
$$

### Summary of the Proxy Finder Algorithm  

#### Objective:  
To identify the *k* best proxy variables for a target variable from a dataset that has that variable explicitly measured (df1) in a second dataset that does not (df2). A good proxy variable accurately predicts a given target while being orthogonal (i.e., uncorrelated) to specified control variables.   

#### Inputs:

 - **df1:** The dataset containing the target variable.
 - **df2:** The dataset we want to find a proxy for the target variable in.
 - **Target:** The target variable in df1 that we would like to find a proxy for in df2.
 - **Predictors:** The variables we will use to predict the target in df1 and df2. These variables must exist in both df1 and df2.
 - **Orthogonal Variables** The control variables that we would like the proxy to be uncorrelated with. Optionally, this can include all predictor variables to reduce direct influence of predictor variable variance on the predicted target variable.
 - **k**: (optional) Number of proxies for the algorithm to recommend. In unspecified, defaults to 1.
 - **Candidates:** (optional) All candidate proxy variables we'd like to check. If unspecified, defaults to all numerical variables in df2.
 - $\mathbf{\alpha}$: (optional) Orthogonality weight (between 0,1 inclusive). Defaults to 0.65.
   
#### Steps:
##### 1. Data Preprocessing:  
 - Rescale all numerical columns in both datasets (df1 and df2) to ensure values are between 0 and 1.
 - Remove any rows with insufficient predictive data.
##### 2. Model Training and Target Prediction:
 - Use predictive model (shallow Neural Network) to predict the target in df1 using the set of predictors. Save the learned weights. Confirm model performance on test (held-out) data.
 - Predict Item: Use the learned weights to predict the focal variable in df2.
##### 3. Proxy Selection:
 - **3.a: Initial Proxy Analysis:** For each variable that is a candidate proxy in df2:
   - Perform a regression analysis (OLS) to predict the target using the candidate proxy.
   - Store the R-squared value of the regression model.
 - **3.b: Orthogonality Constraint:** For each candidate proxy:
   - Calculate the R-squared value for the regression of each orthogonal variable on the candidate proxy.
   - Compute an orthogonality score as the mean of these R-squared values.
   - Adjust the initial R-squared value of the proxy by subtracting a weighted orthogonality score to obtain “Proxy Score”.
 - **3.c: Final Selection:** Sort the candidate proxies based on their Proxy Score and select the top proxies.

#### Output:
 - **Top Proxies:** Return a list of the top candidate proxies that best predict the focal variable while being orthogonal to the specified control variables.
 - **Proxy Scores and Confidence:** Print Proxy Scores and a confidence score based on neural net performance.
