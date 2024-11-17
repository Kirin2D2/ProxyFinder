#!/usr/bin/env python
# coding: utf-8

# # version of proxyFinder algorithm using neural network to make predictions

# In[75]:


import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[2]:


import os 
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))


# In[76]:


def proxy_finder_validate(item, candidates, df1, df2, predictors, orthogonal_vars):

    # validate proxies and st item
    assert item in df1.columns, f'AssertionError: item {item} not in df1.columns'

    assert predictors, f'AssertionError: missing predictors. If you would prefer to not specify predictors, do not pass in a variable.'
    
    for c in predictors:
        assert c in df1.columns, f'AssertionError: predictor {c} not in df1.columns'
        assert c in df2.columns, f'AssertionError: predictor {c} not in df2.columns' # we need same variable in second dataset  
        assert c in df1.select_dtypes(include=['number']).columns, f'predictor {c} is not a numeric column in df1'   
        assert c in df2.select_dtypes(include=['number']).columns, f'predictor {c} is not a numeric column in df2'    
    
    for c in candidates:
        assert c in df2.columns, f'AssertionError: candidate {c} not in df2.columns'
        
    if (orthogonal_vars != None):
        for c in orthogonal_vars:
            assert c in df2.columns, f'AssertionError: orthogonal variable {c} not in df2.columns'
                


# In[79]:


# return a new df that is a copy of df, with: rescale all columns to be
#  between 0 and 1, inclusive. Drop any non-numeric columns. Drop any 
# rows that are missing at least one predictor. 
def data_rescale(df, predictors):
    df = df.copy() # preserve immutability

    # Select only the numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # drop any rows that are missing at least one predictor
    df = df.dropna(subset=predictors)
    # print('the dataframe we\'re rescaling is size: ') # debug
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform it
    scaled_values = scaler.fit_transform(df[numeric_cols])

    # Create a new DataFrame with the scaled values, maintaining the original column names
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
    
    return scaled_df


# In[78]:


# Neural network definition
def build_nn_model(input_dim, learning_rate=0.001, l2_lambda=0.001):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        Dropout(0.5),  
        Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        Dropout(0.5),  
        Dense(1, kernel_regularizer=l2(l2_lambda))
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# In[80]:


# return a trained neural network to predict df[item] using df[predictors_df1]
# report error and crash if predictors don't predict item
def train_nn_model(X_train, y_train, input_dim, epochs=100, learning_rate=0.001, l2_lambda=0.001):
    model = build_nn_model(input_dim, learning_rate, l2_lambda)
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
    return model


# In[109]:


# get predictions from the neural network. Takes in
def get_nn_predictions(df_train, df_test, predictors, target, epochs=100, learning_rate=0.001, l2_lambda=0.001):
    
    # split data for training and testing. 
    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(df_train[predictors].to_numpy(), df_train[target].to_numpy(), test_size=0.2, random_state=42)
    X_test = df_test[predictors].to_numpy()

    # train network and get predictions
    model = train_nn_model(X_train_train, y_train_train, len(predictors), epochs, learning_rate, l2_lambda)
    predictions = model.predict(X_test)

    # exit if correlation between predictions and item is bad
    mse = mean_squared_error(model.predict(X_train_test), y_train_test)
    print(f"Debug statement: MSE = {mse}") ####DEBUG
    print(f'Confidence level: {(int)((1 - (mse / 0.036)) * 100)}% (results under 25% suppressed)')
    if (mse > 0.03):
        print('Input Error: Predictors cannot predict {target} in df1', file=sys.stderr)
        print('Aborting program')
        sys.exit(-1)

   # print(f"Predictions before flattening: {predictions[:10]}") #DEBUG
   # print('predictions after flattening: ', predictions.flatten()[:10])#DEBUG

    return predictions.flatten()


# In[114]:


#final 3 parameters for debugging/fine tuning
def proxy_finder(df_train, df_test, target, predictors, num_proxies=1, orth_weight=0.65, candidates=None, orthogonal_vars=None, epochs=100, learning_rate=0.001, l2_lambda=0.001):
    if candidates is None:
        candidates = list(df_test.select_dtypes(include='number').columns)
    

    proxy_finder_validate(target, candidates, df_train, df_test, predictors, orthogonal_vars)

    #print(f"Predictors: {predictors}") #DEBUGDEBUGDEBUG------------------------------------------------------------
    #print(f"Candidates: {candidates}")

    # Predict status threat scores in df_test
    df_train = data_rescale(df_train, predictors)
    df_test = data_rescale(df_test, predictors)
    df_train = df_train.dropna(subset=target)

    # Check for NaN entries in the specified columns DEBUG
    #for index, row in df_train.iterrows():
     #   if row[target].isnull().any():
      #      print(f"Entry is NaN in row {index}")
        
   # print(df_train.head) ## debug
  #  print(df_test.head)
    predicted_scores = get_nn_predictions(df_train, df_test, predictors, target, epochs, learning_rate, l2_lambda)
    
    df_test['predicted_status_threat'] = predicted_scores
    #print(f"Predicted scores: {predicted_scores[:10]}")  #DEBUG DEBUG------------------------------------------------------------ 

    results = {}
    
    for c in candidates:
        candset = df_test[[c, 'predicted_status_threat']].copy().dropna()
        if candset.empty:
            continue
        
        pred_scores = candset['predicted_status_threat']
        candcol = candset[c]

        X_pred = sm.add_constant(candcol)
        model_pred = sm.OLS(pred_scores, X_pred).fit()
        results[c] = {
            'R_squared': model_pred.rsquared,
            'p_value': model_pred.pvalues[1],
            'coef': model_pred.params[1]
        }
        #print(f"candidate {c}: Results: {results}")  # Debug statement------------------------------------------------------------ 
  
    best_proxies = []

    if orthogonal_vars:
        orth_score = {}
        for c in candidates:
            candset = df_test[[c, 'predicted_status_threat']].copy().dropna()
            pred_scores = candset['predicted_status_threat']
            candcol = candset[c]
        
            X = sm.add_constant(candcol)
            temp_orth_scores = []
            for orth_var in orthogonal_vars:
                orthset = df_test[[orth_var]].copy().dropna()
                common_indices = candset.index.intersection(orthset.index)
                if common_indices.empty:
                    continue
                orth_col = orthset.loc[common_indices, orth_var]
                candcol_common = candset.loc[common_indices, c]

                X_common = sm.add_constant(candcol_common)
                model = sm.OLS(orth_col, X_common).fit()
                temp_orth_scores.append(model.rsquared)
            
            if temp_orth_scores:
                orth_score[c] = sum(temp_orth_scores) / len(temp_orth_scores)
            else:
                orth_score[c] = 0
        
        proxy_scores = {}
        for c in candidates:
            try:
                proxy_scores[c] = (c, (1 - orth_weight) * results[c]['R_squared'] - orth_weight * orth_score[c])
            except KeyError as e:
                continue
        
        sorted_results = sorted(proxy_scores.values(), key=lambda x: x[1], reverse=True)
        
        for i in range(min(num_proxies, len(sorted_results))):
            proxy, score = sorted_results[i]
            best_proxies.append(proxy)
            print(f"Proxy {i+1} for {target}: {proxy} with score: {score}")
    else: 
        sorted_results = sorted(results.items(), key=lambda x: (-x[1]['R_squared'], x[1]['p_value']))
    
        for i in range(min(num_proxies, len(sorted_results))):
            proxy, metrics = sorted_results[i]
            best_proxies.append(proxy)
            print(f"Proxy {i+1} for {target}: {proxy} with R_squared: {metrics['R_squared']} and p_value: {metrics['p_value']}")
    
    return best_proxies


# In[112]:


import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated") # I should probably actually fix this one so it doesn't break with future updates


# Suppress numpy invalid operation warnings
np.seterr(invalid='ignore')

datafile_train =  r"C:\Users\kirin\OURSIP_summer24\temp_yougov.dta"
datafile_test =  r"C:\Users\kirin\OURSIP_summer24\temp_anes.dta"
df_train = pd.read_stata(datafile_train)
df_test = pd.read_stata(datafile_test)



# In[113]:


target = 'christian_nationalism'  # The target variable in the training set
predictors = [
                   'turnout20post', #not numeric: either manually convert to numbers or use convert_categoricals=true
                   'presvote20post',
                   'housevote20post',
                   'senvote20post',
                   #'trump_presidential_approval',
                   'pff_jb',
                   'pff_dt',
                   'ideo7',
                   'pid7',
                  # 'para_social_grid_2',
                   'election_fairnness',
                   #'cab_b', w1, w2, w3
                   'educ',
                   #'race',
                   'hispanic',
                   'partisan_violence',
                   'immigrant_citizenship',
                   'immigrant_deport',
                   'auth_grid_1',
                   'auth_grid_3',
                   'auth_grid_2',
                   'faminc_new'
                  # 'wc_together', w1, w2, w3
                   #'wc_jobs',
                   #'racial_id',
                   #'hardworkingvlazy',
                   #'pronenot_violence',
                   #'group_disc_black',
                   #'group_disc_hispanic',
                   #'group_disc_white'
                   ]  # Predictors in both training and testing sets




best_proxies = proxy_finder(df_train, df_test, target, predictors, num_proxies=20)
#print(best_proxies)


# In[85]:


import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated") # I should probably actually fix this one so it doesn't break with future updates


# Suppress numpy invalid operation warnings
np.seterr(invalid='ignore')

# Example usage: Clearly, the best proxy for status threat should be status threat & related items. 
datafile_train =  r"C:\Users\kirin\Downloads\W1_W2_W3_Merged_saved (1).dta"
datafile_test =  r"C:\Users\kirin\Downloads\anes2020\anes_timeseries_2020_stata_20220210.dta"
df_train = pd.read_stata(datafile_train)
df_test = pd.read_stata(datafile_test)

target = 'status_threat'  # The target variable in the training set
predictors = [
                   'psc1_W1_01',
                   'christian_nationalism',
                   'authoritarianism',
                   'race_resent',
                   'party_ID',
                   'ideology',
                   'age501',
                   'education']  # Predictors in both training and testing sets
orthogonal_vars = ['psc1_W1_01',
                   'christian_nationalism',
                   'authoritarianism',
                   'race_resent',
                   'party_ID',
                   'ideology',
                   'age501',
                   'education']

best_proxies = proxy_finder(df_train, df_test, target, predictors, orthogonal_vars=orthogonal_vars, num_proxies=20)
#print(best_proxies)

