#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import os
import calendar
import networkx as nx
from pandas.plotting import scatter_matrix, parallel_coordinates
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pylab as plt

import math
from sklearn.metrics import accuracy_score, roc_curve, auc
from dmba import regressionSummary, classificationSummary, liftChart, gainsChart
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
import statsmodels.formula.api as sm

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from dmba import plotDecisionTree, classificationSummary, regressionSummary

import statsmodels.api as sm
from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score

import statsmodels.formula.api as sm

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score


# In[2]:


retail = pd.read_csv('retailsales.csv')


# In[3]:


retail.head(20)


# In[4]:


# random_state is set to a defined value to get the same partitions when re-running the code
trainData= retail.sample(frac=0.6, random_state=1)
# assign rows that are not already in the training set, into validation 
validData = retail.drop(trainData.index)

print('Training   : ', trainData.shape)
print('Validation : ', validData.shape)
print()


# In[5]:


# alternative way using scikit-learn
trainData, validData = train_test_split(retail, test_size=0.40, random_state=1)
print('Training   : ', trainData.shape)
print('Validation : ', validData.shape)


# In[6]:


retail_ts = pd.Series(retail.retailsales.values, index=retail.month)


# In[8]:


plt.plot(retail_ts)


# In[9]:


plt.plot(retail_ts.index, retail_ts)
plt.xlabel('time')  # set x-axis label
plt.ylabel('Retailsales (in mil)')  # set y-axis label


# In[15]:


#2)Histogram

ax = retail.retailsales.hist()
ax.set_xlabel('retailsales')
ax.set_ylabel('count')

plt.show()


# In[11]:


#3)Boxplot

ax = retail.boxplot(column='retailsales')
ax.set_ylabel('retailsales')
plt.suptitle('')  # Suppress the titles
plt.title('')
	
plt.show()


# In[12]:


#4)Subplots

fig, axes = plt.subplots(nrows=1, ncols=4)
retail.boxplot(column='retailsales', ax=axes[0])
retail.boxplot (column='percapitaincome', ax=axes[1])
retail.boxplot (column='population', ax=axes[2])
retail.boxplot (column='unemployment', ax=axes[3])
for ax in axes:
    ax.set_xlabel('subplot')
plt.suptitle('')  # Suppress the overall title
plt.tight_layout()  # Increase the separation between the plots

plt.show()


# In[13]:


#5) Correlation

corr = retail.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


# Change the colormap to a divergent scale and fix the range of the colormap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, cmap="RdBu")

# Include information about values
fig, ax = plt.subplots()
fig.set_size_inches(7, 4)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu", center=0.7, vmin=-1.0, vmax=1.0, ax=ax)

plt.show()



# Color the points by the value
retail.plot.scatter(x='retailsales', y='percapitaincome')


# In[14]:


#Compute mean, standard dev., min, max, median, length, and missing values for all variables
pd.DataFrame({'mean': retail.mean(),
              'sd': retail.std(),
              'min': retail.min(),
              'max': retail.max(),
              'median': retail.median(),
              'length': len(retail),
              'miss.val': retail.isnull().sum(),
             })


# In[16]:


#Correlation matrix

retail.corr().round(2)


# In[17]:



#PCA analysis

retail = pd.read_csv('retailsales.csv')
pcs = PCA(n_components=2)
pcs.fit(retail[['percapitaincome', 'population']])

pcsSummary = pd.DataFrame({'Standard deviation': np.sqrt(pcs.explained_variance_),
                           'Proportion of variance': pcs.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)})
pcsSummary = pcsSummary.transpose()
pcsSummary.columns = ['PC1', 'PC2']
pcsSummary.round(4)


# In[20]:


#The components_ field of pcs gives the individual components. 
#The columns in this matrix are the principal components PC1, PC2.
#The rows are variables in the order they are found in the input matrix, calories and rating. Below is code for weight#


pcsComponents_df = pd.DataFrame(pcs.components_.transpose(), columns=['PC1', 'PC2'], 
                                index=['percapitaincome', 'population'])
pcsComponents_df


# In[21]:


#Use the transform method to get the scores.

scores = pd.DataFrame(pcs.transform(retail[['percapitaincome', 'population']]), 
                      columns=['PC1', 'PC2'])
scores.head()


#Perform a principal component analysis of the whole table ignoring the first non-numerical column.

pcs = PCA()
pcs.fit(retail.iloc[:, 2:6].dropna(axis=0))
pcsSummary_df = pd.DataFrame({'Standard deviation': np.sqrt(pcs.explained_variance_),
                           'Proportion of variance': pcs.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)})
pcsSummary_df = pcsSummary_df.transpose()
pcsSummary_df.columns = ['PC{}'.format(i) for i in range(1, len(pcsSummary_df.columns) + 1)]
pcsSummary_df.round(4)


pcsComponents_df = pd.DataFrame(pcs.components_.transpose(), columns=pcsSummary_df.columns, 
                                index=retail.iloc[:, 2:6].columns)
pcsComponents_df.iloc[:,:6]


# In[23]:


#Multiple linear Regression

retail = pd.read_csv('retailsales.csv')

predictors = ['percapitaincome', 'population', 'unemployment', 'inventory', 'yoygtenp', 'inventorygrowthabovefive', 'percapitagrowthabove']
outcome = 'retailsales'

X = pd.get_dummies(retail[predictors], drop_first=True)
Y = retail[outcome]

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.4, random_state=1)

# train linear regression model
reg = LinearRegression()
reg.fit(train_X, train_Y)


# In[24]:


# print coefficients
print('intercept ', reg.intercept_)
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': reg.coef_}))


# In[25]:


# evaluate performance
# training
regressionSummary(train_Y, reg.predict(train_X))
# validation
regressionSummary(valid_Y, reg.predict(valid_X))


# In[26]:


pred_error_train = pd.DataFrame({
    'residual': train_Y - reg.predict(train_X), 
    'data set': 'training'
})
pred_error_valid = pd.DataFrame({
    'residual': valid_Y - reg.predict(valid_X), 
    'data set': 'validation'
})
boxdata_df = pred_error_train.append(pred_error_valid, ignore_index=True)

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(9, 4)
common = {'bins': 100, 'range': [-30000, 30000]}
pred_error_train.hist(ax=axes[0], **common)
pred_error_valid.hist(ax=axes[1], **common)
boxdata_df.boxplot(ax=axes[2], by='data set')

axes[0].set_title('training')
axes[1].set_title('validation')
axes[2].set_title(' ')
axes[2].set_ylim(-30000, 30000)
plt.suptitle('Prediction errors') 
plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.35)

plt.show()


# In[27]:


#Adjusted R2 BIC and AIC

pred_Y = reg.predict(train_X)

print('adjusted r2 : ', adjusted_r2_score(train_Y, pred_Y, reg))
print('AIC : ', AIC_score(train_Y, pred_Y, reg))
print('BIC : ', BIC_score(train_Y, pred_Y, reg))




# Use predict() to make predictions on a new set
reg_lm_pred = reg.predict(valid_X)

result = pd.DataFrame({'Predicted': reg_lm_pred, 'Actual': valid_Y,
                       'Residual': valid_Y - reg_lm_pred})
print(result.head(20))

# Compute common accuracy measures
regressionSummary(valid_Y, reg_lm_pred)


#determine residuals and create histogram

reg_lm_pred = reg.predict(valid_X)
all_residuals = valid_Y - reg_lm_pred

# Determine the percentage of datapoints with a residual in [-24000, 24000] = approx. 75\%
print(len(all_residuals[(all_residuals > -24000) & (all_residuals < 24000)]) / len(all_residuals))

ax = pd.DataFrame({'Residuals': all_residuals}).hist(bins=25)

plt.tight_layout()
plt.show()


# In[28]:


#Exhaustive search for reducing predictors - ranks variables and need and gives scores

def train_model(variables):
    model = LinearRegression()
    model.fit(train_X[variables], train_Y)
    return model

def score_model(model, variables):
    pred_Y = model.predict(train_X[variables])
    # we negate as score is optimized to be as low as possible
    return -adjusted_r2_score(train_Y, pred_Y, model)

allVariables = train_X.columns
results = exhaustive_search(allVariables, train_model, score_model)

data = []
for result in results:
    model = result['model']
    variables = result['variables']
    AIC = AIC_score(train_Y, model.predict(train_X[variables]), model)
    
    d = {'n': result['n'], 'r2adj': -result['score'], 'AIC': AIC}
    d.update({var: var in result['variables'] for var in allVariables})
    data.append(d)
pd.set_option('display.width', 100)
print(pd.DataFrame(data, columns=('n', 'r2adj', 'AIC') + tuple(sorted(allVariables))))
pd.reset_option('display.width')


# In[29]:


#backward elimination

def train_model(variables):
    model = LinearRegression()
    model.fit(train_X[variables], train_Y)
    return model

def score_model(model, variables):
    return AIC_score(train_Y, model.predict(train_X[variables]), model)

best_model, best_variables = backward_elimination(train_X.columns, train_model, score_model, verbose=True)

print(best_variables)

regressionSummary(valid_Y, best_model.predict(valid_X[best_variables]))


# In[30]:


# Forward selection

# The initial model is the constant model - this requires special handling
# in train_model and score_model
def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(train_X[variables], train_Y)
    return model

def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(train_Y, [train_Y.mean()] * len(train_Y), model, df=1)
    return AIC_score(train_Y, model.predict(train_X[variables]), model)

best_model, best_variables = forward_selection(train_X.columns, train_model, score_model, verbose=True)

print(best_variables)


# In[31]:


#stepwise_selection

best_model, best_variables = stepwise_selection(train_X.columns, train_model, score_model, verbose=True)

print(best_variables)


# In[37]:


# Regularization (Shrinkage Models)

lasso = Lasso(normalize=True, alpha=1)
lasso.fit(train_X, train_Y)
regressionSummary(valid_Y, lasso.predict(valid_X))

lasso_cv = LassoCV(normalize=True, cv=5)

lasso_cv.fit(train_X, train_Y)
regressionSummary(valid_Y, lasso_cv.predict(valid_X))
print('Lasso-CV chosen regularization: ', lasso_cv.alpha_)
print(lasso_cv.coef_)

ridge = Ridge(normalize=True, alpha=1)
ridge.fit(train_X, train_Y)
regressionSummary(valid_Y, ridge.predict(valid_X))

bayesianRidge = BayesianRidge(normalize=True)
bayesianRidge.fit(train_X, train_Y)
regressionSummary(valid_Y, bayesianRidge.predict(valid_X))
print('Bayesian ridge chosen regularization: ', bayesianRidge.lambda_ / bayesianRidge.alpha_)


# In[42]:


reg = LinearRegression()
reg.fit(train_X, train_Y)

pd.DataFrame({'features': train_X.columns, 'coefficient': reg.coef_, 
             'lasso': lasso.coef_,  'lassoCV': lasso_cv.coef_, 'bayesianRidge': bayesianRidge.coef_})


# run a linear regression of sales on the remaining predictors in the training set
train_df = train_X.join(train_Y)

predictors = train_X.columns
formula = 'retailsales ~ ' + ' + '.join(predictors)

reg = sm.ols(formula=formula, data=train_df).fit()
print(reg.summary())


# In[43]:


#KNN

retail = pd.read_csv('retailsales1.csv')

retail['Number'] = retail.index + 1
retail

#train and valid KNN

trainData, validData = train_test_split(retail, test_size=0.4, random_state=26)
print(trainData.shape, validData.shape)
newretail = pd.DataFrame([{'inventorygrowth': 5, 'populationgrowth': 1.1}])
newretail

#scatterplot
fig, ax = plt.subplots()

subset = trainData.loc[trainData['yoygtenp']=='NO']
ax.scatter(subset.inventorygrowth, subset.populationgrowth, marker='o', label='NOGROWTH', color='C1')

subset = trainData.loc[trainData['yoygtenp']=='YES']
ax.scatter(subset. inventorygrowth, subset. populationgrowth, marker='D', label='GROWTH', color='C0')

ax.scatter(newretail.inventorygrowth, newretail.populationgrowth, marker='*', label='Newretail', color='black', s=150)

plt.xlabel('inventorygrowth')  # set x-axis label
plt.ylabel('populationgrowth')  # set y-axis label
for _, row in trainData.iterrows():
    ax.annotate(row.Number, (row.inventorygrowth + 2, row. populationgrowth))
    
handles, labels = ax.get_legend_handles_labels()
ax.set_xlim(0, 15)
ax.set_ylim(0, 2)
ax.legend(handles, labels, loc=4)

plt.show()



def plotDataset(ax, data, showLabel=True, **kwargs):
    subset = data.loc[data['yoygtenp']=='NO']
    ax.scatter(subset.inventorygrowth, subset.populationgrowth, marker='o', label='NO' if showLabel else None, color='C1', **kwargs)

    subset = data.loc[data['yoygtenp']=='YES']
    ax.scatter(subset.inventorygrowth, subset.populationgrowth, marker='D', label='YES' if showLabel else None, color='C0', **kwargs)

    plt.xlabel('inventorygrowth')  # set x-axis label
    plt.ylabel('populationgrowth')  # set y-axis label
    for _, row in data.iterrows():
        ax.annotate(row.Number, (row.inventorygrowth + 2, row.populationgrowth))

fig, ax = plt.subplots()

plotDataset(ax, trainData)
plotDataset(ax, validData, showLabel=False, facecolors='none')

ax.scatter(newretail.inventorygrowth, newretail.populationgrowth, marker='*', label='Newretail ', color='black', s=150)

plt.xlabel('inventorygrowth')  # set x-axis label
plt.ylabel('populationgrowth')  # set y-axis label
    
handles, labels = ax.get_legend_handles_labels()
ax.set_xlim(1, 15)
ax.legend(handles, labels, loc=4)

plt.show()



#Initialize normalized training, validation, and complete data frames. Use the training data to learn the transformation.


scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['inventorygrowth', 'populationgrowth']])  # Note the use of an array of column names

# Transform the full dataset
retailNorm = pd.concat([pd.DataFrame(scaler.transform(retail[['inventorygrowth', 'populationgrowth']]), 
                                    columns=['zinventorygrowth', 'zpopulationgrowth']),
                       retail[['yoygtenp', 'Number']]], axis=1)
trainNorm = retailNorm.iloc[trainData.index]
validNorm = retailNorm.iloc[validData.index]
newretailNorm = pd.DataFrame(scaler.transform(newretail), columns=['zinventorygrowth', 'zpopulationgrowth'])


# In[44]:



#Use k-nearest neighbour

knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zinventorygrowth', 'zpopulationgrowth']])
distances, indices = knn.kneighbors(newretailNorm)
print(trainNorm.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element


# In[45]:


#accuracy

train_X = trainNorm[['zinventorygrowth', 'zpopulationgrowth']]
train_y = trainNorm['yoygtenp']
valid_X = validNorm[['zinventorygrowth', 'zpopulationgrowth']]
valid_y = validNorm['yoygtenp']

# Train a classifier for different values of k
results = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
    results.append({
        'k': k,
        'accuracy': accuracy_score(valid_y, knn.predict(valid_X))
    })

# Convert results to a pandas data frame
results = pd.DataFrame(results)
print(results)


# Retrain with full dataset---KNN
retail_X = retailNorm[['zinventorygrowth', 'zpopulationgrowth']]
retail_y = retailNorm['yoygtenp']
knn = KNeighborsClassifier(n_neighbors=4).fit(retail_X, retail_y)
distances, indices = knn.kneighbors(newretailNorm)
print(knn.predict(newretailNorm))
print('Distances',distances)
print('Indices', indices)
print(retailNorm.iloc[indices[0], :])


# In[46]:


#NAIVE BAYES

retail = pd.read_csv('retailsales2.csv')
retail.head()

# convert to categorical
retail.inventorygrowthabovefive = retail.inventorygrowthabovefive.astype('category')
retail.populationgrowthabove = retail.populationgrowthabove.astype('category')


predictors = ['inventorygrowthabovefive', 'populationgrowthabove']
outcome = 'yoygtenp'

X = pd.get_dummies(retail[predictors])
y = (retail[outcome] == 'NO').astype(int)
classes = ['YES', 'NO']

# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=1)

# run naive Bayes
retail_nb = MultinomialNB(alpha=0.01)
retail_nb.fit(X_train, y_train)

# predict probabilities
predProb_train = retail_nb.predict_proba(X_train)
predProb_valid = retail_nb.predict_proba(X_valid)

# predict class membership
y_valid_pred = retail_nb.predict(X_valid)
y_train_pred = retail_nb.predict(X_train)


retail.inventorygrowthabovefive = retail.inventorygrowthabovefive.astype('category')
retail.populationgrowthabove = retail.populationgrowthabove.astype('category')

retail['yoygtenp'] = retail['yoygtenp'].astype('category')


predictors = ['inventorygrowthabovefive', 'populationgrowthabove']
outcome = 'yoygtenp'

X = pd.get_dummies(retail[predictors])
y = retail['yoygtenp']
classes = list(y.cat.categories)

# split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=1)

# run naive Bayes
retail_nb = MultinomialNB(alpha=0.01)
retail_nb.fit(X_train, y_train)

# predict probabilities
predProb_train = retail_nb.predict_proba(X_train)
predProb_valid = retail_nb.predict_proba(X_valid)

# predict class membership
y_valid_pred = retail_nb.predict(X_valid)
y_train_pred = retail_nb.predict(X_train)


#First construct a frequency table and then convert it to the propability table


# split the original data frame into a train and test using the same random_state
train_df, valid_df = train_test_split(retail, test_size=0.4, random_state=1)

pd.set_option('precision', 4)
# probability of flight status
print(train_df['yoygtenp'].value_counts() / len(train_df))
print()

for predictor in predictors:
    # construct the frequency table
    df = train_df[['yoygtenp', predictor]]
    freqTable = df.pivot_table(index='yoygtenp', columns=predictor, aggfunc=len)

    # divide each row by the sum of the row to get conditional probabilities
    propTable = freqTable.apply(lambda x: x / sum(x), axis=1)
    print(propTable)
    print()
pd.reset_option('precision')



# Subset a specific set/ predicting for new data
df = pd.concat([pd.DataFrame({'actual': y_valid, 'predicted': y_valid_pred}),
                pd.DataFrame(predProb_valid, index=y_valid.index)], axis=1)
mask = ((X_valid.inventorygrowthabovefive_YES == 1) & (X_valid. populationgrowthabove_YES == 1))

print(df[mask])



#Confusionmatrix
classificationSummary(y_train, y_train_pred, class_names=classes) 

print()

classificationSummary(y_valid, y_valid_pred, class_names=classes)


# In[47]:


#Regressiontree

get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary


retail = pd.read_csv('retailsales.csv')

predictors = ['percapitaincome', 'population', 'unemployment', 'inventory', 'yoygtenp', 'inventorygrowthabovefive', 'percapitagrowthabove']
outcome = 'retailsales'

X = pd.get_dummies(retail[predictors], drop_first=True)
y = retail[outcome]

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

# user grid search to find optimized tree
param_grid = {
    'max_depth': [5, 10, 15, 20, 25], 
    'min_impurity_decrease': [0, 0.001, 0.005, 0.01], 
    'min_samples_split': [10, 20, 30, 40, 50], 
}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Initial parameters: ', gridSearch.best_params_)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
    'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008], 
    'min_samples_split': [14, 15, 16, 18, 20, ], 
}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved parameters: ', gridSearch.best_params_)

regTree = gridSearch.best_estimator_


regressionSummary(train_y, regTree.predict(train_X))
regressionSummary(valid_y, regTree.predict(valid_X))

#plot reg tree

plotDecisionTree(regTree, feature_names=train_X.columns)
plotDecisionTree(regTree, feature_names=train_X.columns, rotate=True)


# In[48]:


#Classification Tree
retail = pd.read_csv('retailsales1.csv')

predictors = ['inventorygrowth', 'populationgrowth']
outcome = 'yoygtenp'

X = pd.get_dummies(retail[predictors], drop_first=True)
y = retail[outcome]

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_X, train_y)

plotDecisionTree(fullClassTree, feature_names=train_X.columns)#Confusion matrix on classification tree

classificationSummary(train_y, fullClassTree.predict(train_X))
classificationSummary(valid_y, fullClassTree.predict(valid_X))


# Five-fold cross-validation of the full decision tree classifier
treeClassifier = DecisionTreeClassifier()

scores = cross_val_score(treeClassifier, train_X, train_y, cv=5)
print('Accuracy scores of each fold: ', [f'{acc:.3f}' for acc in scores])
print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')


# In[49]:


#ensembles.

#code for bagging and boosting trees

get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from dmba import classificationSummary


retail = pd.read_csv('retailsales2.csv')


predictors = ['inventorygrowthabovefive', 'populationgrowthabove']
outcome = 'yoygtenp'


# split into training and validation

X = pd.get_dummies(retail[predictors], drop_first=True)
y = retail[outcome]


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, random_state=3)

#singletree

defaultTree = DecisionTreeClassifier(random_state=1)
defaultTree.fit(X_train, y_train)

classes = defaultTree.classes_
classificationSummary(y_valid, defaultTree.predict(X_valid), class_names=defaultTree.classes_)


#bagging
bagging = BaggingClassifier(DecisionTreeClassifier(random_state=1), 
                            n_estimators=100, random_state=1)
bagging.fit(X_train, y_train)

classificationSummary(y_valid, bagging.predict(X_valid), class_names=classes)


#boosting
boost = AdaBoostClassifier(DecisionTreeClassifier(random_state=1), n_estimators=100, random_state=1)
boost.fit(X_train, y_train)

classificationSummary(y_valid, boost.predict(X_valid), class_names=classes)


# In[51]:


#Logistics regression

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score

#code for fitting a logistic regression model

retail = pd.read_csv('retailsales.csv')
retail.head()

# convert to categorical
retail.inventorygrowthabovefive = retail.inventorygrowthabovefive.astype('category')
retail.percapitagrowthabove = retail.percapitagrowthabove.astype('category')


predictors = ['inventorygrowthabovefive', 'percapitagrowthabove']
outcome = 'yoygtenp'

X = pd.get_dummies(retail[predictors])
y = (retail[outcome] == 'NO').astype(int)
classes = ['YES', 'NO']

# partition data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

# fit a logistic regression (set penalty=l2 and C=1e42 to avoid regularization)
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
logit_reg.fit(train_X, train_y)

print('intercept ', logit_reg.intercept_[0])
print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns).transpose())
print()
print('AIC', AIC_score(valid_y, logit_reg.predict(valid_X), df = len(train_X.columns) + 1))


#code for using logistic regression to generate predicted probabilities

logit_reg_pred = logit_reg.predict(valid_X)
logit_reg_proba = logit_reg.predict_proba(valid_X)
logit_result = pd.DataFrame({'actual': valid_y, 
                             'p(0)': [p[0] for p in logit_reg_proba],
                             'p(1)': [p[1] for p in logit_reg_proba],
                             'predicted': logit_reg_pred })

# display four different cases
interestingCases = [27, 93, 21, 70]
print(logit_result.loc[interestingCases])


#confusion matrix
classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))


# In[ ]:




