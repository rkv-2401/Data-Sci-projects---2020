import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = str(input("Enter the file name of the dataset : "))
dataset = pd.read_csv(filename)

print("These are the available features in the dataset : \n", dataset.columns)
dataset.dtypes

#Missing values

missing_values = dataset.isna().sum()
print("No. of missing values is : \n", missing_values)
#Rows where there are missing values
print("\n", dataset[dataset.isna().any(axis = 1)])

#Specific to this dataset
dataset['TotalCharges'] = pd.to_numeric(dataset.TotalCharges, errors = 'coerce')
dataset.dtypes

#Invalid values - after filling in missing values and converting 
invalid_values = dataset.isnull().sum()
print("The number of invalid data values is : ", invalid_values)
#The rows where the condition is true
print("\n", dataset[dataset.isna().any(axis = 1)])

numeric_columns = list(dataset.select_dtypes(include=['int64','float64']))

#Missing value function - need to find a dataset with missing values and create this function
def missing_function(dataset, user_input) :
    #print("Your choice : ", user_input)
    if (user_input == 1) :
        dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())
    elif (user_input == 2) :
        dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].median())
    elif (user_input == 3) :
        dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mode())
    elif (user_input == 4) :
        dataset[numeric_columns] = dataset[numeric_columns].dropna()
        
    print("\n", dataset.isna().sum())    

#Prompting user for treatment of missing, invalid and outlier data.
#print("No. of missing values : ", missing_values)
#print("\n No. of invalid values : ", invalid_values)
#print("\n No. of outliers : ", outlier_sum)
print("\n\nHow do you want to treat the missing values?")
print('''1. Fill with mean
2. Fill with median
3. Fill with mode
4. Drop''')
user_input = int(input("Enter your choice : "))
missing_function(dataset, user_input)

##How do I differentiate between missing and invalid data when using Pandas?

def outlier_treatment(x, user_input2) :
    index = len(x)
    for i in range(0, index) :
        if ((x[i] <= Q1) | (x[i] >= Q3)):
            print("{} is an outlier \n".format(x[i]))
            if (user_input2 == 1): 
                    x[i] = col_name.mean
            if (user_input2 == 2) :
                    if x[i] > Q3:
                        x[i] = Q3
                    elif x[i] < Q1:
                        x[i] = Q1
            print("{}".format(x[i]))
            #Later on add median and random sampling as well.

#EDA - this should loop for every numerical column
for i in numeric_columns :
    col_name = dataset[i]
    print("\nColumn name : ", col_name.name)
    print("The median is :", col_name.describe().loc['50%'])
    print("The modes are : ", col_name.mode())
    std = col_name.std()
    print("The variance is : ", (std ** 2))
#print("The number of invalid data is :")
    print(col_name.describe(exclude = [object]))
# Only modes, variance and missing values aren't covered by .dsecribe(). IQR range is shown, I think.
    q25 = col_name.describe(include = 'all').loc ['25%']
    q75 = col_name.describe(include = 'all').loc['75%']
#include = 'all' for this answer, but you can also supply a list of dtypes
    print("25% :", q25)
    print("75% :", q75)
    IQR = q75 - q25
    print("The IQR is : ", IQR)
    print("The number of missing values is :", col_name.isnull().sum())
    #For this case, let's do IQR. But I also want to implement the auto/unsupervised methods.\
    Q1 = q25 - (1.5*IQR) #Anything lower than this
    Q3 = q75 + (1.5*IQR) #Anything higher than this
    outlier_sum = 0
    def outlier(x):
        outlier_sum = 0
        for i in x :
            if ((i <= Q1) | (i >= Q3)):
                print("{} is an outlier \n".format(i))
                outlier_sum += 1
    outlier(col_name)    
    print("The total number of outliers in this column is : {}".format(outlier_sum))
    plt.figure()
    print(sns.boxplot(col_name)) #- if you want it!
#
print("\n\n")
print('''How do you want to treat the outliers?
1. Mean sampling
2. Trimming''')
user_input2 = int(input("Enter your choice : "))
for i in numeric_columns :
    col_name = dataset[i]
    outlier_treatment(col_name, user_input2)
print("Outlier treatment done.")    

#To determine feature selection, we need to convert last row(churn) into machine-readable form
target_variable = dataset.iloc[:,-1]
target_variable = target_variable.astype('category')
target_variable = target_variable.cat.codes
print(target_variable)

###Visualization

#Visualization
print(sns.pairplot(data = dataset))
for i in numeric_columns :
    col_name = dataset[i]
    plt.figure()
    print(sns.distplot(col_name))
    plt.figure()
    #plt.scatter(dataset[col_name], dataset['TotalCharges]
    #Other multivariate plots?
plt.figure()
print(sns.catplot(data = dataset)) 

#Scatterplot
temp_var = len(numeric_columns)
for i in range(0, temp_var):
    col_name = numeric_columns[i]
    #print("\n", col_name)
    for j in range(1, temp_var):
        next_col = numeric_columns[j]
        plt.scatter(dataset[col_name], dataset[next_col])
        plt.xlabel(col_name)
        plt.ylabel(next_col)
        plt.figure()

for i in range(0, temp_var):
    col_name = numeric_columns[i]
    #print("\n", col_name)
    for j in range(0, temp_var):
        next_col = numeric_columns[j]
        sns.relplot(data = dataset, x = col_name, y = next_col, hue = 'Churn') #Hue = target_variable

##Correlation
dataset.corr()
temp_var = len(numeric_columns)
for i in range(0, temp_var):
    col_name = numeric_columns[i]
    print("\nCorrelation for the column : ", col_name)
    print(dataset.corr()[col_name].sort_values(ascending = False))

from pandas.plotting import scatter_matrix
scatter_matrix(dataset, figsize = (16,12), alpha = 0.3) #Could also plot using sns heatmap   

###Feature selection 

#Feature importance
#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
model = RandomForestRegressor()
model.fit(dataset[numeric_columns], target_variable)
importance = model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

#Feature selection(only on training sets)
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
#X_train,y_train,X_test,y_test = train_test_split(dataset[numeric_columns], target_variable,test_size=0.2)
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(dataset[numeric_columns], target_variable)
sel.get_support()
selected_feat= dataset[numeric_columns].columns[(sel.get_support())]
print(len(selected_feat))
print(selected_feat)

##LASSO(feature selection)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1',  solver='liblinear'))
sel_.fit((dataset[numeric_columns].fillna(0)), target_variable)
sel_.get_support()

selected_feat = dataset[numeric_columns].columns[(sel_.get_support())]
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))
#Identifying removed features
removed_feats = dataset[numeric_columns].columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]

##Forward selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs_model = sfs(clf,
           k_features=2,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS
sfs1 = sfs_model.fit(dataset[numeric_columns], target_variable)

# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

##Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
model = DecisionTreeClassifier()
rfe.fit(dataset[numeric_columns], target_variable)
model.fit(dataset[numeric_columns], target_variable)

#What features?
for i in range(dataset[numeric_columns].shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
