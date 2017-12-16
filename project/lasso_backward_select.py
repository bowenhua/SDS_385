#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#import lightgbm as lgb

sns.set(color_codes=True)
plt.rc("savefig", dpi=300)


traindf = pd.read_csv('train.csv')
# Outlier
traindf = traindf[traindf['GrLivArea'] < 4000]

testdf = pd.read_csv('test.csv')

# Check data types
# traindf.dtypes

#f = plt.figure()
#sns.distplot(traindf['SalePrice'])
#plt.show()
#f.savefig('dist.pdf')
#
#plt.figure()
#sns.jointplot(x="LotArea", y="SalePrice", data=traindf)
##
##plt.figure()
##sns.boxplot(x="MiscFeature", y="SalePrice", data=traindf)
#
#plt.figure()
#sns.boxplot(x="MSZoning", y="SalePrice", data=traindf)

predictiondf = pd.DataFrame(testdf['Id'])

# Log transform
traindf.SalePrice = np.log1p(traindf.SalePrice)

trainY = traindf['SalePrice']
traindf = traindf.drop('SalePrice', 1)


def dfClean(df):
    
    
    # Delete unimportant features
    df = df.drop('Id', 1)
    #df = df.drop('MSSubClass', 1)
    df = df.drop('Utilities', 1)
    df = df.drop('LotConfig', 1)
    df = df.drop('LandSlope', 1)
    df = df.drop('Condition2', 1)
    df = df.drop('Alley', 1)
    df = df.drop('RoofStyle', 1)
    df = df.drop('RoofMatl', 1)
    df = df.drop('Exterior1st', 1)
    df = df.drop('Exterior2nd', 1)
    df = df.drop('MasVnrType', 1)
    df = df.drop('Foundation', 1)
    df = df.drop('BsmtExposure', 1)
    df = df.drop('BsmtFinType1', 1)
    df = df.drop('BsmtFinType2', 1)
    df = df.drop('Heating', 1)
    df = df.drop('Electrical', 1)
    df = df.drop('GarageYrBlt', 1)
    df = df.drop('GarageFinish', 1)
    df = df.drop('GarageCars', 1)
    df = df.drop('MoSold', 1)
    df = df.drop('TotalBsmtSF', 1)
    
    
    # Drop highly correlated features
    df = df.drop('GarageCond', 1)
    df = df.drop('PoolQC', 1)
    

    
    # Modify features
    df["Condition1"] = df["Condition1"].map({"Artery":"Con1Negative", "Feedr":"Con1Negative", "Norm" : "Con1Norm" , "RRNn":"Con1Negative", "RRAn":"Con1Negative", "RRNe":"Con1Negative", "RRAe":"Con1Negative", "PosN":"Con1Positive", "PosA":"Con1Positive"})
    df["ExterQual"] = df["ExterQual"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})  #values in Series that are not in the dictionary (as keys) are converted to NaN
    df["ExterCond"] = df["ExterCond"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})
    df["BsmtQual"] = df["BsmtQual"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})
    df["BsmtCond"] = df["BsmtCond"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})
    df["HeatingQC"] = df["HeatingQC"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})
    df["Functional"] = df["Functional"].map({"Typ":"FunTyp", "Min1":"FunMin", "Min2":"FunMin", "Mod":"FunMod", "Maj1":"FunMaj", "Maj2":"FunMaj","Sev":"FunMaj","Sal":"FunMaj",})
    df["FireplaceQu"] = df["FireplaceQu"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})
    df["GarageQual"] = df["GarageQual"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})    
    #df["GarageCond"] = df["GarageCond"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})    
    #df["PoolQC"] = df["PoolQC"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})  
    df["KitchenQual"] = df["KitchenQual"].map({"Ex":5, "Gd": 4, "TA":3, "Fa":2, "Po":1})
    df["MSSubClass"] = df["MSSubClass"].map({20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"})
    
    # New features
    #df["IsRegularLotShape"] = (df["LotShape"] == "Reg") * 1
    df = df.drop('LotShape', 1)
    #df["IsLandLevel"] = (df["LandContour"] == "Lvl") * 1
    df = df.drop('LandContour', 1)
    #df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1
    df = df.drop('GarageType', 1)
    df["VeryNewHouse"] = (df["YearBuilt"] == df["YrSold"]) * 1

    
    # Fill missing data
    df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    df['BsmtQual'].fillna(0, inplace=True)
    df['BsmtCond'].fillna(0, inplace=True)
    df['FireplaceQu'].fillna(0, inplace=True)
    df['GarageQual'].fillna(0, inplace=True)
    #df['GarageCond'].fillna(0, inplace=True)
    #df['PoolQC'].fillna(0, inplace=True)
    df['Fence'].fillna("NoFence", inplace=True)
    df['MiscFeature'].fillna("NoFeature", inplace=True)
#    df['Utilities'].fillna("AllPub", inplace=True)
#    df['MasVnrType'].fillna("None", inplace=True)
#    df['Alley'].fillna("NA", inplace=True)
    
    
    return df

traindf = dfClean(traindf)
testdf = dfClean(testdf)

# Fill missing data for testdf only

testdf['MSZoning'].fillna('RL', inplace = True)
testdf['BsmtFinSF1'].fillna(0, inplace = True) # Checked the raw data that place does not have bsmt
testdf['BsmtFinSF2'].fillna(0, inplace = True)
testdf['BsmtUnfSF'].fillna(0, inplace = True)
testdf['BsmtFullBath'].fillna(0, inplace = True)
testdf['BsmtHalfBath'].fillna(0, inplace = True)
testdf['KitchenQual'].fillna(3, inplace = True)
testdf['Functional'].fillna('FunTyp', inplace = True)
testdf['GarageArea'].fillna(0, inplace = True)
testdf['SaleType'].fillna('WD', inplace = True)

# Sanity check
if traindf.isnull().any().any() or testdf.isnull().any().any():
    raise ValueError('NaN Values!!')

 
# Get dummy variables
temp = pd.get_dummies(pd.concat((traindf, testdf)), drop_first=True) #drop_first doesn't matter so much for tree predictors, but still we turn it on
traindf = temp.iloc[0:traindf.shape[0],:]
testdf = temp.iloc[traindf.shape[0]:,:]

# Corr heatmap
#corrmat = traindf.corr()
#f, ax = plt.subplots(figsize=(50, 50))
#sns.heatmap(corrmat, vmax=.8, square=True);


# Unskew
cols_skew_candidate = [col for col in traindf if '_' not in col]
#traindf[cols_skew_candidate].skew()
cols_unskew = traindf.loc[:, cols_skew_candidate].columns[abs(traindf.loc[:,cols_skew_candidate].skew()) > 1]
for col in cols_unskew:
    traindf.loc[:,col] = np.log1p(traindf.loc[:,col])
    testdf.loc[:,col] = np.log1p(testdf.loc[:,col])
    

# Check if there is any column with little cardinality
#traindf.astype(bool).sum(axis=1)


# Convert to numerical arrays
trainX = traindf.values
testX = testdf.values
trainY = trainY.values


# Scaling
scaler = preprocessing.StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

# Save to MATLAB format
#sp.io.savemat('data.mat', mdict={'trainX': trainX, 'trainY': trainY, 'testX': testX})

# sklearn gradient boosting
#params = {'n_estimators': 3000, 'max_depth': 3, 'min_samples_split': 2, 'subsample':0.9,
#          'learning_rate': 0.01, 'loss': 'ls'}
#clf = ensemble.GradientBoostingRegressor(**params)
#clf.fit(trainX, trainY)
#
#print(mean_squared_error(trainY, clf.predict(trainX)))
#print(np.sqrt(-cross_val_score(clf, trainX, trainY, scoring = 'neg_mean_squared_error').mean()))
#
#Y_pred_gbr = np.expm1(clf.predict(testX))
#
### Plit importance for clf
##feature_importance = clf.feature_importances_
### make importances relative to max importance
##feature_importance = 100.0 * (feature_importance / feature_importance.max())
##sorted_idx = np.argsort(feature_importance)
##pos = np.arange(sorted_idx.shape[0]) + .5
##plt.subplots(figsize=(50, 50))
##plt.barh(pos, feature_importance[sorted_idx], align='center')
##plt.yticks(pos, np.array(traindf.columns.values)[sorted_idx])
##plt.xlabel('Relative Importance')
##plt.title('Variable Importance')
##plt.show()
#
#
### RandomForest
##rfr = RandomForestRegressor(n_estimators = 100)
##rfr.fit(trainX, trainY)
##print(cross_val_score(rfr, trainX, trainY, scoring = 'neg_mean_squared_error').mean())
#
#
## LightGBM
#
#gbm = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                              learning_rate=0.05, n_estimators=720,
#                              max_bin = 55, bagging_fraction = 0.8,
#                              bagging_freq = 5, feature_fraction = 0.2319,
#                              feature_fraction_seed=9, bagging_seed=9,
#                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#
#gbm.fit(trainX, trainY)
#print(np.sqrt(-cross_val_score(gbm, trainX, trainY, scoring = 'neg_mean_squared_error').mean()))
#
#Y_pred_gbm = np.expm1(gbm.predict(testX))
#
## ElasNet
#
#enet = linear_model.ElasticNetCV(alphas=[0.0001, 0.0003, 0.0005, 0.001, 0.01, 0.1, 1], l1_ratio=[.01, .1, .2, .3, .5, .8, .9, .99], max_iter=2000000).fit(trainX, trainY)
#print(np.sqrt(-cross_val_score(enet, trainX, trainY, scoring = 'neg_mean_squared_error').mean()))
#Y_pred_enet = np.expm1(enet.predict(testX))

# Lasso
lasso = linear_model.LassoCV(alphas=[0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.01, 0.1], max_iter=2000000).fit(trainX, trainY)
print(np.sqrt(-cross_val_score(lasso, trainX, trainY, scoring = 'neg_mean_squared_error').mean()))

lasso_1 = linear_model.Lasso(alpha = 0.023, max_iter=2000000).fit(trainX, trainY)
print(mean_squared_error(trainY, lasso_1.predict(trainX)))
print(sum(lasso_1.coef_ >0))
Y_pred_lasso = np.expm1(lasso_1.predict(testX))

lasso_support = np.abs(lasso_1.coef_)>0 
lasso_param = lasso_1.coef_



# RFE
lr = LinearRegression()
selector = RFE(lr, 20, step=1)
selector.fit(trainX, trainY)

RFE_support = selector.get_support()
RFE_param_nonzero = selector.estimator_.coef_
RFE_param = np.zeros((trainX.shape[1], 1))
count = 0
for idx, truth in enumerate(RFE_support):
    if truth:
        RFE_param[idx] = RFE_param_nonzero[count]
        count += 1
RFE_param = RFE_param.flatten()

Y_pred_RFE = np.expm1(selector.predict(testX))

feature_names = traindf.columns

# BSS
BSS_param = sp.io.loadmat('beta20.mat')
BSS_param = BSS_param['beta']
BSS_intercept = BSS_param[-1]
BSS_param = BSS_param[:-1]
BSS_support = np.abs(BSS_param) >0
BSS_param = BSS_param.flatten()

Y_pred_BSS = np.expm1(testX.dot(BSS_param) + BSS_intercept)

# Dict for comparison
# Key is feature name, value is a 3-tuple of corresponding parameter value for BSS, lasso, RFE

selection = {}

for idx, name in enumerate(feature_names):
    if BSS_support[idx] or lasso_support[idx] or RFE_support[idx]:
        selection[name] = (BSS_param[idx], lasso_param[idx], RFE_param[idx])


BSS_plot_param = [selection[name][0] for name in selection.keys()]
lasso_plot_param = [selection[name][1] for name in selection.keys()]
RFE_plot_param = [selection[name][2] for name in selection.keys()]

## Plot
        
f = plt.figure(figsize=(5, 10))
pos = np.arange(len(selection))
plt.barh(pos-.2, BSS_plot_param, align='center', height = 0.2, label='BSS')
plt.barh(pos, lasso_plot_param, align='center', height = 0.2, label='Lasso')
plt.barh(pos+.2, RFE_plot_param, align='center',height = 0.2, label='Backward Selection')
plt.yticks(pos, feature_names)
plt.xlabel(r'$\beta$')
plt.title('Subset Selection Results')
plt.legend()
plt.show()
f.savefig('selection.pdf')


## Output
#

predictiondf['SalePrice']=Y_pred_lasso
predictiondf.to_csv('prediction.csv',index=False)








