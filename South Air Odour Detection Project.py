#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_excel('/Users/KUANGBixi/Desktop/Odour-train.xlsx')


# In[3]:


test = pd.read_excel('/Users/KUANGBixi/Desktop/Odour-test.xlsx')


# In[4]:


train.info()
train.head()


# In[5]:


test.info()
test.head()


# In[6]:


print('train columns with null values:\n', train.isnull().sum())
print('-'*20)
print('test columns with null values:\n', test.isnull().sum())
print('-'*20)
train.describe(include = 'all')


# In[7]:


train_test_data = [train, test]


# In[8]:


odour_mapping = {'PASS':1, 'NOT PASS':0}
train['气味等级'] = train['气味'].map(odour_mapping)


# In[9]:


train.head()


# In[10]:


def bar_chart(feature):
    PASS = train[train['气味等级']==1][feature].value_counts()
    NOT_PASS = train[train['气味等级']==0][feature].value_counts()
    df = pd.DataFrame([PASS, NOT_PASS])
    df.index = ['PASS', 'NOT_PASS']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))


# In[11]:


bar_chart('空调箱型号')


# In[12]:


bar_chart('壳体材料')


# In[13]:


bar_chart('添加剂种类')


# In[14]:


bar_chart('风扇')


# In[15]:


bar_chart('泡沫')


# In[16]:


bar_chart('胶水种类')


# In[17]:


ac_mapping = {'S201':0, 'C301':1, 'V301':2, 'S301':3, 'S101':4, 'B211':5}
for dataset in train_test_data:
    dataset['空调箱型号'] = dataset['空调箱型号'].map(ac_mapping)


# In[18]:


case_mapping = {'A':0, 'B':1, 'C':2}
for dataset in train_test_data:
    dataset['壳体材料'] = dataset['壳体材料'].map(case_mapping)


# In[19]:


additive_mapping = {'Q':0, 'V':1, 'S':2}
for dataset in train_test_data:
    dataset['添加剂种类'] = dataset['添加剂种类'].map(additive_mapping)


# In[20]:


train.head()


# In[21]:


test.head()


# In[22]:


drop_feature = ['id', '气味']
train = train.drop(drop_feature, axis=1)


# In[23]:


train_data = train.drop('气味等级', axis=1)
target = train['气味等级']


# In[24]:


train_data.head()


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
random_state = 2

classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, train_data, target, scoring = "accuracy", cv = kfold))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[30]:


cv_res


# In[32]:


DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsadaDTC.fit(train_data,target)

ada_best = gsadaDTC.best_estimator_
ada_best


# In[34]:


ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 6],
              "min_samples_split": [2, 3, 6],
              "min_samples_leaf": [1, 3, 6],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsExtC.fit(train_data,target)
ExtC_best = gsExtC.best_estimator_
ExtC_best


# In[36]:


RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 6],
              "min_samples_split": [2, 3, 6],
              "min_samples_leaf": [1, 3, 6],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsRFC.fit(train_data,target)
RFC_best = gsRFC.best_estimator_
RFC_best


# In[37]:


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsGBC.fit(train_data,target)
GBC_best = gsGBC.best_estimator_
GBC_best


# In[38]:


SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", verbose = 1)

gsSVMC.fit(train_data,target)
SVMC_best = gsSVMC.best_estimator_
SVMC_best


# In[41]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft')

votingC = votingC.fit(train_data, target)


# In[43]:


test_data = test.drop('id', axis=1).copy()
test_result = votingC.predict(test_data)

results = pd.DataFrame({
    'id': test['id'],
    'result': test_result
})

results.to_csv("South Air Odour Detection Prediction Result.csv",index=False)


# In[ ]:




