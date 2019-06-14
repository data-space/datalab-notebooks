# Databricks notebook source
# MAGIC %md #Variance Thresholding for Feature Selection

# COMMAND ----------

# MAGIC %md ## Table of Contents
# MAGIC 1. Introduction to variance thresholding
# MAGIC 1. `VarianceThreshold` transformer class
# MAGIC 1. Variance thresholding for feature selection

# COMMAND ----------

# MAGIC %md ##1. Introduction to variance thresholding

# COMMAND ----------

# MAGIC %md 
# MAGIC Variance Thresholding is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold.
# MAGIC This method is motivated by the idea that low variance features normally contain less information. 
# MAGIC 
# MAGIC Notice that data column ranges need to be normalized to make variance values independent from the column domain range. 

# COMMAND ----------

# MAGIC %md Load library.

# COMMAND ----------

import numpy as np

# COMMAND ----------

# MAGIC %md Create a sample data array `X` from a list. `X` has integer features, two of which are the same in every sample.

# COMMAND ----------

X =np.asarray( [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
X

# COMMAND ----------

# MAGIC %md Compute the columnwise variance of each column in `X`.

# COMMAND ----------

np.var(X, axis=0)

# COMMAND ----------

# MAGIC %md Set a threshold. Columns whose variance is lower than or equal to the threshold will be removed.

# COMMAND ----------

threshold = 0

# COMMAND ----------

# MAGIC %md Only select the columns whose variance is above `0` from data array `X` and store it in an object `arr_new`.

# COMMAND ----------

arr_new = X[:, np.var(X, axis=0) > threshold]
arr_new

# COMMAND ----------

# MAGIC %md As expected, `arr_new` has only two columns whose variance are all above the threshold.

# COMMAND ----------

# MAGIC %md ##2. `VarianceThreshold` transformer class

# COMMAND ----------

# MAGIC %md The class `VarianceThreshold` in the `sklearn.feature_selection` module can be used to remove features with low variance. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

# COMMAND ----------

# MAGIC %md Load `VarianceThreshold` transformer class from Scikit-Learn library.

# COMMAND ----------

from sklearn.feature_selection import VarianceThreshold

# COMMAND ----------

# MAGIC %md Create an instance of the `VarianceThreshold` class and store it in an object `selector`. Set threshold to default (`0`). 

# COMMAND ----------

selector = VarianceThreshold()

# COMMAND ----------

# MAGIC %md Fit the transformer class to the sample data array `X` and return a transformed version of `X`. The `fit` method records the variance of each feature from `X`. The `transform` method returns the selected features from `X`.

# COMMAND ----------

selector.fit_transform(X)

# COMMAND ----------

# MAGIC %md With the default setting for threshold, the two column features with variance above 0 are selected.

# COMMAND ----------

# MAGIC %md The transformer class `VarianceThreshold` has attribute `variances_`. Use it to see variances of individual features of `X`. They are the same as in the output of `np.var(X, axis=0)`. 

# COMMAND ----------

selector.variances_

# COMMAND ----------

np.var(X, axis=0)

# COMMAND ----------

# MAGIC %md The above section introduces a transformer class in scikit-learn `VarianceThreshold`. It removes all low-variance features according to a given threshold.

# COMMAND ----------

# MAGIC %md ##3. Variance thresholding for feature selection

# COMMAND ----------

# MAGIC %md ####Example on the iris dataset

# COMMAND ----------

# MAGIC %md Load libraries.

# COMMAND ----------

from sklearn import datasets,model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %md Load the iris dataset from the `sklearn` datasets module. About the iris dataset:
# MAGIC - [iris data](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)

# COMMAND ----------

# MAGIC %md Create a function `load_data`. The function returns a `train_test_split` method which splits the iris data into train and test. Then use the function to get training and testing data and labels.

# COMMAND ----------

def load_data():
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return model_selection.train_test_split(X_train, y_train,test_size=0.25,random_state=0,stratify=y_train)
  
X_train,X_test,y_train,y_test=load_data()

# COMMAND ----------

# MAGIC %md Create an instance of the `DecisionTreeClassifier` estimator and store it in an object `clf`. Fit the `DecisionTreeClassifier` estimator to the training data. The parameters of the estimator are displayed in the output.

# COMMAND ----------

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md Create a `Pipeline` class which is built with a `VarianceThreshold` transformer and a `DecisionTreeClassifier` estimator. Store it in an object `clf_s` and fit it using the training data. The parameters are displayed in the output.

# COMMAND ----------

clf_s = Pipeline([
  ('feature_selection', VarianceThreshold(.8*(1-.8))),
  ('classification', DecisionTreeClassifier(criterion='entropy'))
])
clf_s.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md Create predictions on the testing data and display their accuracy.

# COMMAND ----------

print('Prediction Accuracy without Feature Selection: {}'.format(accuracy_score(y_test, clf.predict(X_test))))
print('Prediction Accuracy with Feature Selection: {}'.format(accuracy_score(y_test, clf_s.predict(X_test))))

# COMMAND ----------

# MAGIC %md The above is an example of how feature selection makes a difference. 
# MAGIC The variance thresholding method may or may not improve the predictive accuracy, but it does:
# MAGIC 
# MAGIC - increase the interpretability of the model.
# MAGIC - reduce the complexity of the model.
# MAGIC - reduce the training time of the model.