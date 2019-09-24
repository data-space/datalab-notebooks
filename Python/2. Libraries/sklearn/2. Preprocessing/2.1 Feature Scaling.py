# Databricks notebook source
# MAGIC %md # Feature Scaling

# COMMAND ----------

# MAGIC %md ## Table of Contents
# MAGIC 
# MAGIC 1. Introduction to Feature Scaling
# MAGIC 1. Standardization
# MAGIC 1. Scaling features to a range 
# MAGIC 1. Scaling data with outliers
# MAGIC 1. Scaling in a pipeline
# MAGIC 1. Summary

# COMMAND ----------

# MAGIC %md ## 1. Introduction to Feature Scaling

# COMMAND ----------

# MAGIC %md Feature scaling is a common requirement for many machine learning estimators implemented in scikit-learn: estimators might behave badly when the input numerical attributes have very different scales. There are two common ways to get all attributes to have the same scale: standardization and min-max scaling. In addition, if some outliers are present in the set, robust scalers are more appropriate. This notebook will introduce these feature scaling methods. In the end, the behavior of feature scaling can be tested using the `Pipeline` class with a downstream estimator. 

# COMMAND ----------

# MAGIC %md ## 2. Standardization

# COMMAND ----------

# MAGIC %md Standardization is a process to make data look like standard normally distributed data: first it substracts the mean value of each feature, and then it divides by the variance so that the resulting distribution has zero mean and unit variance. 

# COMMAND ----------

# MAGIC %md The function `scale` in `sklearn.preprocessing` provides a quick and easy way to perform standardization on a single array-like dataset. Below is an example which uses the `scale` function to standardize data.

# COMMAND ----------

# MAGIC %md Load libraries.

# COMMAND ----------

from sklearn import preprocessing
import numpy as np

# COMMAND ----------

# MAGIC %md Transform a toy dataset `X_train` using the `scale` function and store the scaled data in `X_scaled`.

# COMMAND ----------

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
X_scaled

# COMMAND ----------

# MAGIC %md Scaled data has zero mean and unit variance:

# COMMAND ----------

X_scaled.mean(axis=0),\
X_scaled.std(axis=0)

# COMMAND ----------

# MAGIC %md The function `scale` can be used to transform dataset into scaled data with zero mean and unit variance. 

# COMMAND ----------

# MAGIC %md ### StandardScaler

# COMMAND ----------

# MAGIC %md The `preprocessing` module further provides a utility class `StandardScaler` that implements the Transformer API to compute the mean and standard deviation on a training set using the `fit` method, and to later transform the training or testing set using the `transform` method with the computed mean and standard deviation.

# COMMAND ----------

# MAGIC %md The same dataset as above is used in this example.

# COMMAND ----------

X_train

# COMMAND ----------

# MAGIC %md Create a `StandardScaler` object and store it in the `scaler` variable. When the `x_train` dataset is fit by the `scaler` object then the means and standard deviations of the features in the `x_train` dataset are stored in that object.

# COMMAND ----------

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
print(scaler.mean_)  
print(scaler.var_)  

# COMMAND ----------

# MAGIC %md The `scaler` object can transform the fitted data (`X_train`).

# COMMAND ----------

x_scaled=scaler.transform(X_train)
x_scaled

# COMMAND ----------

# MAGIC %md Scaled data has zero mean and unit variance:

# COMMAND ----------

x_scaled.mean(axis=0),\
x_scaled.std(axis=0)

# COMMAND ----------

# MAGIC %md Create a test dataset `X_test`. Transform the `X_test` dataset using the `scaler` object through subtracting the stored means from the feature values and then dividing by the stored standard variance. Store the outcome in the `x_t_scaled` object.

# COMMAND ----------

X_test = [[-1., 1., 0.]]
x_t_scaled=scaler.transform(X_test)
x_t_scaled

# COMMAND ----------

# MAGIC %md The above session introduces a quick and easy way to implement standardization on single dataset and using the `StandardScaler` class which can standardize features on the training data and be reapplied on test data.

# COMMAND ----------

# MAGIC %md ## 3. Scaling features to a range 

# COMMAND ----------

# MAGIC %md An alternative way of feature scaling is to transform a feature so its values lie between a given minimum and maximum value, often between 0 and 1, or so that the maximum absolute value of each feature is scaled to unit size. This can be achieved using `MinMaxScaler` or `MaxAbsScaler`, respectively. 

# COMMAND ----------

# MAGIC %md ### MinMaxScaler

# COMMAND ----------

# MAGIC %md This example scales features to [0,1] using `MinMaxScaler` on a training set so as to be able to later reapply the same transformation on the testing set.

# COMMAND ----------

# MAGIC %md Look at the dataset first. Create an object called `min_max_scaler` and then use the training dataset to fit it. Store in the `min_max_scaler` object the minimum and maximum values of each feature in the training data. Then transform the training data and pass the scaled data to the `X_train_minmax` object.  

# COMMAND ----------

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax

# COMMAND ----------

# MAGIC %md When the `min_max_scaler` object is fit to the `X_train` dataset the maximum, minimum and range from the feature values of that dataset are stored in the `data_max_`, `data_min_` and `data_range_` attributes.

# COMMAND ----------

min_max_scaler.data_max_,\
min_max_scaler.data_min_,\
min_max_scaler.data_range_

# COMMAND ----------

# MAGIC %md Create a test dataset `X_test`. Transform the `X_test` dataset using the `min_max_scaler` object through subtracting the stored minimum from the feature values and then dividing by the stored data range. Store the outcome in the `x_test_minmax` object.

# COMMAND ----------

X_test = np.array([[ -3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax

# COMMAND ----------

# MAGIC %md ### MaxAbsScaler 

# COMMAND ----------

# MAGIC %md Another way is to scale features so that the maximal absolute value of each feature in the training set will be `1.0`. It does not shift/center the data, and thus does not destroy any sparsity. This can be achieved using `MaxAbsScaler`. Here is an example of scaling features to [-1,1] using `MaxAbsScaler` on a training set so as to be able to later reapply the same transformation on the testing set.

# COMMAND ----------

# MAGIC %md Use the training dataset to fit the `max_abs_scaler` object. Store the maximum absolute value of each feature in the `max_abs_scaler` object. Then transform the training data and pass the scaled data to the `X_train_maxabs` object.  

# COMMAND ----------

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs 

# COMMAND ----------

# MAGIC %md For example, the first column `[1,2,0]` of the `X_train_maxabs` object is transformed into `[0.5,1,0]` in the `X_train_maxabs` object. This is because the maximum absolute value of the first column is `2` and each value is divided by 2 to be scaled into value in [-1,1].

# COMMAND ----------

# MAGIC %md The scaler is fitted to have the maximal absolute value of each feature.

# COMMAND ----------

max_abs_scaler.max_abs_

# COMMAND ----------

# MAGIC %md Create a test dataset `X_test`. Transform the `X_test` dataset using the `max_abs_scaler` object through dividing by maximum absolute values of features in the training data set. Then store the scaled data in the `X_test_maxabs` object.

# COMMAND ----------

X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs

# COMMAND ----------

# MAGIC %md The above session introduces scaling features to a specific range with two scalers `MinMaxScaler` which scales each feature to a given range (normally [0,1]) and `MaxAbsScaler` which scales each feature by its maximum absolute value.

# COMMAND ----------

# MAGIC %md ## 4. Scaling data with outliers

# COMMAND ----------

# MAGIC %md If your data contains outliers, scaling using the mean and variance of the data is likely to not work very well. In these cases, you can use `robust_scale` and `RobustScaler` as drop-in replacements instead. They use more robust estimates for the center and range of your data. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

# COMMAND ----------

# MAGIC %md The class `RobustScaler` implements the Transformer API to compute the Median and interquartile range on a training set so as to be able to later reapply the same transformation on the testing set. Below is an example of using the scaler on a toy dataset.

# COMMAND ----------

# MAGIC %md Use the training dataset to fit the `robust_scaler` object. Store the Median and interquartile range of each feature in the `robust_scaler` object. Then transform the training data and pass the scaled data to the `X_train_robust` object.

# COMMAND ----------

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
robust_scaler = preprocessing.RobustScaler(quantile_range=(25, 75))
X_train_robust = robust_scaler.fit_transform(X_train)
X_train_robust

# COMMAND ----------

# MAGIC %md Median and interquartile range are then stored to be used on later data using the transform method.

# COMMAND ----------

robust_scaler.scale_,\
robust_scaler.center_

# COMMAND ----------

# MAGIC %md The scaler instance can then be used on new data to transform it the same way it did on the training set (minus median and divide by interquartile range):

# COMMAND ----------

X_test = np.array([[ -3., -1.,  4.]])
X_test_robust = robust_scaler.transform(X_test)
X_test_robust

# COMMAND ----------

# MAGIC %md The above session introduces one scaler `RobustScaler` which can scale features using statistics that are robust to outliers.

# COMMAND ----------

# MAGIC %md ##5. Scaling in a Pipeline

# COMMAND ----------

# MAGIC %md This example shows whether or not scaling the features of the California Housing Prices dataset has any impact on the performance of a k-Nearest-Neighbors estimator.

# COMMAND ----------

# MAGIC %md Load libraries.

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# MAGIC %md Read the dataset using the `fetch_california_housing` function and then split it into train and test using the `train_test_split` function. 

# COMMAND ----------

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
X_train, X_test, y_train, y_test=train_test_split(X_full,y_full,test_size=0.2, random_state=20)

# COMMAND ----------

# MAGIC %md Here we use a k-nearest neighbors regressor as part of a pipeline that includes scaling, and for the purposes of comparison, a knn regressor trained on the unscaled data has been provided in the following code cell. 

# COMMAND ----------

steps=[('scaler', StandardScaler()),
       ('knn',    KNeighborsRegressor())]

pipeline=Pipeline(steps)

# COMMAND ----------

# MAGIC %md Fit the pipeline using `X_train` as training data and `y_train` as target values, and pass the computed parameters to an object `knn_scaled`. Also, fit a knn regressor using unscaled training data and pass the computed parameters to the object `knn_unscaled`.

# COMMAND ----------

knn_scaled = pipeline.fit(X_train, y_train)
knn_unscaled = KNeighborsRegressor().fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md Compute and print metrics.

# COMMAND ----------

print('Prediction Error with Scaling: {}'.format(mean_squared_error(y_test, knn_scaled.predict(X_test))))
print('Prediction Error without Scaling: {}'.format(mean_squared_error(y_test, knn_unscaled.predict(X_test))))

# COMMAND ----------

# MAGIC %md The output above shows that feature scaling has significantly improved the performance of k-nearest neighbors regressor in predicting using the California Housing Prices dataset.

# COMMAND ----------

# MAGIC %md ##6. Summary 
# MAGIC The above example shows how feature scaling can be tested using the `Pipeline` class with a downstream estimator. Here the k-NN estimator performs better with scaling than without scaling. 
# MAGIC 
# MAGIC With few exceptions, machine learning algorithms do not perform well when the input numerical variables have very different scales. In general, algorithms that exploit distances or similarities (e.g. in form of scalar product) between data samples, such as k-NN, cluster analysis, Support Vector Machines and neural networks, are sensitive to feature transformations.
# MAGIC 
# MAGIC Graphical-model based classifiers, such as Fisher LDA or Naive Bayes, as well as Decision trees and Tree-based ensemble methods (Random Forests, XGBoost) are invariant to feature scaling ('invariant' means the performance of estimator will not be altered by feature scaling). 
