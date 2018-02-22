# Databricks notebook source
# MAGIC %md ##Indexing DataFrames

# COMMAND ----------

# MAGIC %md Import libraries.

# COMMAND ----------

import pandas as pd
import numpy  as np

# COMMAND ----------

# MAGIC %md Create `sales` dictionary.

# COMMAND ----------

sales={'eggs': [47, 110, 221, 77],
       'salt': [12., 50., 89., 87.],
       'spam': [17, 31, 72, 56]}

# COMMAND ----------

sales

# COMMAND ----------

# MAGIC %md Convert `sales` dict into pandas dataframe.

# COMMAND ----------

pd.DataFrame(sales)

# COMMAND ----------

# MAGIC %md Add row index for the pandas dataframe, and name the dataframe `sales_df`.

# COMMAND ----------

sales_df = pd.DataFrame(sales, index=['Jan', 'Feb', 'Mar', 'Apr'])

# COMMAND ----------

sales_df

# COMMAND ----------

# MAGIC %md Now we have a pandas dataframe with `4` rows and `3` columns.

# COMMAND ----------

# MAGIC %md The pandas library offers several ways of indexing data in dataframes:
# MAGIC 
# MAGIC 1. Indexing using square brackets, example: df[`column`][`row`];
# MAGIC 1. Using column attribute and row label, example: df.`column`[`row`];
# MAGIC 1. Using the `.loc` accessor, example: df.loc[`row`, `column`];
# MAGIC 1. Using the `.iloc` accessor, example: df.iloc[`row number`, `column number`].
# MAGIC 
# MAGIC Let's try accessing the dataframe in different ways.

# COMMAND ----------

sales_df['salt']['Mar']

# COMMAND ----------

sales_df.salt['Mar']

# COMMAND ----------

sales_df.loc['Mar', 'salt']

# COMMAND ----------

sales_df.iloc[2, 1]

# COMMAND ----------

# MAGIC %md ##Slicing DataFrames

# COMMAND ----------

# MAGIC %md Slicing and indexing using `.loc` with labels:
# MAGIC - 'Jan' to 'Apr' rows (inclusive);
# MAGIC - 'eggs' to 'salt' columns (inclusive).

# COMMAND ----------

sales_df.loc['Jan':'Apr', 'eggs':'salt']

# COMMAND ----------

# MAGIC %md Slicing and indexing using `.iloc` with positions:
# MAGIC 
# MAGIC - From row 1 up to but not including row 3;
# MAGIC - From column 1 onwards.

# COMMAND ----------

sales_df.iloc[1:3, 1:]

# COMMAND ----------

# MAGIC %md Lists are also acceptable in indexing DataFrames using `.loc` and `.iloc`, such as:

# COMMAND ----------

sales_df.iloc[[0, 2, 3], 0:2]

# COMMAND ----------

# MAGIC %md Selecting Series vs. 1-column DataFrame

# COMMAND ----------

type(sales_df['eggs'])

# COMMAND ----------

type(sales_df[['eggs']])

# COMMAND ----------

# MAGIC %md Slice the row labels 'Feb' to 'Apr'.

# COMMAND ----------

sales_df.loc['Feb':'Apr']

# COMMAND ----------

# MAGIC %md Slice the row labels in reverse order from 'Apr' to 'Feb', with `-1`.

# COMMAND ----------

sales_df.loc['Apr':'Feb':-1]

# COMMAND ----------

# MAGIC %md Subselecting DataFrames with lists

# COMMAND ----------

rows = ['Feb', 'Mar']
cols = ['salt', 'spam']
sales_df.loc[rows, cols]

# COMMAND ----------

# MAGIC %md ##Filtering DataFrames

# COMMAND ----------

# MAGIC %md Create a Boolean Series.

# COMMAND ----------

sales_df.salt > 50

# COMMAND ----------

# MAGIC %md Filtering with a Boolean Series.

# COMMAND ----------

sales_df[sales_df.salt > 50]

# COMMAND ----------

# MAGIC %md Combining filters with various logical operators.

# COMMAND ----------

sales_df[(sales_df.salt > 50) & (sales_df.eggs > 80)] 

# COMMAND ----------

sales_df[(sales_df.salt > 50) | (sales_df.eggs > 80)] 

# COMMAND ----------

# MAGIC %md Filtering a column based on another.

# COMMAND ----------

sales_df.eggs[sales_df.salt > 25]

# COMMAND ----------

# MAGIC %md DataFrames with zeros and NaNs

# COMMAND ----------

df2 = sales_df.copy()
df2['bacon'] = [0, 0, 50, 60]
df2

# COMMAND ----------

# MAGIC %md Select columns with all non-zero values.

# COMMAND ----------

df2.loc[:, df2.all()]

# COMMAND ----------

# MAGIC %md Select columns with any non-zero values (contain at least one value that is non-zero).

# COMMAND ----------

df2.loc[:, df2.any()]

# COMMAND ----------

# MAGIC %md Select columns without any NaN values by jointly using `notnull()` and `all()` methods.

# COMMAND ----------

df2.loc[:, df2.notnull().all()]

# COMMAND ----------

# MAGIC %md Reversely, we can also select columns with at least one NaN value by using `isnull()` and `any()` methods. However, in `df2` there's no such column.

# COMMAND ----------

df2.loc[:, df2.isnull().any()]

# COMMAND ----------

# MAGIC %md Drop rows with any or all NaNs with `dropna()`. However, in our example, none of the columns contain NaN values, so nothing was dropped.

# COMMAND ----------

df2.dropna(how='any')
df2.dropna(how='all')

# COMMAND ----------

# MAGIC %md ##Transforming DataFrames

# COMMAND ----------

# MAGIC %md DataFrame vectorized methods

# COMMAND ----------

sales_df.floordiv(12) #Convert all values in dozens unit

# COMMAND ----------

# MAGIC %md Np vectorized methods

# COMMAND ----------

np.floor_divide(sales_df, 12)

# COMMAND ----------

# MAGIC %md Plain Python functions with `def`.

# COMMAND ----------

def dozens(n):
  return n//12;

# COMMAND ----------

sales_df.apply(dozens)

# COMMAND ----------

# MAGIC %md One-time use Python functions with `lambda`.

# COMMAND ----------

sales_df.apply(lambda n: n//12)

# COMMAND ----------

# MAGIC %md Storing a transformation with a new column

# COMMAND ----------

sales_df['eggs_in_dozens'] = sales_df.eggs.floordiv(12)

# COMMAND ----------

sales_df

# COMMAND ----------

# MAGIC %md Working with string values of the index.

# COMMAND ----------

sales_df.index

# COMMAND ----------

sales_df.index = sales_df.index.str.upper() #Transform the indexes into all upper case

# COMMAND ----------

sales_df.index

# COMMAND ----------

# MAGIC %md `index` methods can't use `apply` methods, but have `map` instead.

# COMMAND ----------

sales_df.index = sales_df.index.map(str.lower) #Transforms the indexes into lower case

# COMMAND ----------

sales_df.index

# COMMAND ----------

# MAGIC %md Defining columns using computation of other columns.

# COMMAND ----------

sales_df['salt&eggs'] = sales_df['salt'] + sales_df['eggs']

# COMMAND ----------

sales_df

# COMMAND ----------

# MAGIC %md Add quarter to `df`.

# COMMAND ----------

sales_df['quarter'] = ['first', 'first', 'first', 'second']

# COMMAND ----------

sales_df

# COMMAND ----------

# MAGIC %md Use `.map()` to create the quarter number with a dict.

# COMMAND ----------

quarter_in_num = {'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th'}
sales_df['quarter#'] =  sales_df.quarter.map(quarter_in_num)

# COMMAND ----------

sales_df

# COMMAND ----------

# MAGIC %md __The End__