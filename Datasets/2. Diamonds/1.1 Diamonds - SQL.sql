-- Databricks notebook source
-- MAGIC %md # Read diamonds Dataset with SQL

-- COMMAND ----------

-- MAGIC %md ## Introduction
-- MAGIC This notebook imports the `diamonds` dataset. This entails:
-- MAGIC 2. Reading the datafile (into a dataframe/table)
-- MAGIC 3. Checking the datatypes (of each column in the dataframe)
-- MAGIC 4. Setting these datatypes (if they were not initially read correctly)
-- MAGIC 
-- MAGIC The sections of this notebook (listed below, except the Setup section) correspond to each step. 
-- MAGIC Note that the columns of the diamonds dataset are all initially read correctly. 
-- MAGIC Other notebooks require more work to set the column datatypes correctly. 

-- COMMAND ----------

-- MAGIC %md ## Contents
-- MAGIC 1. Setup
-- MAGIC 2. Read datafile
-- MAGIC 3. Check column types
-- MAGIC 4. Set column types 

-- COMMAND ----------

-- MAGIC %md ## 1. Setup

-- COMMAND ----------

-- MAGIC %md The notebook `Include` 
-- MAGIC - contains some references 
-- MAGIC - loads libraries
-- MAGIC - defines the function `get_filepaths` in R and Python to facilitate locating the datafile
-- MAGIC 
-- MAGIC Display the notebook results to see these references and the libraries. 

-- COMMAND ----------

-- MAGIC %r
-- MAGIC diamonds_filepath = '/dbfs/mnt/datalab-datasets/file-samples/diamonds.csv'

-- COMMAND ----------

-- MAGIC %python
-- MAGIC diamonds_filepath = '/dbfs/mnt/datalab-datasets/file-samples/diamonds.csv'

-- COMMAND ----------

-- MAGIC %md ### 2. Read using SQL

-- COMMAND ----------

-- MAGIC %md Delete the `diamonds` table if it exists. The `create` command will not create a table if it already exists. 

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC drop table if exists diamonds

-- COMMAND ----------

-- MAGIC %md The `OK` output indicates that the command was successful. 

-- COMMAND ----------

-- MAGIC %md Create the `diamonds` table from the datafile. 

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC create temporary table diamonds 
-- MAGIC using CSV 
-- MAGIC options(path="/mnt/datalab-datasets/file-samples/diamonds.csv", 
-- MAGIC         header=TRUE)

-- COMMAND ----------

-- MAGIC %md Note that the command succeeded.

-- COMMAND ----------

-- MAGIC %md Display all columns from the `diamonds` table. 

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select *
-- MAGIC from diamonds

-- COMMAND ----------

-- MAGIC %md Note the `select` command in Spark SQL will display only the first 1000 rows of a table. 

-- COMMAND ----------

-- MAGIC %md __The End__