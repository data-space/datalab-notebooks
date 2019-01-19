// Databricks notebook source
// MAGIC 
// MAGIC %md # Mount `datalab-datasets`

// COMMAND ----------

// MAGIC %md ![alt](http://www.scatter.com/images/DataLab_logo.jpg)

// COMMAND ----------

// MAGIC %md After running the following two code cells， your Databricks account will have access to a collection of datasets made available by the Data Lab. This includes datasets for the course MA705/MA402A. These files/sub-directories will be (once you run these two commands) in the directory `/dbfs/mnt/datlab-datasets`.
// MAGIC 
// MAGIC These two cells only need to be run once. If you run them a second time the second command will return the exception `Directory already mounted`. This is OK and is not really an error. 
// MAGIC 
// MAGIC Go ahead and run the two cells below. Note that this is Scala code, not R code. 

// COMMAND ----------

val ACCESS_KEY = "AKIAJ6BO6D7ODSLYIEHA"
val SECRET_KEY = "G45nc02RvLgDm5Ob+dYCgK5yvHtjWOD3By3TnsNN"
val ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")
val AWS_BUCKET_NAME = "datalab-datasets"
val MOUNT_NAME      = "datalab-datasets"

// COMMAND ----------

dbutils.fs.mount(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$AWS_BUCKET_NAME", 
                 s"/mnt/$MOUNT_NAME")

// COMMAND ----------

// MAGIC %md Check that the above command worked as expected (and look for the sub-directories we expect.)

// COMMAND ----------

// MAGIC %sh ls /dbfs/mnt/datalab-datasets/

// COMMAND ----------

// MAGIC %md You should see directories called `JSON` and `texts`.

// COMMAND ----------

// MAGIC %md Next: open the `Contents data-lab` notebook to explore the contents of `'/dbfs/mnt/datalab-datasets`.

// COMMAND ----------

// MAGIC %md __The End__