# Databricks notebook source
# MAGIC %md #Analyzing Twitter data using NetworkX

# COMMAND ----------

# MAGIC %md ####Purpose of the study:

# COMMAND ----------

# MAGIC %sh /databricks/python3/bin/pip3 install networkx

# COMMAND ----------

import networkx as nx

# COMMAND ----------

import pickle
friends = pickle.load(open( "/dbfs/FileStore/tmp/hugo_following_data.pkl", "rb" ) )
followers = pickle.load(open( "/dbfs/FileStore/tmp/hugo_follower_data.pkl", "rb" ) )

# COMMAND ----------

friends.head()

# COMMAND ----------

followers.head()

# COMMAND ----------

hugo=nx.DiGraph()

# COMMAND ----------

for friend in friends["name"]:
  hugo.add_edge("hugobowne", friend)

# COMMAND ----------

for follower in followers["name"]:
  hugo.add_edge(follower, "hugobowne")

# COMMAND ----------

len(hugo.nodes())

# COMMAND ----------

len(hugo.edges())

# COMMAND ----------

sorted(hugo.degree(), key=lambda item: item[1], reverse=True)

# COMMAND ----------

