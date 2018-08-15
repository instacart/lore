import os
from glob import glob
import logging
import lore
lore.env.require(lore.dependencies.PYSPARK)
os.environ['SPARK_CONF_DIR'] = os.path.join(lore.env.ROOT, 'config', 'spark')
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

SPARK_MASTER = os.environ.get('SPARK_MASTER', 'local[4]')

spark_conf = SparkConf().setAppName(lore.env.APP).setMaster(SPARK_MASTER).set('spark.driver.memory', '5G')       

spark_context = SparkContext(conf=spark_conf)
spark = SparkSession(spark_context)
