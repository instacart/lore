import os
from glob import glob
import logging

import lore
lore.env.require(lore.dependencies.SPARK)

os.environ['SPARK_CONF_DIR'] = os.path.join(lore.env.ROOT, 'config', 'spark')

from pyspark import SparkConf, SparkContext

log_levels = {
    lore.env.DEVELOPMENT: "INFO",
    lore.env.TEST: "WARN",
}
logger = logging.getLogger(__name__)
spark_logger = logging.getLogger("py4j")
spark_logger.setLevel(logger.level)

pyfiles = [y for x in os.walk(lore.env.APP_DIR) for y in glob(os.path.join(x[0], '*.py'))]

conf = SparkConf()
conf.setMaster("local")
conf.setAppName(lore.env.APP)

# conf.set("log4j.rootCategory", log_levels.get(lore.env.NAME, "ERROR"))

# s_logger = logging.getLogger('py4j.java_gateway')
# s_logger.setLevel(logging.ERROR)

context = SparkContext(
    conf=conf,
    # pyFiles=pyfiles
)

# for file in pyfiles:
#     context.addFile(file)
