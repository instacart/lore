import os
import configparser
import logging
logger = logging.getLogger(__name__)

from lore import env, util, ansi
from lore.env import aws_config

print(ansi.success('RUNNING') + ' Lore Tests Suite')

env.name = env.TEST

logfile = os.path.join(os.getcwd(), 'logs', env.name + '.log')
util.add_log_file_handler(logfile)
