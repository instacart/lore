import os

from lore import env, util
import lore
lore.env.require(lore.dependencies.TEST)

env.NAME = env.TEST

logfile = os.path.join(os.getcwd(), 'logs', env.NAME + '.log')
util.add_log_file_handler(logfile)
