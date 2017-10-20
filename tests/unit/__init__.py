import os

from lore import env, util, ansi

print(ansi.success('RUNNING') + ' Lore Tests Suite')

env.name = env.TEST

logfile = os.path.join(os.getcwd(), 'logs', env.name + '.log')
util.add_log_file_handler(logfile)
