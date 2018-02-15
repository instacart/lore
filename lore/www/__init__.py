import logging
import pkgutil

import lore
import lore.util
import lore.env

from flask import Flask, request


app = Flask(lore.env.project)

logger = logging.getLogger(__name__)

_models = pkgutil.iter_modules([lore.env.project + '/' + 'models'])

@app.route('/')
def index():
    names = str([name for _, name, _ in pkgutil.iter_modules([lore.env.project + '/' + 'models'])])
    return 'Hello %s!' % lore.env.project + '\n' + names

for _, name, _ in _models:
    def defined_predict(name):
        def predict_name():
            logger.debug('watchout yall, gonna predict for model %s(%s)' % (name, request.args))
            return name + '\n' + str(request.args)
        predict_name.__name__ = name + '_predict'
        return predict_name

    app.route('/' + name + '/predict')(defined_predict(name))

