import inspect
import importlib
import json
import logging
import pkgutil


import lore
import lore.util
import lore.env
from lore.env import require
from lore.util import timer

require(
    lore.dependencies.PANDAS +
    lore.dependencies.FLASK
)
import pandas
from flask import Flask, request


app = Flask(lore.env.APP)

logger = logging.getLogger(__name__)


@app.route('/')
def index():
    names = str([name for _, name, _ in pkgutil.iter_modules([lore.env.APP + '/' + 'models'])])
    return 'Hello %s!' % lore.env.APP + '\n' + names


for module_finder, module_name, _ in pkgutil.iter_modules([lore.env.APP + '/' + 'models']):
    module = importlib.import_module(lore.env.APP + '.models.' + module_name)
    for class_name, member in inspect.getmembers(module):
        if not (inspect.isclass(member) and issubclass(member, lore.models.base.Base)):
            continue

        qualified_name = module_name + '.' + class_name
        with timer('load %s' % qualified_name):
            best = member.load()

        def predict():
            logger.debug(request.args)
            data = {arg: request.args.getlist(arg) for arg in request.args.keys()}
            try:
                data = pandas.DataFrame(data)
            except ValueError:
                return 'Malformed data!', 400

            logger.debug(data)
            try:
                result = best.predict(data)
            except KeyError as ex:
                return 'Missing data!', 400
            return json.dumps(result.tolist()), 200

        predict.__name__ = best.name + '.predict'

        rule = '/' + qualified_name + '/predict.json'
        logger.info('Adding url rule for prediction: %s' % rule)
        app.add_url_rule(rule, view_func=predict)
