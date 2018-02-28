# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import argparse
import datetime
import dateutil
import glob
from io import open
import importlib
import inspect
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys

import lore
from lore import ansi, env, util
from lore.util import timer, which

try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError

try:
    reload
except NameError:
    from importlib import reload

logger = logging.getLogger(__name__)


class HelpfulParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))


def main(args=None):
    parser = HelpfulParser(prog='lore')
    parser.add_argument(
        '--version',
        action='version',
        version='lore %s' % lore.__version__
    )
    
    commands = parser.add_subparsers(help='common commands')
    
    init_parser = commands.add_parser('init', help='create a new lore project')
    init_parser.set_defaults(func=init)
    init_parser.add_argument('name', metavar='NAME', help='the name of the project')
    init_parser.add_argument('--git-ignore', default=True)
    init_parser.add_argument('--python-version', default=None)
    
    api_parser = commands.add_parser(
        'api',
        help='serve the api'
    )
    api_parser.set_defaults(func=api)
    
    console_parser = commands.add_parser(
        'console',
        help='launch an interactive python shell'
    )
    console_parser.set_defaults(func=console)
    
    exec_parser = commands.add_parser(
        'exec',
        help='run a shell command in this project\'s virtual env'
    )
    exec_parser.set_defaults(func=execute)
    
    install_parser = commands.add_parser(
        'install',
        help='install dependencies in a virtualenv'
    )
    install_parser.set_defaults(func=install)
    install_parser.add_argument(
        '--native',
        help='build optimized native dependencies (tensorflow)',
        action='store_true'
    )
    install_parser.add_argument(
        '--upgrade',
        help='recalculate requirements.frozen.txt with current versions',
        action='store_true'
    )
    
    generate_parser = commands.add_parser(
        'generate',
        help='create a new model'
    )
    generators = generate_parser.add_subparsers()
    scaffold_parser = generators.add_parser(
        'scaffold',
    )
    scaffold_parser.set_defaults(func=generate_scaffold)
    scaffold_parser.add_argument('name', metavar='NAME', help='name of the project')
    scaffold_parser.add_argument(
        '--keras',
        help='create a keras scaffold',
        action='store_true'
    )
    scaffold_parser.add_argument(
        '--xgboost',
        help='create a xgboost scaffold',
        action='store_true'
    )
    scaffold_parser.add_argument(
        '--sklearn',
        help='create a sklearn scaffold',
        action='store_true'
    )
    scaffold_parser.add_argument(
        '--holdout',
        help='create a holdout pipeline',
        action='store_true'
    )
    scaffold_parser.add_argument(
        '--regression',
        help='inherit from lore.estimators.keras.Regression',
        action='store_true'
    )
    scaffold_parser.add_argument(
        '--binary_classifier',
        help='inherit from lore.estimators.keras.BinaryClassifier',
        action='store_true'
    )
    scaffold_parser.add_argument(
        '--multi_classifier',
        help='inherit from lore.estimators.keras.MultiClassifier',
        action='store_true'
    )

    model_parser = generators.add_parser('model')
    model_parser.set_defaults(func=generate_model)
    model_parser.add_argument('name', metavar='NAME', help='name of the model')
    model_parser.add_argument(
        '--keras',
        help='inherit from lore.models.keras.Base',
        action='store_true'
    )
    model_parser.add_argument(
        '--xgboost',
        help='inherit from lore.models.xgboost.Base',
        action='store_true'
    )
    model_parser.add_argument(
        '--sklearn',
        help='inherit from lore.models.sklearn.Base',
        action='store_true'
    )

    estimator_parser = generators.add_parser('estimator')
    estimator_parser.set_defaults(func=generate_estimator)
    estimator_parser.add_argument('name', metavar='NAME', help='name of the estimator')
    estimator_parser.add_argument(
        '--keras',
        help='inherit from lore.estimators.keras.Base',
        action='store_true'
    )
    estimator_parser.add_argument(
        '--xgboost',
        help='create a xgboost scaffold',
        action='store_true'
    )
    estimator_parser.add_argument(
        '--sklearn',
        help='create a sklearn scaffold',
        action='store_true'
    )
    estimator_parser.add_argument(
        '--regression',
        help='inherit from lore.estimators.keras.Regression',
        action='store_true'
    )
    estimator_parser.add_argument(
        '--binary_classifier',
        help='inherit from lore.estimators.keras.BinaryClassifier',
        action='store_true'
    )
    estimator_parser.add_argument(
        '--multi_classifier',
        help='inherit from lore.estimators.keras.MultiClassifier',
        action='store_true'
    )

    pipeline_parser = generators.add_parser('pipeline')
    pipeline_parser.set_defaults(func=generate_pipeline)
    pipeline_parser.add_argument('name', metavar='NAME', help='name of the pipeline')
    pipeline_parser.add_argument(
        '--holdout',
        help='inherit from lore.pipelines.holdout.Base',
        action='store_true'
    )

    generate_test_parser = generators.add_parser('test')
    generate_test_parser.set_defaults(func=generate_test)
    generate_test_parser.add_argument('name', metavar='NAME', help='name of the model')
    generate_test_parser.add_argument(
        '--keras',
        help='create a keras test',
        action='store_true'
    )
    generate_test_parser.add_argument(
        '--xgboost',
        help='create a xgboost test',
        action='store_true'
    )
    generate_test_parser.add_argument(
        '--sklearn',
        help='create a sklearn test',
        action='store_true'
    )

    generate_notebooks_parser = generators.add_parser('notebooks')
    generate_notebooks_parser.set_defaults(func=generate_notebooks)
    generate_notebooks_parser.add_argument('name', metavar='NAME', help='name of the model')
    generate_notebooks_parser.add_argument(
        '--keras',
        help='create a keras notebook',
        action='store_true'
    )
    generate_notebooks_parser.add_argument(
        '--xgboost',
        help='create a xgboost notebook',
        action='store_true'
    )
    generate_notebooks_parser.add_argument(
        '--sklearn',
        help='create a sklearn notebook',
        action='store_true'
    )

    fit_parser = commands.add_parser(
        'fit',
        help="train models"
    )
    fit_parser.set_defaults(func=fit)
    fit_parser.add_argument(
        'model',
        metavar='MODEL',
        help='fully qualified model including module name. e.g. app.models.project.Model'
    )
    fit_parser.add_argument(
        '--test',
        help='calculate the loss on the prediction against the test set',
        action='store_true'
    )
    fit_parser.add_argument(
        '--score',
        help='score the model, typically inverse of loss',
        action='store_true'
    )
    fit_parser.add_argument(
        '--upload',
        help='upload model to store after fitting',
        action='store_true'
    )

    hyper_fit_parser = commands.add_parser(
        'hyper_fit',
        help="search model hyper parameters"
    )
    hyper_fit_parser.set_defaults(func=hyper_fit)
    hyper_fit_parser.add_argument(
        'model',
        metavar='MODEL',
        help='fully qualified model including module name. e.g. app.models.project.Model'
    )
    hyper_fit_parser.add_argument(
        '--test',
        help='calculate the loss on the prediction against the test set',
        action='store_true'
    )
    hyper_fit_parser.add_argument(
        '--score',
        help='score the model, typically inverse of loss',
        action='store_true'
    )
    hyper_fit_parser.add_argument(
        '--upload',
        help='upload model to store after fitting',
        action='store_true'
    )

    server_parser = commands.add_parser(
        'server',
        help='launch the flask server to provide an api to your models'
    )
    server_parser.set_defaults(func=server)
    server_parser.add_argument(
        '--host',
        help='listen on host',
    )
    server_parser.add_argument(
        '-p',
        '--port',
        help='listen on port'
    )

    pip_parser = commands.add_parser(
        'pip',
        help='pass a command to this project\'s virtual env pip'
    )
    pip_parser.set_defaults(func=pip)
    
    python_parser = commands.add_parser(
        'python',
        help='pass a command to this project\'s virtual env python'
    )
    python_parser.set_defaults(func=python)
    
    notebook_parser = commands.add_parser(
        'notebook',
        help='pass a command to this project\'s virtual env jupyter notebook'
    )
    notebook_parser.set_defaults(func=notebook)

    lab_parser = commands.add_parser(
        'lab',
        help='launch jupyter labs, with access to all envs'
    )
    lab_parser.set_defaults(func=lab)

    test_parser = commands.add_parser(
        'test',
        help='run tests'
    )
    test_parser.add_argument(
        'modules',
        nargs='?',
        help='test only certain modules, e.g. tests.unit.test_foo,tests.unit.test_bar'
    )
    test_parser.set_defaults(func=test)
    
    (known, unknown) = parser.parse_known_args(args)
    if '--env-launched' in unknown:
        unknown.remove('--env-launched')
    if '--env-checked' in unknown:
        unknown.remove('--env-checked')
    if hasattr(known, 'func'):
        known.func(known, unknown)
    else:
        parser.print_help(sys.stderr)


def api(parsed, unknown):
    api_path = os.path.join(env.root, env.project, 'api')
    endpoint_paths = []
    consumer_paths = []
    if 'HUB_LISTENERS' in os.environ:
        for name in os.environ.get('HUB_LISTENERS').split(','):
            endpoint = os.path.join(api_path, name + '_endpoint.py')
            consumer = os.path.join(api_path, name + '_consumer.py')
            if os.path.exists(endpoint):
                endpoint_paths.append(endpoint)
            elif os.path.exists(consumer):
                consumer_paths.append(consumer)
            else:
                raise IOError('No file found for listener "%s". The following paths were checked:\n  %s\n  %s' % (
                    name, consumer, endpoint))
    else:
        endpoint_paths = glob.glob(os.path.join(api_path, '*_endpoint.py'))
        consumer_paths = glob.glob(os.path.join(api_path, '*_consumer.py'))
    
    for path in endpoint_paths + consumer_paths:
        module = os.path.basename(path)[:-3]
        if sys.version_info.major == 2:
            import imp
            imp.load_source(module, path)
        elif sys.version_info.major == 3:
            import importlib.util
            spec = importlib.util.spec_from_file_location(module, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    util.strip_one_off_handlers()
    
    if len(endpoint_paths) > 0 and len(consumer_paths) > 0:
        from hub.listeners.combined import CombinedListener as Listener
    elif len(endpoint_paths) > 0:
        from hub.listeners.endpoint import EndpointListener as Listener
    elif len(consumer_paths) > 0:
        from hub.listeners.consumer import ConsumerListener as Listener
    else:
        raise IOError('No hub listeners found in %s' % api_path)
    
    try:
        Listener(
            os.environ.get('HUB_APP_NAME', env.project),
            concurrency=os.environ.get("HUB_CONCURRENCY", 4),
            host_index=os.environ.get("RABBIT_HOST_INDEX")
        ).start()
    except KeyboardInterrupt:
        exit(ansi.error('INTERRUPT') + ' Shutting down...')


def _get_valid_fit_args(method):
    if hasattr(method, '__wrapped__'):
        return _get_valid_fit_args(method.__wrapped__)
    return inspect.getargspec(method)


def _filter_private_attributes(dict):
    return {k: v for k, v in dict.items() if k[0] != '_'}


def _cast_attr(value, default):
    if isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    elif isinstance(default, datetime.date):
        return dateutil.parse(value).date()
    elif isinstance(default, datetime.datetime):
        return dateutil.parse(value)
    else:
        return value


def _get_model(name):
    module, klass = name.rsplit('.', 1)
    try:
        module = importlib.import_module(module)
        Model = getattr(module, klass)
    except (AttributeError, ModuleNotFoundError):
        sys.exit(
            ansi.error() + ' "' + name + '" does not exist in this directoy! Are you sure you typed the fully qualified module.Class name correctly?')

    return Model

def _pair_args(unknown):
    # handle args passed with ' ' or '=' between name and value
    attrs = [arg[2:] if arg[0:2] == '--' else arg for arg in unknown]  # strip --
    attrs = [attr.split('=') for attr in attrs]  # split
    attrs = [attr for sublist in attrs for attr in sublist]  # flatten
    grouped = list(zip(*[iter(attrs)] * 2))  # pair up
    unpaired = []
    if len(attrs) % 2 != 0:
        unpaired.append(attrs[-1])
    return grouped, unpaired


def fit(parsed, unknown):
    print(ansi.success('FITTING ') + parsed.model)
    Model = _get_model(parsed.model)
    model = Model()

    valid_model_fit_args = _get_valid_fit_args(model.fit)
    valid_estimator_fit_args = _get_valid_fit_args(model.estimator.fit)
    valid_fit_args = valid_model_fit_args.args[1:] + valid_estimator_fit_args.args[1:]

    model_attrs = _filter_private_attributes(model.__dict__)
    pipeline_attrs = _filter_private_attributes(model.pipeline.__dict__)
    estimator_attrs = _filter_private_attributes(model.estimator.__dict__)
    estimator_attrs.pop('model', None)

    grouped, unpaired = _pair_args(unknown)

    # assign args to their receivers
    fit_args = {}
    unknown_args = []
    for name, value in grouped:
        if name in model_attrs:
            value = _cast_attr(value, getattr(model, name))
            setattr(model, name, value)
        elif name in pipeline_attrs:
            value = _cast_attr(value, getattr(model.pipeline, name))
            setattr(model.pipeline, name, value)
        elif name in estimator_attrs:
            value = _cast_attr(value, getattr(model.estimator, name))
            setattr(model.estimator, name, value)
        elif name in valid_model_fit_args.args:
            index = valid_model_fit_args.args.index(name)
            from_end = index - len(valid_model_fit_args.args)
            default = None
            if from_end < len(valid_model_fit_args.defaults):
                default = valid_model_fit_args.defaults[from_end]
            fit_args[name] = _cast_attr(value, default)
        elif name in valid_estimator_fit_args.args:
            index = valid_estimator_fit_args.args.index(name)
            from_end = index - len(valid_estimator_fit_args.args)
            default = None
            if from_end < len(valid_estimator_fit_args.defaults):
                default = valid_estimator_fit_args.defaults[from_end]
            fit_args[name] = _cast_attr(value, default)
        else:
            unknown_args.append(name)

    unknown_args += unpaired

    if unknown_args:
        msg = ansi.bold("Valid model attributes") + ": %s\n" % ', '.join(sorted(model_attrs.keys()))
        msg += ansi.bold("Valid estimator attributes") + ": %s\n" % ', '.join(sorted(estimator_attrs.keys()))
        msg += ansi.bold("Valid pipeline attributes") + ": %s\n" % ', '.join(sorted(pipeline_attrs.keys()))
        msg += ansi.bold("Valid fit arguments") + ": %s\n" % ', '.join(sorted(valid_fit_args))
        
        sys.exit(ansi.error() + ' Unknown arguments: %s\n%s' % (unknown_args, msg))
    
    model.fit(score=parsed.score, test=parsed.test, **fit_args)
    print(ansi.success() + ' Fitting: %i\n%s' % (model.fitting, json.dumps(model.stats, indent=2)))
    
    
def hyper_fit(parsed, unknown):
    print(ansi.success('HYPER PARAM FITTING ') + parsed.model)
    # TODO
    

def server(parsed, unknown):
    host = parsed.host or os.environ.get('HOST') or '0.0.0.0'
    port = parsed.port or os.environ.get('PORT') or '5000'
    args = [env.bin_flask, 'run', '--port', port, '--host', host] + unknown
    os.environ['FLASK_APP'] = lore.env.flask_app
    os.execv(env.bin_flask, args)


def console(parsed, unknown):
    install_jupyter_kernel()
    sys.argv[0] = env.bin_jupyter
    args = [env.bin_jupyter, 'console', '--kernel', env.project] + unknown
    startup = '.ipython'
    if not os.path.exists(startup):
        with open(startup, 'w+') as file:
            file.write('import lore\n')

    print(ansi.success('JUPYTER') + ' ' + str(env.bin_jupyter))
    os.environ['PYTHONSTARTUP'] = startup
    os.execv(env.bin_jupyter, args)


def execute(parsed, unknown):
    if len(unknown) == 0:
        print(ansi.error() + ' no args to execute!')
        return
    
    print(ansi.success('EXECUTE ') + ' '.join(unknown))
    
    os.environ['PATH'] = os.path.join(env.prefix, 'bin') + ':' + os.environ['PATH']
    subprocess.check_call(unknown, env=os.environ)


def init(parsed, unknown):
    template = os.path.join(os.path.dirname(__file__), 'template', 'init')
    
    if os.path.exists(parsed.name):
        sys.exit(
            ansi.error() + ' "' + parsed.name + '" already exists in this directoy! Lore can not create a new project with this name.')
    
    shutil.copytree(template, parsed.name, symlinks=False, ignore=None)
    os.chdir(parsed.name)
    shutil.move('app', parsed.name)
    
    requirements = 'lore'
    if unknown:
        requirements += '[' + ','.join([r[2:] for r in unknown]) + ']'
    with open('requirements.txt', 'wt') as file:
        file.write(requirements)

    python_version = parsed.python_version or '3.6.4'
    
    with open('runtime.txt', 'wt') as file:
        file.write('python-' + python_version + '\n')
    reload(lore.env)
    install(parsed, unknown)


def install(parsed, unknown):
    env.validate()
    if platform.system() == 'Darwin':
        install_darwin()
    elif platform.system() == 'Linux':
        install_linux()
    elif platform.system() == 'Windows':
        print(
            ansi.warning() + ' pyenv does not '
                             'support Windows. You\'ll have the task of managing'
                             ' python versions yourself 😰'
        )
        return
    else:
        raise KeyError('unknown system: ' % platform.system())
    
    install_python_version()
    create_virtual_env()
    install_requirements(parsed)
    
    if hasattr(parsed, 'native') and parsed.native:
        install_tensorflow()


_jinja2_env = None


def _render_template(name, **kwargs):
    global _jinja2_env
    if _jinja2_env is None:
        import jinja2
        
        _jinja2_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                os.path.join(os.path.dirname(__file__), 'template')
            ),
            trim_blocks=True,
            lstrip_blocks=True
        )
    return _jinja2_env.get_template(name).render(**kwargs)


def generate_scaffold(parsed, unknown):
    generate_model(parsed, unknown)
    generate_estimator(parsed, unknown)
    generate_pipeline(parsed, unknown)
    generate_test(parsed, unknown)
    generate_notebooks(parsed, unknown)


def _generate_template(type, parsed, **kwargs):
    import inflection
    name = parsed.name
    kwargs = kwargs or {}
    kwargs['keras'] = parsed.keras
    kwargs['xgboost'] = parsed.xgboost
    kwargs['sklearn'] = parsed.sklearn
    kwargs['major_version'] = sys.version_info[0]
    kwargs['full_version'] = lore.env.python_version
    notebooks = ['features', 'architecture']
    name = inflection.underscore(name)
    if type == 'notebooks':
        for notebook in notebooks:
            _generate_template(notebook, parsed, **kwargs)
        return
        
    if type == 'test':
        destination = os.path.join(inflection.pluralize(type), 'unit', 'test_' + name + '.py')
    elif type in notebooks:
        destination = os.path.join('notebooks', name, type + '.ipynb')
    else:
        destination = os.path.join(lore.env.project, inflection.pluralize(type), name + '.py')

    if os.path.exists(destination):
        sys.exit(ansi.error() + ' %s already exists' % destination)

    dir = os.path.dirname(destination)
    if not os.path.exists(dir):
        os.makedirs(dir)
        if type not in notebooks:
            open(os.path.join(dir, '__init__.py'), 'w')
        
    kwargs['app_name'] = lore.env.project
    kwargs['module_name'] = name
    kwargs['class_name'] = inflection.camelize(name)
    code = _render_template(type + '.py.j2', **kwargs)
    
    with open(destination, 'w+') as file:
        file.write(code)
    
    print(ansi.success('CREATED ') + destination)


def generate_model(parsed, unknown):
    _generate_template('model', parsed)
    

def generate_estimator(parsed, unknown):
    base = 'Base'
    if parsed.regression:
        base = 'Regression'
    elif parsed.binary_classifier:
        base = 'BinaryClassifier'
    elif parsed.multi_classifier:
        base = 'MultiClassifier'

    _generate_template('estimator', parsed, base=base)
    

def generate_pipeline(parsed, unknown):
    if not parsed.holdout:
        sys.exit(ansi.error() + ' unknown pipeline type; try --holdout')

    _generate_template('pipeline', parsed)


def generate_test(parsed, unknown):
    _generate_template('test', parsed)


def generate_notebooks(parsed, unknown):
    _generate_template('notebooks', parsed)


def pip(parsed, unknown):
    args = [env.bin_python, '-m', 'pip'] + unknown
    print(ansi.success('EXECUTE ') + ' '.join(args))
    subprocess.check_call(args)


def python(parsed, unknown):
    args = [env.bin_python] + unknown
    print(ansi.success('EXECUTE ') + ' '.join(args))
    subprocess.check_call(args)


def test(parsed, unknown):
    with timer('boot time'):
        if 'LORE_ENV' not in os.environ:
            env.name = env.TEST
            logger.level = logging.WARN
        
        import unittest
        if parsed.modules:
            names = parsed.modules.split(',')
            print(ansi.success('RUNNING ') + 'tests in ' + str(names))
            suite = unittest.TestLoader().loadTestsFromNames(names)
        else:
            print(ansi.success('RUNNING ') + 'all tests')
            suite = unittest.defaultTestLoader.discover(env.tests_dir)
    
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    else:
        sys.exit(0)


def notebook(parsed, unknown):
    install_jupyter_kernel()
    args = [env.bin_jupyter, 'notebook'] + unknown
    print(ansi.success('JUPYTER') + ' ' + str(env.bin_jupyter))
    os.execv(env.bin_jupyter, args)


def lab(parsed, unknown):
    install_jupyter_kernel()
    args = [env.bin_jupyter, 'lab'] + unknown
    print(ansi.success('JUPYTER') + ' ' + str(env.bin_jupyter))
    os.execv(env.bin_jupyter, args)


def install_darwin():
    install_gcc_5()
    install_pyenv()
    install_graphviz()
    

def install_linux():
    install_pyenv()


def install_homebrew():
    if which('brew'):
        return
    
    print(ansi.success('INSTALL') + ' homebrew')
    subprocess.check_call((
        '/usr/bin/ruby',
        '-e',
        '"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"'
    ))


def install_pyenv():
    home = os.environ.get('HOME')
    if not home:
        return
    
    pyenv = os.path.join(home, '.pyenv')
    bin_pyenv = os.path.join(pyenv, 'bin', 'pyenv')
    virtualenv = os.path.join(pyenv, 'plugins', 'pyenv-virtualenv')
    if os.path.exists(bin_pyenv) and os.path.exists(virtualenv):
        return
    
    if os.path.exists(pyenv) and not os.path.isfile(bin_pyenv):
        print(ansi.warning() + ' pyenv executable is not present at %s' % bin_pyenv)
        while True:
            answer = input('Would you like to blow away ~/.pyenv and rebuild from scratch? [Y/n] ')
            if answer in ['', 'y', 'Y']:
                shutil.rmtree(pyenv)
                break
            elif answer in ['n', 'N']:
                sys.exit(ansi.error() + ' please fix pyenv before continuing')
            else:
                print('please enter Y or N')
    
    if not os.path.exists(pyenv):
        print(ansi.success('INSTALLING') + ' pyenv')
        subprocess.check_call((
            'git',
            'clone',
            'https://github.com/pyenv/pyenv.git',
            pyenv
        ))
    else:
        print(ansi.success('CHECK') + ' existing pyenv installation')
    env.pyenv = pyenv
    env.bin_pyenv = bin_pyenv
    env.set_python_version(env.python_version)
    
    if not os.path.exists(virtualenv):
        print(ansi.success('INSTALLING') + ' pyenv virtualenv')
        subprocess.check_call((
            'git',
            'clone',
            'https://github.com/pyenv/pyenv-virtualenv.git',
            virtualenv
        ))
    else:
        print(ansi.success('CHECK') + ' existing virtualenv installation')


def install_xcode():
    result = subprocess.call(('xcode-select', '--install'), stderr=subprocess.PIPE)
    if result > 0:
        print(ansi.success('CHECK') + ' xcode-select --install')
    else:
        print(ansi.success('INSTALL') + ' xcode command line tools')


def install_gcc_5():
    if which('gcc-5'):
        return
    
    install_homebrew()
    print(ansi.success('INSTALL') + ' gcc 5 for xgboost')
    subprocess.check_call(('brew', 'install', 'gcc@5'))


def install_bazel():
    if which('bazel'):
        return
    
    install_homebrew()
    print(ansi.success('INSTALL') + ' bazel for tensorflow')
    subprocess.check_call(('brew', 'install', 'bazel'))


def install_graphviz():
    install_homebrew()
    try:
        if subprocess.check_output(('brew', 'ls', '--versions', 'graphviz')):
            return
    except:
        pass
    print(ansi.success('INSTALL') + ' graphviz')
    subprocess.check_call(('brew', 'install', 'graphviz'))


def install_tensorflow():
    description = subprocess.check_output(
        (env.bin_python, '-m', 'pip', 'show', 'tensorflow')
    ).decode('utf-8')
    version = re.match(
        '.*^Version: ([^\n]+)', description, re.S | re.M
    ).group(1)
    if not version:
        sys.exit(ansi.error() + ' tensorflow is not in requirements.txt')
    
    print(ansi.success('NATIVE') + ' tensorflow ' + version)
    
    python_version = ''.join(env.python_version.split('.')[0:2])
    cached = os.path.join(
        env.pyenv,
        'cache',
        'tensorflow_pkg',
        'tensorflow-' + version + '-cp' + python_version + '*'
    )
    
    paths = glob.glob(cached)
    
    if not paths:
        build_tensorflow(version)
        paths = glob.glob(cached)
    
    path = paths[0]
    
    subprocess.check_call((env.bin_python, '-m', 'pip', 'uninstall', '-y', 'tensorflow'))
    print(ansi.success('INSTALL') + ' tensorflow native build')
    subprocess.check_call((env.bin_python, '-m', 'pip', 'install', path))


def build_tensorflow(version):
    install_bazel()
    print(ansi.success('BUILD') + ' tensorflow for this architecture')
    
    tensorflow_repo = os.path.join(env.pyenv, 'cache', 'tensorflow')
    cache = os.path.join(env.pyenv, 'cache', 'tensorflow_pkg')
    if not os.path.exists(tensorflow_repo):
        subprocess.check_call((
            'git',
            'clone',
            'https://github.com/tensorflow/tensorflow',
            tensorflow_repo
        ))
    
    subprocess.check_call(
        ('git', 'checkout', '--', '.'),
        cwd=tensorflow_repo
    )
    subprocess.check_call(
        ('git', 'checkout', 'master'),
        cwd=tensorflow_repo
    )
    subprocess.check_call(
        ('git', 'pull'),
        cwd=tensorflow_repo
    )
    subprocess.check_call(
        ('git', 'checkout', 'v' + version),
        cwd=tensorflow_repo
    )
    major, minor, patch = env.python_version.split('.')
    lib = os.path.join('lib', 'python' + major + '.' + minor, 'site-packages')
    new_env = {
        'PATH': os.environ['PATH'],
        'PYTHON_BIN_PATH': env.bin_python,
        'PYTHON_LIB_PATH': os.path.join(env.prefix, lib),
        'TF_NEED_MKL': '0',
        'CC_OPT_FLAGS': '-march=native -O2',
        'TF_NEED_JEMALLOC': '1',  # only available on linux regardless
        'TF_NEED_GCP': '0',
        'TF_NEED_HDFS': '0',
        'TF_ENABLE_XLA': '0',
        'TF_NEED_VERBS': '0',
        'TF_NEED_OPENCL': '0',
        'TF_NEED_S3': '0',
        'TF_NEED_GDR': '0',
        'TF_NEED_CUDA': '0',  # TODO enable CUDA when appropriate
        'TF_CUDA_CLANG': '1',
        'TF_CUDA_VERSION': '8.0.61',
        'CUDA_TOOLKIT_PATH': '/usr/local/cuda',
        'CUDNN_INSTALL_PATH': '/usr/local/cuda',
        'TF_CUDNN_VERSION': '5.1.10',
        'TF_CUDA_CLANG': '/usr/bin/gcc',
        'TF_CUDA_COMPUTE_CAPABILITIES': '3.5,5.2',
        'TF_NEED_MPI': '0'
    }
    subprocess.check_call(('./configure',), cwd=tensorflow_repo, env=new_env)
    # TODO remove this hack when tensorflow fixes their build
    # https://github.com/tensorflow/tensorflow/issues/12979
    pip = subprocess.Popen(
        (
            'sed',
            '-i',
            "'\@https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz@d'",
            'tensorflow/workspace.bzl'
        ),
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    (stdout, stderr) = pip.communicate()
    pip.wait()
    
    subprocess.check_call((
        'bazel',
        'build',
        '--config=opt',
        # '--config=cuda',  TODO enable CUDA when appropriate
        'tensorflow/tools/pip_package:build_pip_package',
    ), cwd=tensorflow_repo)
    
    subprocess.check_call((
        'bazel-bin/tensorflow/tools/pip_package/build_pip_package',
        cache
    ), cwd=tensorflow_repo)


def install_python_version():
    if env.launched():
        return
    
    if not env.python_version:
        env.set_python_version('.'.join(sys.version_info))
        print(ansi.warning() + ' %s does not exist. Creating with %s' %
              (env.version_path, env.python_version))
        with open(env.version_path, 'w', encoding='utf-8') as f:
            f.write(env.python_version + os.linesep)
    
    if not env.pyenv:
        sys.exit(
            ansi.error() + ' pyenv is not installed. Lore is broken. try:\n'
                           ' $ pip uninstall lore && pip install lore\n'
        )
    
    versions = subprocess.check_output(
        (env.bin_pyenv, 'versions', '--bare')
    ).decode('utf-8').split(os.linesep)
    if env.python_version not in versions:
        print(ansi.success('INSTALL') + ' python %s' % env.python_version)
        if platform.system() == 'Darwin':
            install_xcode()
        subprocess.check_call(('git', '-C', env.pyenv, 'pull'))
        subprocess.check_call((env.bin_pyenv, 'install', env.python_version))
        subprocess.check_call((env.bin_pyenv, 'rehash'))


def create_virtual_env():
    if env.pyenv:
        try:
            os.unlink(os.path.join(env.pyenv, 'versions', env.project))
        except OSError as e:
            pass
    
    if os.path.exists(env.bin_python):
        return
    
    print(ansi.success('CREATE') + ' virtualenv: %s' % env.project)
    subprocess.check_call((
        env.bin_pyenv,
        'virtualenv',
        env.python_version,
        env.project
    ))


def install_requirements(args):
    source = env.requirements
    if not os.path.exists(source):
        sys.exit(
            ansi.error() + ' %s is missing. You should check it in to version '
                           'control.' % ansi.underline(source)
        )
    
    pip_install(source, args)
    freeze_requirements()
    install_jupyter_kernel()


def install_jupyter_kernel():
    if os.path.exists(env.jupyter_kernel_path):
        return
    
    print(ansi.success('INSTALL') + ' jupyter kernel')
    subprocess.check_call((
        env.bin_python,
        '-m',
        'ipykernel',
        'install',
        '--user',
        '--name=' + env.project
    ))


def freeze_requirements():
    source = env.requirements
    
    print(ansi.success('EXECUTE') + ' ' + env.bin_python + ' -m pip freeze -r ' + source)
    vcs = split_vcs_lines()
    pip = subprocess.Popen(
        (env.bin_python, '-m', 'pip', 'freeze', '-r', source),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    (stdout, stderr) = pip.communicate()
    pip.wait()
    
    restore_vcs_lines(vcs)
    
    present = stdout.decode('utf-8').split(os.linesep)
    errors = stderr.decode('utf-8').split(os.linesep)
    missing = [line for line in errors if 'package is not installed' in line]
    regex = re.compile(r'contains ([\w\-\_]+)')
    needed = [m.group(1).lower() for l in missing for m in [regex.search(l)] if m]
    
    added_index = present.index('## The following requirements were added by pip freeze:')
    unsafe = None
    if added_index:
        added = present[added_index + 1:-1]
        present = set(present[0:added_index])
        safe = set()
        unsafe = set()
        for package in added:
            name = package.split('==')[0]
            
            for bad in vcs:
                if name in bad:
                    unsafe.add(package)
                    continue
            
            if name.lower() in needed:
                needed.remove(name.lower())
            
            safe.add(package)
        present |= safe
        present -= unsafe
    
    if needed:
        args = [env.bin_python, '-m', 'pip', 'install'] + needed
        print(ansi.success('EXECUTE ') + ' '.join(args))
        subprocess.check_call(args)
        return freeze_requirements()
    
    if unsafe:
        if vcs:
            print(
                ansi.warning() + ' Non pypi packages were detected in your ' +
                ansi.underline('requirements.txt') + ' that can not be '
                                                     'completely frozen by pip. ' + os.linesep + os.linesep +
                os.linesep.join(vcs)
            )
        
        print(
            ansi.info() + ' You should check the following packages in to ' +
            ansi.underline('requirements.txt') + ' or `lore pip uninstall` them'
        )
        if vcs:
            print(ansi.warning('unless') + ' they are covered by the previously listed packages'
                                           ' that pip can not freeze.')
        print(
            os.linesep + os.linesep.join(unsafe) + os.linesep
        )
    
    with open(source, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join(sorted(present, key=lambda s: s.lower())).strip() + os.linesep)
        if vcs:
            f.write(os.linesep.join(vcs) + os.linesep)


def split_vcs_lines():
    with open(env.requirements, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    vcs = [line for line in lines if
           re.match(r'^(-e )?(git|svn|hg|bzr).*', line)]
    if not vcs:
        return vcs
    
    if os.path.exists(env.requirements_vcs):
        with open(env.requirements_vcs, 'r', encoding='utf-8') as f:
            new = set(f.readlines())
            vcs = list(set(vcs).union(new))
    
    lines = list(set(lines) - set(vcs))
    with open(env.requirements, 'w', encoding='utf-8') as f:
        f.write(''.join(sorted(lines)))
    
    with open(env.requirements_vcs, 'w', encoding='utf-8') as f:
        f.write(''.join(sorted(vcs)))
    return vcs


def restore_vcs_lines(vcs):
    if not os.path.exists(env.requirements_vcs):
        return
    with open(env.requirements, 'r', encoding='utf-8') as f:
        original = f.read()
    with open(env.requirements, 'w', encoding='utf-8') as f:
        f.write(''.join(vcs) + original)
    os.remove(env.requirements_vcs)


def pip_install(path, args):
    if not os.path.exists(path):
        return
    
    pip_args = [env.bin_python, '-m', 'pip', 'install', '-r', path]
    if hasattr(args, 'upgrade') and args.upgrade:
        pip_args += ['--upgrade', '--upgrade-strategy=eager']
    print(ansi.success('EXECUTE ') + ' '.join(pip_args))
    try:
        subprocess.check_call(pip_args)
    except subprocess.CalledProcessError:
        sys.exit(
            ansi.error() + ' could not:\n $ lore pip install -r %s\nPlease try '
                           'installing failed packages manually, or upgrade failed '
                           'packages with:\n $ lore install --upgrade ' % path
        )


if __name__ == '__main__':
    main()
