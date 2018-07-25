# -*- coding: utf-8 -*-
"""
Lore Environment
****************

Lore maintains an independent python virtualenv for each app, along with
several ways to set environment variables that allow Lore apps
apps to be 100% replicated from development to production, without any
day to day effort on the behalf of developers. There is no manual activation,
or magic env vars, or hidden files that break python for everything else. No
knowledge required of venv, pyenv, pyvenv, virtualenv, virtualenvwrapper,
pipenv, conda. Ain’t nobody got time for that.

The first thing `lore` does when launched from a command line, is to find
the correct virtualenv, with the perfect set of dependencies and relaunch
the same command in that environment. Virtualenv names are based on the
Lore app name, so if you have two apps with the same name, they will share
a virtualenv by default.

:any:`lore.env` provides constants to make working with the correct
executables easy. The most common is :any:`lore.env.NAME`.
"""
from __future__ import absolute_import, print_function, unicode_literals

import glob
import locale
import os
import re
import socket
import subprocess
import sys
import platform
from io import open

import pkg_resources

from lore import ansi


# -- Python 2/3 Compatability ------------------------------------------------

if hasattr(__builtins__, 'ModuleNotFoundError'):
    ModuleNotFoundError = __builtins__.ModuleNotFoundError
else:
    ModuleNotFoundError = ImportError

try:
    reload = reload
except NameError:
    from importlib import reload

try:
    StandardError = StandardError
except NameError:
    StandardError = Exception

try:
    import configparser
except ModuleNotFoundError:
    import ConfigParser as configparser

try:
    from urllib.parse import urlparse as parse_url
    from urllib.request import urlretrieve as retrieve_url
except ModuleNotFoundError:
    from urlparse import urlparse as parse_url
    from urllib import urlretrieve as retrieve_url

# WORKAROUND HACK
# Python3 inserts __PYVENV_LAUNCHER__, that breaks pyenv virtualenv
# by changing the venv python symlink to the current python, rather
# than the correct pyenv version, among other problems. We pop it
# in our process space, since python has already made it's use of it.
#
# see https://bugs.python.org/issue22490
os.environ.pop('__PYVENV_LAUNCHER__', None)


_new_requirements = False


def require(packages):
    """Ensures that a pypi package has been installed into the App's python environment.
    If not, the package will be installed and your env will be rebooted.

    Example:
        ::

            lore.env.require('pandas')
            # -> pandas is required. Dependencies added to requirements.txt

    :param packages: requirements.txt style name and versions of packages
    :type packages: [unicode]

    """
    global INSTALLED_PACKAGES, _new_requirements

    if _new_requirements:
        INSTALLED_PACKAGES = None

    set_installed_packages()
    if not INSTALLED_PACKAGES:
        return

    if not isinstance(packages, list):
        packages = [packages]

    missing = []
    for package in packages:
        name = re.split(r'[!<>=]', package)[0].lower()
        if name not in INSTALLED_PACKAGES:
            print(ansi.info() + ' %s is required.' % package)
            missing += [package]

    if missing:
        mode = 'a' if os.path.exists(REQUIREMENTS) else 'w'
        with open(REQUIREMENTS, mode) as requirements:
            requirements.write('\n' + '\n'.join(missing) + '\n')
        print(ansi.info() + ' Dependencies added to requirements.txt. Rebooting.')
        _new_requirements = True
        import lore.__main__
        lore.__main__.install(None, None)
        reboot('--env-checked')


def exists():
    """Test whether a lore environmnet can be found from the current working directory.

    :return: :any:`True` if the environment exists
    :rtype: bool
    """
    return PYTHON_VERSION is not None


def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  :any:`True` if the environment is launched
    :rtype: bool
    """
    if not PREFIX:
        return False

    return os.path.realpath(sys.prefix) == os.path.realpath(PREFIX)


def validate():
    """Display error messages and exit if no lore environment can be found.
    """
    if not os.path.exists(os.path.join(ROOT, APP, '__init__.py')):
        message = ansi.error() + ' Python module not found.'
        if os.environ.get('LORE_APP') is None:
            message += ' $LORE_APP is not set. Should it be different than "%s"?' % APP
        else:
            message += ' $LORE_APP is set to "%s". Should it be different?' % APP
        sys.exit(message)

    if exists():
        return

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = 'lore'
    sys.exit(
        ansi.error() + ' %s is only available in lore '
                       'app directories (missing %s)' % (
            ansi.bold(command),
            ansi.underline(VERSION_PATH)
        )
    )


def launch():
    """Ensure that python is running from the Lore virtualenv past this point.
    """
    if launched():
        check_version()
        os.chdir(ROOT)
        return

    if not os.path.exists(BIN_LORE):
        missing = ' %s virtualenv is missing.' % APP
        if '--launched' in sys.argv:
            sys.exit(ansi.error() + missing + ' Please check for errors during:\n $ lore install\n')
        else:
            print(ansi.warning() + missing)
            import lore.__main__
            lore.__main__.install(None, None)

    reboot('--env-launched')


def reboot(*args):
    """Reboot python in the Lore virtualenv
    """
    args = list(sys.argv) + list(args)
    if args[0] == 'python' or not args[0]:
        args[0] = BIN_PYTHON
    elif os.path.basename(sys.argv[0]) in ['lore', 'lore.exe']:
        args[0] = BIN_LORE
    try:
        os.execv(args[0], args)
    except Exception as e:
        if args[0] == BIN_LORE and args[1] == 'console' and JUPYTER_KERNEL_PATH:
            print(ansi.error() + ' Your jupyter kernel may be corrupt. Please remove it so lore can reinstall:\n $ rm ' + JUPYTER_KERNEL_PATH)
        raise e


def check_version():
    """Sanity check version information for corrupt virtualenv symlinks
    """
    if sys.version_info[0:3] == PYTHON_VERSION_INFO[0:3]:
        return

    sys.exit(
        ansi.error() + ' your virtual env points to the wrong python version. '
                       'This is likely because you used a python installer that clobbered '
                       'the system installation, which breaks virtualenv creation. '
                       'To fix, check this symlink, and delete the installation of python '
                       'that it is brokenly pointing to, then delete the virtual env itself '
                       'and rerun lore install: ' + os.linesep + os.linesep + BIN_PYTHON +
        os.linesep
    )


def check_requirements():
    """Make sure all listed packages from requirements.txt have been installed into the virtualenv at boot.
    """
    if not os.path.exists(REQUIREMENTS):
        sys.exit(
            ansi.error() + ' %s is missing. Please check it in.' % ansi.underline(REQUIREMENTS)
        )

    with open(REQUIREMENTS, 'r', encoding='utf-8') as f:
        dependencies = f.readlines()

    vcs = [d for d in dependencies if re.match(r'^(-e )?(git|svn|hg|bzr).*', d)]

    dependencies = list(set(dependencies) - set(vcs))

    missing = []
    try:
        pkg_resources.require(dependencies)
    except (
        pkg_resources.ContextualVersionConflict,
        pkg_resources.DistributionNotFound,
        pkg_resources.VersionConflict
    ) as error:
        missing.append(str(error))
    except pkg_resources.RequirementParseError:
        pass

    if missing:
        missing = ' missing requirement:\n  ' + os.linesep.join(missing)
        if '--env-checked' in sys.argv:
            sys.exit(ansi.error() + missing + '\nRequirement installation failure, please check for errors in:\n $ lore install\n')
        else:
            print(ansi.warning() + missing)
            import lore.__main__
            lore.__main__.install_requirements(None)
            reboot('--env-checked')


def get_config(path):
    """Load a config from disk

    :param path: target config
    :type path: unicode
    :return:
    :rtype: configparser.Config
    """
    if configparser is None:
        return None

    # Check for env specific configs first
    if os.path.exists(os.path.join(ROOT, 'config', NAME, path)):
        path = os.path.join(ROOT, 'config', NAME, path)
    else:
        path = os.path.join(ROOT, 'config', path)

    if not os.path.isfile(path):
        return None

    conf = open(path, 'rt').read()
    conf = os.path.expandvars(conf)

    config = configparser.SafeConfigParser()
    if sys.version_info[0] == 2:
        from io import StringIO
        config.readfp(StringIO(unicode(conf)))
    else:
        config.read_string(conf)
    return config


def read_version(path):
    """Attempts to read a python version string from a runtime.txt file

    :param path: to source of the string
    :return: python version
    :rtype: unicode or None
    """
    version = None
    if os.path.exists(path):
        version = open(path, 'r', encoding='utf-8').read().strip()

    if version:
        return re.sub(r'^python-', '', version)

    return version


def extend_path():
    """Adds Lore App modules to the path to making importing easy, including :any:`LIB`
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    if LIB not in sys.path:
        sys.path.insert(0, LIB)


def load_env_file():
    """Adds environment variables defined in :any:`ENV_FILE` to os.environ.
       Supports bash style comments and variable interpolation.
    """
    if not os.path.exists(ENV_FILE):
        return

    for line in open(ENV_FILE, 'r'):
        line = line.strip()
        if not line:
            continue
        name, value = line.split('=', 1)
        if not name or not value or name.startswith('#') or len(name) == 0 or name.isspace():
            continue
        if re.match(r'^(["\']).*\1$', value):
            if value.startswith('"'):
                value = os.path.expandvars(value)
            value = value[1:-1]
        os.environ[name] = value


def load_env_directory():
    """Adds environment variables defined in :any:`ENV_DIRECTORY` to os.environ.
       Each file will be added to os.environ via filename = contents.
       Supports bash style comments and variable interpolation.
    """
    for var in glob.glob(os.path.join(ENV_DIRECTORY, '*')):
        if os.path.isfile(var):
            os.environ[os.path.basename(var)] = os.path.expandvars(open(var, encoding='utf-8').read())


def set_installed_packages():
    """Idempotently caches the list of packages installed in the virtualenv.
       Can be run safely before the virtualenv is created, and will be rerun
       afterwards.
    """
    global INSTALLED_PACKAGES, REQUIRED_VERSION
    if INSTALLED_PACKAGES:
        return

    if os.path.exists(BIN_PYTHON):
        pip = subprocess.Popen(
            (BIN_PYTHON, '-m', 'pip', 'freeze'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        (stdout, stderr) = pip.communicate()
        pip.wait()

        INSTALLED_PACKAGES = [r.decode().split('==')[0].lower() for r in stdout.split()]
        REQUIRED_VERSION = next((package for package in INSTALLED_PACKAGES if re.match(r'^lore[!<>=]', package)), None)
        if REQUIRED_VERSION:
            REQUIRED_VERSION = re.split(r'[!<>=]', REQUIRED_VERSION)[-1]


def set_python_version(python_version):
    global PYTHON_VERSION, PYTHON_VERSION_INFO, PREFIX, BIN_PYTHON, BIN_LORE, BIN_JUPYTER, BIN_FLASK, FLASK_APP

    PYTHON_VERSION = python_version

    if PYTHON_VERSION:
        PYTHON_VERSION_INFO = tuple([int(i) if i.isdigit() else i for i in PYTHON_VERSION.split('.')])
        if platform.system() == 'Windows':
            PREFIX = os.path.join(ROOT.lower(), '.python')
            bin_venv = os.path.join(PREFIX, 'scripts')
            BIN_PYTHON = os.path.join(bin_venv, 'python.exe')
            BIN_LORE = os.path.join(bin_venv, 'lore.exe')
            BIN_JUPYTER = os.path.join(bin_venv, 'jupyter.exe')
            BIN_FLASK = os.path.join(bin_venv, 'flask.exe')
            FLASK_APP = os.path.join(PREFIX, 'lib', 'site-packages', 'lore', 'www', '__init__.py')
        else:
            sys_prefix = os.path.realpath(sys.prefix)
            sys_version = '%s.%s.%s' % (sys.version_info[0], sys.version_info[1], sys.version_info[2])
            if ROOT in sys_prefix and PYTHON_VERSION == sys_version:
                # launched python installed in a subdirectory of the App that has the correct version
                PREFIX = sys_prefix
            else:
                PREFIX = os.path.join(
                    PYENV,
                    'versions',
                    PYTHON_VERSION,
                    'envs',
                    APP
                )
            python_major = 'python' + str(PYTHON_VERSION_INFO[0])
            python_minor = python_major + '.' + str(PYTHON_VERSION_INFO[1])
            python_patch = python_minor + '.' + str(PYTHON_VERSION_INFO[2])

            BIN_PYTHON = os.path.join(PREFIX, 'bin', python_patch)
            if not os.path.exists(BIN_PYTHON):
                BIN_PYTHON = os.path.join(PREFIX, 'bin', python_minor)
            if not os.path.exists(BIN_PYTHON):
                BIN_PYTHON = os.path.join(PREFIX, 'bin', python_major)
            if not os.path.exists(BIN_PYTHON):
                BIN_PYTHON = os.path.join(PREFIX, 'bin', 'python')
            BIN_LORE = os.path.join(PREFIX, 'bin', 'lore')
            BIN_JUPYTER = os.path.join(PREFIX, 'bin', 'jupyter')
            BIN_FLASK = os.path.join(PREFIX, 'bin', 'flask')
            FLASK_APP = os.path.join(PREFIX, 'lib', python_minor, 'site-packages', 'lore', 'www', '__init__.py')


# -- Check Local -------------------------------------------------------------
# It's critical to check locale.getpreferredencoding() before changing os.environ, to see what python actually has configured.
UNICODE_LOCALE = True  #: does the current python locale support unicode?
UNICODE_UPGRADED = False  #: did lore change current system locale for unicode support?

if platform.system() != 'Windows':
    if 'utf' not in locale.getpreferredencoding().lower():
        if os.environ.get('LANG', None):
            UNICODE_LOCALE = False
        else:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            UNICODE_UPGRADED = True

# -- Load Environment --------------------------------------------------------
ENV_FILE = os.environ.get('ENV_FILE', '.env')  #: environment variables will be loaded from this file first
load_env_file()

ENV_DIRECTORY = os.environ.get('ENV_DIRECTORY', '/conf/env')  #: more environment variables will be loaded from files in this directory
load_env_directory()

# -- Environment Names -------------------------------------------------------
TEST = 'test'  #: environment that definitely should reflect exactly what happens in production
DEVELOPMENT = 'development'  #: environment for mucking about
PRODUCTION = 'production'  #: environment that actually matters
DEFAULT_NAME = DEVELOPMENT  #: the environment you get when you just can't be bothered to care

# -- Key Paths ---------------------------------------------------------------
PYTHON_VERSION_INFO = []  #: Parsed version of python required by this Lore app.
PREFIX = None  #: path to the Lore app virtualenv
BIN_PYTHON = None  #: path to virtualenv python executable
BIN_LORE = None  #: path to virtualenv lore executable
BIN_JUPYTER = None  #: path to virtualenv jupyter executable
BIN_FLASK = None  #: path to virtualenv flask executable
FLASK_APP = None  #: path to the current lore app's flask app

VERSION_PATH = 'runtime.txt'  #: Path to the specification of this apps Python version.
PYTHON_VERSION = os.environ.get('LORE_PYTHON_VERSION', None)  #: Version of python required by this Lore app.
ROOT = os.environ.get('LORE_ROOT', None)  #: Relative root for all app files. Determined by :envvar:`LORE_ROOT`, or iterating up directories until a :file:`runtime.txt` is found. If no :file:`runtime.txt` is found :any:`os.getcwd` is used.

if ROOT:
    if not PYTHON_VERSION:
        PYTHON_VERSION = read_version(os.path.join(ROOT, VERSION_PATH))
else:
    ROOT = os.getcwd()
    if not PYTHON_VERSION:
        while True:
            PYTHON_VERSION = read_version(os.path.join(ROOT, VERSION_PATH))
            if PYTHON_VERSION:
                break

            ROOT = os.path.dirname(ROOT)
            if ROOT.count(os.path.sep) == 1:
                ROOT = os.getcwd()
                break

ROOT = os.path.realpath(ROOT)
HOME = os.environ.get('HOME', ROOT)  #: :envvar:`HOME` directory of the current user or ``ROOT`` if unset
APP = os.environ.get('LORE_APP', ROOT.split(os.sep)[-1])  #: The name of this Lore app
REQUIREMENTS = os.path.join(ROOT, 'requirements.txt')  #: requirement files
REQUIREMENTS_VCS = os.path.join(ROOT, 'requirements.vcs.txt')

PYENV = os.environ.get('PYENV_ROOT', os.path.join(HOME, '.pyenv'))  #: Path to pyenv root
if os.path.exists(PYENV):
    PYENV = os.path.realpath(PYENV)
BIN_PYENV = os.path.join(PYENV, 'bin', 'pyenv')  #: path to pyenv executable

set_python_version(PYTHON_VERSION)

HOST = socket.gethostname()  #: current machine name: :any:`socket.gethostname`
NAME = os.environ.get('LORE_ENV', TEST if len(sys.argv) > 1 and sys.argv[1] == 'test' else DEVELOPMENT)  #: current environment name, e.g. :code:`'development'`, :code:`'test'`, :code:`'production'`
WORK_DIR = 'tests' if NAME == TEST else os.environ.get('WORK_DIR', ROOT)  #: root for disk based work
MODELS_DIR = os.path.join(WORK_DIR, 'models')  #: disk based model store
DATA_DIR = os.path.join(WORK_DIR, 'data')  #: disk based caching and data dependencies
LOG_DIR = os.path.join(ROOT if NAME == TEST else WORK_DIR, 'logs')  #: log file storage
TESTS_DIR = os.path.join(ROOT, 'tests')  #: Lore app test suite

LIB = os.path.join(ROOT, 'lib')  #: packages in :file:`./lib` are also available for import in the Lore app.

extend_path()

JUPYTER_KERNEL_PATH = None
try:
    import jupyter_core.paths
    JUPYTER_KERNEL_PATH = os.path.join(jupyter_core.paths.jupyter_data_dir(), 'kernels', APP)  #: location of jupyter kernels
except ModuleNotFoundError:
    pass

STDOUT_EXCEPTIONS = True  #: exceptions are always logged, in addition they can be printed to any:`sys.stdout`

# -- Package cache -----------------------------------------------------------
INSTALLED_PACKAGES = None
REQUIRED_VERSION = None

# -- UI ----------------------------------------------------------------------
COLOR = {
    DEVELOPMENT: ansi.GREEN,
    TEST: ansi.BLUE,
    PRODUCTION: ansi.RED,
}.get(NAME, ansi.YELLOW)  #: color code environment names for logging

# -- Config Files ------------------------------------------------------------
AWS_CONFIG = get_config('aws.cfg')
DATABASE_CONFIG = get_config('database.cfg')
REDIS_CONFIG = get_config('redis.cfg')
