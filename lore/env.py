"""
Lore Environment

Key attributes and paths for this project
"""
from __future__ import absolute_import

import glob
import locale
import os
import re
import sys
import platform
from io import open
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import socket

# WORKAROUND HACK
# Python3 inserts __PYVENV_LAUNCHER__, that breaks pyenv virtualenv
# by changing the venv python symlink to the current python, rather
# than the correct pyenv version, among other problems. We pop it
# in our process space, since python has already made it's use of it.
#
# see https://bugs.python.org/issue22490
os.environ.pop('__PYVENV_LAUNCHER__', None)

try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError

try:
    import configparser
except ModuleNotFoundError:
    configparser = None

try:
    import jupyter_core.paths
except ModuleNotFoundError:
    jupyter_core = False

from lore import ansi

TEST = 'test'
DEVELOPMENT = 'development'
PRODUCTION = 'production'

unicode_locale = True
unicode_upgraded = False

if platform.system() != 'Windows':
    if 'utf' not in locale.getpreferredencoding().lower():
        if os.environ.get('LANG', None):
            unicode_locale = False
        else:
           locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
           unicode_upgraded = True


def read_version(path):
    version = None
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            version = f.read().strip()
    
    if version:
        return re.sub(r'^python-', '', version)
    
    return version


version_path = 'runtime.txt'
python_version = None
python_version_info = None
if 'LORE_ROOT' in os.environ:
    root = os.environ.get('LORE_ROOT')
else:
    root = os.getcwd()
    while True:
        python_version = read_version(os.path.join(root, 'runtime.txt'))
        if python_version:
            break
        
        python_version = read_version(os.path.join(root, '.python-version'))
        if python_version:
            break
        
        root = os.path.dirname(root)
        if root.count(os.path.sep) == 1:
            root = os.getcwd()
            break

lib = os.path.join(root, 'lib')
if lib not in sys.path:
    sys.path.insert(0, lib)

# Load environment variables from disk
for var in glob.glob('/conf/env/*'):
    if os.path.isfile(var):
        os.environ[os.path.basename(var)] = open(var, encoding='utf-8').read()

env_file = os.path.join(root, '.env')
if os.path.isfile(env_file):
    from dotenv import load_dotenv
    load_dotenv(env_file)

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    default_name = TEST
else:
    default_name = DEVELOPMENT

host = socket.gethostname()
home = os.environ.get('HOME', root)
name = os.environ.get('LORE_ENV', default_name)
project = os.environ.get('LORE_PROJECT', root.split(os.sep)[-1])
sys.path = [root] + sys.path
work_dir = 'tests' if name == TEST else os.environ.get('WORK_DIR', root)
models_dir = os.path.join(work_dir, 'models')
data_dir = os.path.join(work_dir, 'data')
log_dir = os.path.join(root if name == TEST else work_dir, 'logs')
tests_dir = os.path.join(root, 'tests')
if jupyter_core:
    jupyter_kernel_path = os.path.join(jupyter_core.paths.jupyter_data_dir(), 'kernels', project)
else:
    jupyter_kernel_path = '[UNKNOWN] (upgrade jupyter-core)'
    
color = {
    DEVELOPMENT: ansi.GREEN,
    TEST: ansi.BLUE,
    PRODUCTION: ansi.RED,
}.get(name, ansi.YELLOW)


pyenv = os.path.join(home, '.pyenv')
if os.path.exists(pyenv):
    pyenv = os.path.realpath(pyenv)
    bin_pyenv = os.path.join(pyenv, 'bin', 'pyenv')
else:
    pyenv = None
    bin_pyenv = None

prefix = None
bin_python = None
bin_lore = None
bin_jupyter = None
bin_flask = None
flask_app = None
requirements = os.path.join(root, 'requirements.txt')
requirements_vcs = os.path.join(root, 'requirements.vcs.txt')


def set_python_version(version):
    """Set the python version for this lore project, to establish the location
    of key binaries.
    
    :param version:
    :type version: unicode
    """
    global python_version
    global python_version_info
    global prefix
    global bin_python
    global bin_lore
    global bin_jupyter
    global bin_flask
    global flask_app
    
    python_version = version
    if python_version:
        python_version_info = tuple([int(i) if i.isdigit() else i for i in version.split('.')])
        if pyenv:
            prefix = os.path.join(
                pyenv,
                'versions',
                python_version,
                'envs',
                project
            )
        elif platform.system() == 'Windows':
            prefix = os.path.join(root.lower(), '.python')
            bin_venv = os.path.join(prefix, 'scripts')
            bin_python = os.path.join(bin_venv, 'python.exe')
            bin_lore = os.path.join(bin_venv, 'lore.exe')
            bin_jupyter = os.path.join(bin_venv, 'jupyter.exe')
            bin_flask = os.path.join(bin_venv, 'flask.exe')
            flask_app = os.path.join(prefix, 'lib', 'site-packages', 'lore', 'www', '__init__.py')
            return

        else:
            prefix = os.path.realpath(sys.prefix)
            
        python_major = 'python' + str(python_version_info[0])
        python_minor = python_major + '.' + str(python_version_info[1])
        python_patch = python_minor + '.' + str(python_version_info[2])
        
        bin_python = os.path.join(prefix, 'bin', python_patch)
        if not os.path.exists(bin_python):
            bin_python = os.path.join(prefix, 'bin', python_minor)
        if not os.path.exists(bin_python):
            bin_python = os.path.join(prefix, 'bin', python_major)
        if not os.path.exists(bin_python):
            bin_python = os.path.join(prefix, 'bin', 'python')
        bin_lore = os.path.join(prefix, 'bin', 'lore')
        bin_jupyter = os.path.join(prefix, 'bin', 'jupyter')
        bin_flask = os.path.join(prefix, 'bin', 'flask')
        flask_app = os.path.join(prefix, 'lib', python_minor, 'site-packages', 'lore', 'www', '__init__.py')
    else:
        python_version_info = []
        prefix = None
        bin_python = None
        bin_lore = None
        bin_jupyter = None
        bin_flask = None

set_python_version(python_version)


def exists():
    """Test whether the current working directory has a valid lore environment.
    
    :return:  bool True if the environment is valid
    """
    return python_version is not None


def launched():
    """Test whether the current python environment is the correct lore env.

    :return:  bool True if the environment is launched
    """
    if not prefix:
        return False
    
    return os.path.realpath(sys.prefix) == os.path.realpath(prefix)


def validate():
    if not os.path.exists(os.path.join(root, project, '__init__.py')):
        sys.exit(ansi.error() + ' Python module not found. Do you need to change $LORE_PROJECT from "%s"?' % project)

    if exists():
        return

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = 'lore'
    sys.exit(
        ansi.error() + ' %s is only available in lore '
                       'project directories (missing %s)' % (
            ansi.bold(command),
            ansi.underline(version_path)
        )
    )


def launch():
    if launched():
        check_version()
        os.chdir(root)
        return
    
    if not os.path.exists(bin_lore):
        missing = ' %s virtualenv is missing.' % project
        if '--launched' in sys.argv:
            sys.exit(ansi.error() + missing + ' Please check for errors during:\n $ lore install\n')
        else:
            print(ansi.warning() + missing)
            import lore.__main__
            lore.__main__.install(None, None)
        
    reboot('--env-launched')
    

def reboot(*args):
    args = list(sys.argv) + list(args)
    if args[0] == 'python' or not args[0]:
        args[0] = bin_python
    elif os.path.basename(sys.argv[0]) in ['lore', 'lore.exe']:
        args[0] = bin_lore
    try:
        os.execv(args[0], args)
    except Exception as e:
        if args[0] == bin_lore and args[1] == 'console':
            print(ansi.error() + ' Your jupyter kernel may be corrupt. Please remove it so lore can reinstall:\n $ rm ' + jupyter_kernel_path)
        raise e


def check_version():
    if sys.version_info[0:3] == python_version_info[0:3]:
        return

    sys.exit(
        ansi.error() + ' your virtual env points to the wrong python version. '
                       'This is likely because you used a python installer that clobbered '
                       'the system installation, which breaks virtualenv creation. '
                       'To fix, check this symlink, and delete the installation of python '
                       'that it is brokenly pointing to, then delete the virtual env itself '
                       'and rerun lore install: ' + os.linesep + os.linesep + bin_python +
        os.linesep
    )


def check_requirements():
    if not os.path.exists(requirements):
        sys.exit(
            ansi.error() + ' %s is missing. Please check it in.' % ansi.underline(requirements)
        )
    
    with open(requirements, 'r', encoding='utf-8') as f:
        dependencies = f.readlines()
    
    vcs = [d for d in dependencies if re.match(r'^(-e )?(git|svn|hg|bzr).*', d)]
    
    dependencies = list(set(dependencies) - set(vcs))
    
    missing = []
    try:
        pkg_resources.require(dependencies)
    except (pkg_resources.ContextualVersionConflict, DistributionNotFound, VersionConflict) as error:
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
    else:
        return True


def get_config(path):
    if configparser is None:
        return None
    
    # Check for env specific configs first
    if os.path.exists(os.path.join(root, 'config', name, path)):
        path = os.path.join(root, 'config', name, path)
    else:
        path = os.path.join(root, 'config', path)

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

aws_config = get_config('aws.cfg')
database_config = get_config('database.cfg')
redis_config = get_config('redis.cfg')
