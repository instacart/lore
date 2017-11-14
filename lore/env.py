"""
Lore Environment

Key attributes and paths for this project
"""
from __future__ import absolute_import

try:
    import configparser
except ImportError:
    configparser = None
    
import glob
import os
import re
import sys
from io import open
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import socket

from lore import ansi

TEST = 'test'
DEVELOPMENT = 'development'
PRODUCTION = 'production'


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
        if root == '/':
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
        else:
            prefix = os.path.realpath(sys.prefix)
            
        bin_python = os.path.join(prefix, 'bin', 'python' + '.'.join([str(i) for i in python_version_info[0:2]]))
        if not os.path.exists(bin_python):
            bin_python = os.path.join(prefix, 'bin', 'python' + str(python_version_info[0]))
        if not os.path.exists(bin_python):
            bin_python = os.path.join(prefix, 'bin', 'python')
        bin_lore = os.path.join(prefix, 'bin', 'lore')
        bin_jupyter = os.path.join(prefix, 'bin', 'jupyter')
    else:
        python_version_info = []
        prefix = None
        bin_python = None
        bin_lore = None
        bin_jupyter = None

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


def launch(install_missing=False):
    if launched():
        check_version()
        os.chdir(root)
        return
    
    if not os.path.exists(bin_lore):
        missing = ' %s virtualenv is missing.' % project
        if install_missing:
            print(ansi.warning() + missing)
            import lore.__main__
            lore.__main__.install(None)
            return launch(False)
        else:
            sys.exit(ansi.error() + missing + ' Please run:\n $ lore install\n')

    if sys.argv[0] == 'python' or not sys.argv[0]:
        sys.argv[0] = bin_python
    elif sys.argv[0][-4:] == 'lore':
        sys.argv[0] = bin_lore

    os.execv(sys.argv[0], sys.argv)


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


def check_requirements(install_missing=False):
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
        if install_missing:
            print(ansi.warning() + missing)
            import lore.__main__
            lore.__main__.install_requirements(None)
            return check_requirements(False)
        else:
            sys.exit(ansi.error() + missing + '\nPlease run:\n $ lore install\n')


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
