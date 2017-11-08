# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os
import sys
import atexit

from lore import env, util, ansi
from lore.ansi import underline
from lore.util import timer

logger = logging.getLogger(__name__)

if not (sys.version_info.major == 3 and sys.version_info.minor >= 6):
    ModuleNotFoundError = ImportError


__author__ = 'Montana Low and Jeremy Stanley'
__copyright__ = 'Copyright © 2017, Instacart'
__credits__ = ['Montana Low', 'Jeremy Stanley', 'Emmanuel Turlay']
__license__ = 'MIT'
__version__ = '0.4.45'
__maintainer__ = 'Montana Low'
__email__ = 'montana@instacart.com'
__status__ = 'Development Status :: 3 - Alpha'


def banner():
    import socket
    import getpass
    
    return '%s in %s on %s' % (
        ansi.foreground(ansi.GREEN, env.project),
        ansi.foreground(env.color, env.name),
        ansi.foreground(ansi.CYAN,
                        getpass.getuser() + '@' + socket.gethostname())
    )


lore_no_env = False
if hasattr(sys, 'lore_no_env'):
    lore_no_env = sys.lore_no_env

if len(sys.argv) > 1 and sys.argv[0][-4:] == 'lore' and sys.argv[1] in ['install', 'init']:
    lore_no_env = True

if not lore_no_env:
    # everyone else gets validated and launched on import
    env.validate()
    env.launch()

if env.launched():
    print(banner())
    logger.info(banner())
    logger.debug('python environment: %s' % env.prefix)

    if not lore_no_env:
        with timer('check requirements', logging.DEBUG):
            install_missing = env.name in [env.DEVELOPMENT, env.TEST]
            env.check_requirements(install_missing)
        
    try:
        with timer('numpy init', logging.DEBUG):
            import numpy
        
            numpy.random.seed(1)
            logger.debug('numpy.random.seed(1)')
    except ModuleNotFoundError as e:
        pass

    try:
        with timer('rollbar init', logging.DEBUG):
            import rollbar
            rollbar.init(
                os.environ.get("ROLLBAR_ACCESS_TOKEN", None),
                allow_logging_basic_config=False,
                environment=env.name,
                enabled=(env.name != env.DEVELOPMENT),
                handler='blocking',
                locals={"enabled": True})
    
            def report_error(exc_type, value, tb):
                import traceback
                logger.critical('Exception: %s' % ''.join(
                    traceback.format_exception(exc_type, value, tb)))
                if hasattr(sys, 'ps1'):
                    print(''.join(traceback.format_exception(exc_type, value, tb)))
                else:
                    rollbar.report_exc_info((exc_type, value, tb))
            sys.excepthook = report_error

    except ModuleNotFoundError as e:
        def report_error(exc_type, value, tb):
            import traceback
            logger.critical('Exception: %s' % ''.join(
                traceback.format_exception(exc_type, value, tb)))
            
        sys.excepthook = report_error
        pass
