# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import logging
import sys
import os

import lore.dependencies
from lore import env, util, ansi
from lore.util import timer


logger = logging.getLogger(__name__)


__author__ = 'Montana Low and Jeremy Stanley'
__copyright__ = 'Copyright © 2017, Instacart'
__credits__ = ['Montana Low', 'Jeremy Stanley', 'Emmanuel Turlay', 'Shrikar Archak']
__license__ = 'MIT'
__version__ = '0.6.14'
__maintainer__ = 'Montana Low'
__email__ = 'montana@instacart.com'
__status__ = 'Development Status :: 4 - Beta'


def banner():
    import socket
    import getpass

    return '%s in %s on %s with %s & %s' % (
        ansi.foreground(ansi.GREEN, env.APP),
        ansi.foreground(env.COLOR, env.NAME),
        ansi.foreground(
            ansi.CYAN,
            getpass.getuser() + '@' + socket.gethostname()
        ),
        ansi.foreground(ansi.YELLOW, 'Python ' + env.PYTHON_VERSION),
        ansi.foreground(ansi.YELLOW, 'Lore ' + __version__)
    )


lore_no_env = False
if hasattr(sys, 'lore_no_env'):
    lore_no_env = sys.lore_no_env


no_env_commands = ['--version', 'install', 'init', 'server']
if len(sys.argv) > 1 and os.path.basename(sys.argv[0]) in ['lore', 'lore.exe'] and sys.argv[1] in no_env_commands:
    lore_no_env = True

if not lore_no_env:
    # everyone else gets validated and launched on import
    env.validate()
    env.launch()

if env.launched():
    print(banner())
    logger.info(banner())
    logger.debug('python environment: %s' % env.PREFIX)

    if not lore_no_env:
        with timer('check requirements', logging.DEBUG):
            env.check_requirements()
