# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import logging
import logging.handlers
import os
import re
import sys
import six
from contextlib import contextmanager
from datetime import datetime

from lore import ansi, env

if sys.version_info.major == 3:
    import shutil


class SecretFilter(logging.Filter):
    PASSWORD_MATCH = re.compile(
        r'((secret|key|access|pass|pw)[^\s]*\s*[=:]\s*)[^\s]+',
        flags=re.IGNORECASE
    )
    URL_MATCH = re.compile(
        r'://([^:]+):([^@]+)(@.+)',
        flags=re.IGNORECASE
    )
    
    def filter(self, record):
        if record is None:
            return True
        record.msg = str(record.msg)
        record.msg = re.sub(SecretFilter.PASSWORD_MATCH, r'\1XXX', record.msg)
        record.msg = re.sub(SecretFilter.URL_MATCH, r'://XXX:XXX\3', record.msg)
        return True


class ConsoleFormatter(logging.Formatter):
    colors = {
        logging.DEBUG: ansi.debug,
        logging.INFO: ansi.info,
        logging.WARNING: ansi.warning,
        logging.ERROR: ansi.error,
        logging.CRITICAL: ansi.critical
    }
    
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        timestamp = timestamp + '.%03d' % record.msecs
        level = '%-8s' % record.levelname
        level = ConsoleFormatter.colors[record.levelno](level)
        location = ansi.foreground(ansi.CYAN, record.name) + ':' + \
                   ansi.foreground(ansi.CYAN, str(record.lineno))

        msg = record.msg
        if record.args:
            msg = msg % record.args
        return '%s  %s %s => %s' % (timestamp, level, location, msg)


logger = logging.getLogger()

def add_log_file_handler(path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    handler = logging.FileHandler(path)
    handler.setFormatter(ConsoleFormatter())
    handler.addFilter(SecretFilter())
    logger.addHandler(handler)

def add_log_stream_handler(stream=sys.stdout):
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ConsoleFormatter())
    handler.addFilter(SecretFilter())
    logger.addHandler(handler)

def add_syslog_handler(address):
    syslog = logging.handlers.SysLogHandler(address=address)
    syslog.setFormatter(logging.Formatter(fmt='%(name)s:%(lineno)s %(message)s'))
    syslog.addFilter(SecretFilter())
    logger.addHandler(syslog)

if env.exists():
    logfile = os.path.join(env.log_dir, env.name + '.log')
    add_log_file_handler(logfile)


class PrintInterceptor(object):
    def __init__(self, stream, level=logging.INFO):
        self.stream = stream
        self.errors = stream.errors
        self.encoding = stream.encoding
        self.level = level
    
    def write(self, value):
        self.stream.write(value)
        if value:
            if value[-1] == '\n':
                value = value[:-1]
            if logger.handlers:
                calling_logger(2).log(self.level, value)
    
    def flush(self):
        self.stream.flush()


sys.stderr = PrintInterceptor(sys.stderr, logging.WARNING)

log_levels = {
    env.DEVELOPMENT: logging.DEBUG,
    env.TEST: logging.DEBUG,
}
logger.setLevel(log_levels.get(env.name, logging.INFO))


if env.name != env.DEVELOPMENT and env.name != env.TEST:
    address = None
    for f in ('/dev/log', '/var/run/syslog',):
        if os.path.exists(f):
            address = f
            break
            
    if address:
        add_syslog_handler(address)


def strip_one_off_handlers():
    global logger
    for child in logging.Logger.manager.loggerDict.values():
        if isinstance(child, logging.Logger):
            for one_off in child.handlers:
                child.removeHandler(one_off)
            child.setLevel(logger.level)

strip_one_off_handlers()


@contextmanager
def timer(message="elapsed time:", level=logging.INFO, caller_level=3):
    start = datetime.now()
    try:
        yield
    finally:
        calling_logger(caller_level).log(level, '%s %s' % (message, datetime.now() - start))


def parametrized(decorator):
    def layer(*args, **kwargs):
        def repl(f):
            return decorator(f, *args, **kwargs)
        return repl
    return layer


@parametrized
def timed(func, level):
    def wrapper(*args, **kwargs):
        with timer('.'.join([func.__module__, func.__name__]), level=level, caller_level=4):
            return func(*args, **kwargs)
    
    return wrapper


def parametrized(decorator):
    def layer(*args, **kwargs):
        def repl(f):
            return decorator(f, *args, **kwargs)
        return repl
    return layer


@parametrized
def timed(func, level):
    def wrapper(*args, **kwargs):
        with timer('.'.join([func.__module__, func.__name__]), level=level):
            return func(*args, **kwargs)
    
    return wrapper


def which(command):
    if sys.version_info.major < 3:
        paths = os.environ['PATH'].split(os.pathsep)
        return any(
            os.access(os.path.join(path, command), os.X_OK) for path in paths
        )
    else:
        return shutil.which(command)


def calling_logger(height=1):
    """ Obtain a logger for the calling module.

    Uses the inspect module to find the name of the calling function and its
    position in the module hierarchy. With the optional height argument, logs
    for caller's caller, and so forth.
    
    see: http://stackoverflow.com/a/900404/48251
    """
    stack = inspect.stack()
    height = min(len(stack) - 1, height)
    caller = stack[height]
    scope = caller[0].f_globals
    path = scope['__name__']
    if path == '__main__':
        path = scope['__package__'] or os.path.basename(sys.argv[0])
    return logging.getLogger(path)
