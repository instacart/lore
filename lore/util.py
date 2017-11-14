# -*- coding: utf-8 -*-
from __future__ import absolute_import

import atexit
import inspect
import logging
import logging.handlers
import os
import re
import sys
import six
import time
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime

from lore import ansi, env

if sys.version_info.major == 3:
    import shutil

if not (sys.version_info.major == 3 and sys.version_info.minor >= 6):
    ModuleNotFoundError = ImportError


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
def timer(message="elapsed time:", level=logging.INFO, caller_level=3, librato=True):
    global _librato
    
    start = datetime.now()
    try:
        yield
    finally:
        time = datetime.now() - start
        if librato and _librato and level >= logging.INFO:
            librato_name = 'timer.' + message.replace(' ', '.').lower()
            if librato_name.endswith(':'):
                librato_name = librato_name[0:-1]
            librato_record(librato_name, time.total_seconds())
        calling_logger(caller_level).log(level, '%s %s' % (message, time))


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

_librato = None

if env.launched():
    # Numpy
    try:
        with timer('numpy init', logging.DEBUG):
            import numpy
            
            numpy.random.seed(1)
            logger.debug('numpy.random.seed(1)')
    except ModuleNotFoundError as e:
        pass
    
    # Rollbar
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
        
            def report_exception(exc_type=None, value=None, tb=None):
                global project
                if exc_type is None:
                    exc_type, value, tb = sys.exc_info()
                stacktrace = ''.join(traceback.format_exception(exc_type, value, tb))
                logger.exception('Exception: %s' % stacktrace)
                if hasattr(sys, 'ps1'):
                    print(stacktrace)
                else:
                    try:
                        rollbar.report_exc_info(extra_data={"app": env.project})
                    except Exception as e:
                        logger.exception('reporting to rollbar: %s' % e)
    
            sys.excepthook = report_exception
    
    except ModuleNotFoundError as e:
        def report_exception(exc_type=None, value=None, tb=None):
            global project
            if exc_type is None:
                exc_type, value, tb = sys.exc_info()
            stacktrace = ''.join(traceback.format_exception(exc_type, value, tb))
            logger.exception('Exception: %s' % stacktrace)
            if hasattr(sys, 'ps1'):
                print(stacktrace)
        
        sys.excepthook = report_exception
    
    # Librato
    try:
        import librato
        from librato.aggregator import Aggregator

        # client side aggregation
        LIBRATO_MIN_AGGREGATION_PERIOD = 5
        LIBRATO_MAX_AGGREGATION_PERIOD = 60

        _librato = None
        if os.getenv('LIBRATO_USER'):
            try:
                _librato = librato.connect(os.getenv('LIBRATO_USER'), os.getenv('LIBRATO_TOKEN'))
                _librato_aggregator = None
                _librato_timer = None
                _librato_start = None
                _librato_lock = threading.RLock()
                logger.info('connected to librato with user: %s' % os.getenv('LIBRATO_USER'))
            except:
                logger.exception('unable to start librato')
                report_exception()
                _librato = None
        else:
            logger.warning('librato variables not found')
            

    except ModuleNotFoundError as e:
        pass
    

    def librato_record(name, value):
        global _librato, _librato_lock, _librato_aggregator, _librato_timer, _librato_start
        
        if _librato is None:
            return
        try:
            name = '.'.join([env.project, env.name, name])
            with _librato_lock:
                _librato_cancel_timer()
                if _librato_aggregator is None:
                    _librato_aggregator = librato.aggregator.Aggregator(_librato, source=env.host)
                    _librato_start = time.time()

                _librato_aggregator.add(name, value)

                if time.time() - _librato_start > LIBRATO_MAX_AGGREGATION_PERIOD:
                    librato_submit()
                else:
                    _librato_timer = threading.Timer(LIBRATO_MIN_AGGREGATION_PERIOD, librato_submit)
                    _librato_timer.start()
        except:
            report_exception()
    
    def librato_submit(background=True):
        global _librato, _librato_lock, _librato_aggregator, _librato_timer, _librato_start

        if _librato is None:
            return

        with _librato_lock:
            _librato_cancel_timer()
            submission_aggregator = _librato_aggregator
            _librato_aggregator = None
            _librato_start = None

        if background:
            threading.Thread(target=submission_aggregator.submit).start()
        else:
            try:
                submission_aggregator.submit()
            except:
                report_exception()
    atexit.register(librato_submit, False)

    def _librato_cancel_timer():
        global _librato_timer
        if _librato_timer:
            _librato_timer.cancel()
            _librato_timer = None
