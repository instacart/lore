# -*- coding: utf-8 -*-
"""
ANSI makes it easy!
*******************

:any:`lore.ansi` makes formatting text output super simple! Lore doesn't have
much of a UI. Text output should be excellent.

.. role:: strike
    :class: strike
"""
from __future__ import absolute_import, print_function, unicode_literals

import platform

RESET = 0  #: Game over!

BOLD = 1  #: For people with a heavy hand.
FAINT = 2  #: For people with a light touch.
ITALIC = 3  #: Are Italian's emphatic, or is Italy slanted? `Etymology <https://www.etymonline.com/word/italic>`_ is fun.
UNDERLINE = 4  #: It's got a line under it, no need for a PhD here.
STROBE = 5  #: For sadists looking to cause seizures. Doesn't work except on masochist's platforms.
BLINK = 6  #: For that patiently waiting cursor effect. Also doesn't work, since sadists ruined it for everyone.
INVERSE = 7  #: Today is backwards day.
CONCEAL = 8  #: Why would you do this‽ Your attempt has been logged, and will be reported to the authorities.
STRIKE = 9  #: Adopt that sense of humility, let other people know you w̶e̶r̶e̶ ̶w̶r̶o̶n̶g learned from experience.

BLACK = 30  #: If you gaze long into an abyss, the abyss will gaze back into you.
RED = 31  #: Hot and loud. Like a fire engine, anger, or you've just bitten your cheek for the third time today.
GREEN = 32  #: The most refreshingly natural color. Growth and softness.
YELLOW = 33  #: Daisies and deadly poison dart frogs. Salted butter and lightning. Like scotch filtered through gold foil.
BLUE = 34  #: Skies, oceans, infinite depths. The color of hope and melancholy.
MAGENTA = 35  #: For latin salsa dresses with matching shoes. Also, the radiant color of T brown dwarf stars as long as sodium and potassium atoms absorb the :any:`GREEN` light in the spectrum.
CYAN = 36  #: Only printers who prefer CMYK over RGB would name this color. It's :any:`BLUE` stripped of soul, injected with 10,000 volts. A true Frankenstein's monster.
WHITE = 37  #: The sum of all colors, to the point there is no color left at all. Floats nicely in the abyss.
DEFAULT = 39  #: You get 3 guesses what this color is, and the first 2 don't count.


if platform.system() == 'Windows':
    ###
    # This is an unfortunate hack to enable ansi output on Windows
    # https://bugs.python.org/issue30075
    import msvcrt
    import ctypes
    import os

    from ctypes import wintypes # pylint: disable=ungrouped-imports

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    ERROR_INVALID_PARAMETER = 0x0057
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

    def _check_bool(result, func, args):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return args

    LPDWORD = ctypes.POINTER(wintypes.DWORD)
    kernel32.GetConsoleMode.errcheck = _check_bool
    kernel32.GetConsoleMode.argtypes = (wintypes.HANDLE, LPDWORD)
    kernel32.SetConsoleMode.errcheck = _check_bool
    kernel32.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)

    def set_conout_mode(new_mode, mask=0xffffffff):
        # don't assume StandardOutput is a console.
        # open CONOUT$ instead
        fdout = os.open('CONOUT$', os.O_RDWR)
        try:
            hout = msvcrt.get_osfhandle(fdout)
            old_mode = wintypes.DWORD()
            kernel32.GetConsoleMode(hout, ctypes.byref(old_mode))
            mode = (new_mode & mask) | (old_mode.value & ~mask)
            kernel32.SetConsoleMode(hout, mode)
            return old_mode.value
        finally:
            os.close(fdout)

    def enable_vt_mode():
        mode = mask = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        try:
            return set_conout_mode(mode, mask)
        except WindowsError as e:  # pylint: disable=undefined-variable
            if e.winerror == ERROR_INVALID_PARAMETER:
                raise NotImplementedError
            raise
    import atexit
    atexit.register(set_conout_mode, enable_vt_mode())


def debug(content='DEBUG'):
    """ debug style

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return gray(7, content)


def info(content='INFO'):
    """ info style

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return foreground(BLUE, content)


def warning(content='WARNING'):
    """ warning style

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return foreground(MAGENTA, content)


def success(content='SUCCESS'):
    """ success style

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return foreground(GREEN, content)


def error(content='ERROR'):
    """ error style

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return foreground(RED, content)


def critical(content='CRITICAL'):
    """ for really big fuck ups, not to be used lightly.

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return blink(foreground(bright(RED), content))


def foreground(color, content, readline=False):
    """ Color the text of the content

    :param color: pick a constant, any constant
    :type color: int
    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return encode(color, readline=readline) + content + encode(DEFAULT, readline=readline)


def background(color, content):
    """ Color the background of the content

    :param color: pick a constant, any constant
    :type color: int
    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return encode(color + 10) + content + encode(DEFAULT + 10)


def bright(color):
    """ Brighten a color

    :param color: pick a constant, any constant
    :type color: int
    :type content: unicode
    :return: brighter version of the color
    :rtype: unicode
    """
    return color + 60


def gray(level, content):
    """ Grayscale

    :param level: [0-15] 0 is almost black, 15 is nearly white
    :type level: int
    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return encode('38;5;%i' % (232 + level)) + content + encode(DEFAULT)


def rgb(red, green, blue, content):
    """ Colors a content using rgb for h
    :param red: [0-5]
    :type red: int
    :param green: [0-5]
    :type green: int
    :param blue: [0-5]
    :type blue: int
    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    color = 16 + 36 * red + 6 * green + blue
    return encode('38;5;' + str(color)) + content + encode(DEFAULT)


def bold(content):
    """ Bold content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(BOLD, content)


def faint(content):
    """ Faint content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(FAINT, content)


def italic(content):
    """ Italic content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(ITALIC, content)


def underline(content):
    """ Underline content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(UNDERLINE, content)


def strobe(content):
    """ Quickly blinking content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(STROBE, content)


def blink(content):
    """ Slowing blinking content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(BLINK, content)


def inverse(content):
    """ Inverted content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(BLINK, content)


def conceal(content):
    """ Why do you persist in this nonsense?

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(CONCEAL, content)


def strike(content):
    """ Strike through content

    :param content: Whatever you want to say...
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return style(STRIKE, content)


def reset():
    """ Remove any active ansi styles

    :return: string resetting marker
    :rtype: unicode
    """
    return encode(RESET)


def style(effect, content):
    """ add a particular style to the content

    :param effect: style
    :type effect: int
    :param content: Whatever you want to say...  string
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    return encode(effect) + content + encode(RESET)


def encode(code, readline=False):
    """ Adds escape and control characters for ANSI codes

    :param code: pick a constant, any constant
    :type code: int
    :param readline: add readline compatibility, which causes bugs in other formats
    :type content: unicode
    :return: ansi string
    :rtype: unicode
    """
    if readline:
        return '\001\033[' + str(code) + 'm\002'

    return '\033[' + str(code) + 'm'
