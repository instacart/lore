# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""Lore doesn't have much of a UI. Text output should be excellent.

"""
RESET = 0

BOLD = 1
FAINT = 2
ITALIC = 3
UNDERLINE = 4
STROBE = 5
BLINK = 6
INVERSE = 7
CONCEAL = 8
STRIKE = 9

BLACK = 30
RED = 31
GREEN = 32
YELLOW = 33
BLUE = 34
MAGENTA = 35
CYAN = 36
WHITE = 37
DEFAULT = 39


def debug(content='DEBUG'):
    return gray(7, content)


def info(content='INFO'):
    return foreground(BLUE, content)


def warning(content='WARNING'):
    """ warning style
    
    :param content:
    :return: string
    """
    return foreground(MAGENTA, content)


def success(content='SUCCESS'):
    """ success style
    
    :param content:
    :return: string
    """
    return foreground(GREEN, content)


def error(content='ERROR'):
    """ error style
    
    :param content:
    :return: string
    """
    return foreground(RED, content)


def critical(content='CRITICAL'):
    return blink(foreground(bright(RED), content))


def foreground(color, content, readline=False):
    """ Color the text of the content
    
    :param color:
    :param content:
    :return:
    """
    return encode(color, readline=readline) + content + encode(DEFAULT, readline=readline)


def background(color, content):
    """ Color the background of the content
    
    :param color:
    :param content:
    :return:
    """
    return encode(color + 10) + content + encode(DEFAULT + 10)


def bright(color):
    """ Brighten colors
    
    :param color:
    :return: brighter version of the color
    """
    return color + 60


def gray(gray, content):
    """ Grayscale
    
    :param gray: [0-15] 0 is almost black, 15 is nearly white
    :param content: content
    :return: string
    """
    return encode('38;5;%i' % (232 + gray)) + content + encode(DEFAULT)


def rgb(red, green, blue, content):
    """ Colors a content using rgb for h
    :param red: [0-5]
    :param green: [0-5]
    :param blue: [0-5]
    :param content: content to markup
    :return: string
    """
    rgb = 16 + 36 * red + 6 * green + blue
    return encode('38;5;' + str(rgb)) + content + encode(DEFAULT)


def bold(content):
    """ Bold content
    
    :param content:
    :return: string
    """
    return style(BOLD, content)


def faint(content):
    """ Faint content
    
    :param content:
    :return: string
    """
    return style(FAINT, content)


def italic(content):
    """ Italic content
    
    :param content:
    :return: string
    """
    return style(ITALIC, content)


def underline(content):
    """ Underline content
    
    :param content:
    :return: string
    """
    return style(UNDERLINE, content)


def strobe(content):
    """ Quickly blinking content
    
    :param content:
    :return: string
    """
    return style(STROBE, content)


def blink(content):
    """ Slowing blinking content
    
    :param content:
    :return: string
    """
    return style(BLINK, content)


def strike(content):
    """ Strike through content
    
    :param content:
    :return: string
    """
    return style(STRIKE, content)


def reset():
    """ Remove any active ansi styles
    
    :return: string resetting marker
    """
    return encode(RESET)


def style(effect, content):
    """ add a particular style to the content
    
    :param effect: style
    :param content: string
    :return: string
    """
    return encode(effect) + content + encode(effect + 20)


def encode(code, readline=False):
    """ Adds escape and control characters for ANSI codes
    
    :param code:
    :return:
    """
    if readline:
        return '\001\033[' + str(code) + 'm\002'

    return '\033[' + str(code) + 'm'
