#The MIT License (MIT)
#
#Copyright (c) 2015 Andre Queiroz
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
#
# Holt-Winters algorithms to forecasting
# Coded in Python 2 by: Andre Queiroz
# Description: This module contains three exponential smoothing algorithms. They are Holt's linear trend method and Holt-Winters seasonal methods (additive and multiplicative).
# References:
#  Hyndman, R. J.; Athanasopoulos, G. (2013) Forecasting: principles and practice. http://otexts.com/fpp/. Accessed on 07/03/2013.
#  Byrd, R. H.; Lu, P.; Nocedal, J. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.

# https://gist.github.com/andrequeiroz/5888967

from __future__ import division
from sys import exit
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b

def RMSE(params, *args):

  Y = args[0]
  type = args[1]
  rmse = 0

  if type == 'linear':

    alpha, beta = params
    a = [Y[0]]
    b = [Y[1] - Y[0]]
    y = [a[0] + b[0]]

    for i in range(len(Y)):

      a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
      b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
      y.append(a[i + 1] + b[i + 1])

  else:

    alpha, beta, gamma = params
    m = args[2]   
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

    if type == 'additive':

      s = [Y[i] - a[0] for i in range(m)]
      y = [a[0] + b[0] + s[0]]

      for i in range(len(Y)):

        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])

    elif type == 'multiplicative':

      s = [Y[i] / a[0] for i in range(m)]
      y = [(a[0] + b[0]) * s[0]]

      for i in range(len(Y)):

        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
        y.append((a[i + 1] + b[i + 1]) * s[i + 1])

    else:

      exit('Type must be either linear, additive or multiplicative')
    
  rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))

  return rmse

def linear(x, fc, alpha = None, beta = None):

  Y = x[:]

  if (alpha == None or beta == None):

    initial_values = array([0.3, 0.1])
    boundaries = [(0, 1), (0, 1)]
    type = 'linear'

    parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type), bounds = boundaries, approx_grad = True)
    alpha, beta = parameters[0]

  a = [Y[0]]
  b = [Y[1] - Y[0]]
  y = [a[0] + b[0]]
  rmse = 0

  for i in range(len(Y) + fc):

    if i == len(Y):
      Y.append(a[-1] + b[-1])

    a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
    y.append(a[i + 1] + b[i + 1])

  rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

  return Y[-fc:], alpha, beta, rmse

def additive(x, m, fc, alpha = None, beta = None, gamma = None):

  Y = x[:]

  if (alpha == None or beta == None or gamma == None):

    initial_values = array([0.3, 0.1, 0.1])
    boundaries = [(0, 1), (0, 1), (0, 1)]
    type = 'additive'

    parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
    alpha, beta, gamma = parameters[0]

  a = [sum(Y[0:m]) / float(m)]
  b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
  s = [Y[i] - a[0] for i in range(m)]
  y = [a[0] + b[0] + s[0]]
  rmse = 0

  for i in range(len(Y) + fc):

    if i == len(Y):
      Y.append(a[-1] + b[-1] + s[-m])

    a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
    s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
    y.append(a[i + 1] + b[i + 1] + s[i + 1])

  rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

  return Y[-fc:], alpha, beta, gamma, rmse

def multiplicative(x, m, fc, alpha = None, beta = None, gamma = None):

  Y = x[:]

  if (alpha == None or beta == None or gamma == None):

    initial_values = array([0.0, 1.0, 0.0])
    boundaries = [(0, 1), (0, 1), (0, 1)]
    type = 'multiplicative'

    parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
    alpha, beta, gamma = parameters[0]

  a = [sum(Y[0:m]) / float(m)]
  b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
  s = [Y[i] / a[0] for i in range(m)]
  y = [(a[0] + b[0]) * s[0]]
  rmse = 0

  for i in range(len(Y) + fc):

    if i == len(Y):
      Y.append((a[-1] + b[-1]) * s[-m])

    a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
    s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
    y.append((a[i + 1] + b[i + 1]) * s[i + 1])

  rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

  return Y[-fc:], alpha, beta, gamma, rmse