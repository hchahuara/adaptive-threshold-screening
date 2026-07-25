import time
import random
import idx2numpy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from skimage import draw
from datetime import datetime
from math import log10, sqrt

eps = np.finfo(float).eps

## UTIL FUNCTIONS

def sigmoid(u):
  return 1/(1 + np.exp(-u, dtype=np.float32))

def softmax(u):
  e = np.exp(u-np.max(u), dtype=np.float32)
  return e/(np.broadcast_to(np.expand_dims(np.sum(e, axis=1, dtype=np.float32), axis=-1), u.shape) + 1e-16)

## REGULARIZATIONS

def l1norm(u):
  return np.linalg.norm(u, ord=1)

def l12norm(u):
  return np.linalg.norm(np.linalg.norm(u, ord=2, axis=1), ord=1, axis=0)

## PROXIMAL OPERATORS

def prox_l1norm(v, tau):
  return np.maximum(np.abs(v)-tau, 0)*np.sign(v)

def prox_l12norm(v, tau):
  vnorm = np.expand_dims(np.float32(np.linalg.norm(v, ord=2, axis=1)), axis=-1)
  return np.maximum(vnorm-tau, 0)*v/(vnorm + eps)

## LOSS FUNCTIONS

def lsqrloss(b, u):
  return 0.5*np.linalg.norm(u-b, 2)**2

def binceloss(z, u):

  s = sigmoid(u)
  h = z*np.log(s, dtype=np.float32) + (1-z)*np.log(1-s, dtype=np.float32)

  h[np.isnan(h)] = 0
  h[np.isinf(h)] = 0

  return -np.sum(h)

def celoss(z, u):

  h = z*np.log(softmax(u), dtype=np.float32)

  h[np.isnan(h)] = 0
  h[np.isinf(h)] = 0

  return -np.sum(h)

def hingeloss(z, u):
  return (1/z.shape[0])*np.sum(np.maximum(1-z*u))

def linregloss(b, u):
  return 0.5*np.linalg.norm(u-b, 'fro')**2

## GRADIENT OF LOSS

def grad_lsqrloss(b, u):
  return u-b

def grad_binceloss(z, u):
  return sigmoid(u)-z

def grad_celoss(z, u):
  return softmax(u)-z

def grad_hingeloss(z, u):
  return -(1/z.shape[0])*z*(u*z < 1)

def grad_linregloss(b, u):
  return u-b

## FENCHEL CONJUGATES

def lsqrloss_conj(b, u):
  return 0.5*np.linalg.norm(u+b, 2)**2 - 0.5*np.linalg.norm(b, 2)**2

def binceloss_conj(z, u):

  r = z+u
  h = r*np.log(r, dtype=np.float32) + (1-r)*np.log(1-r, dtype=np.float32)

  h[np.isnan(h)] = 0
  h[np.isinf(h)] = 0

  return np.sum(h)

def celoss_conj(z, u):

  r = z+u
  h = r*np.log(r)

  h[np.isnan(h)] = 0
  h[np.isinf(h)] = 0

  return np.sum(h)

def hingeloss_conj(z, u):
  return -(1/z.shape[0])*np.sum(z.shape[0]*u/z)

def linregloss_conj(b, u):
  return 0.5*np.linalg.norm(u+b, 'fro')**2 - 0.5*np.linalg.norm(b, 'fro')**2
