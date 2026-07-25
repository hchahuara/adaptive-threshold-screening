import time
import math
import numpy as np
import scipy.io
import random
import idx2numpy
import matplotlib.pyplot as plt
import source.model as model

from skimage import draw
from datetime import datetime
from math import log10, sqrt

def imnoise_gaussian(x, mean, var):

  row, col = x.shape
  sigma = var**0.5
  noise = np.random.normal(mean, sigma, (row,col))

  return x + noise

def generate_awgn(M, snr_dB):

  noise = np.random.randn(M)
  noise /= np.std(noise)
  noise_power = np.sum(noise**2)/len(noise)
  desired_noise_power = noise_power/(10**(snr_dB/10))
    
  return np.sqrt(desired_noise_power)*noise

def compute_psnr(original, compressed, max_value=None):

  mse = np.mean((original-compressed)**2)

  if mse == 0:
    return float('inf')

  if max_value is None:
    max_value = np.max(original)-np.min(original)

  return 20*log10(max_value/math.sqrt(mse))

def multinomial_predict(X, Omega):

  y_prob = model.softmax(np.matmul(X, Omega, dtype=np.float32))
  y_sel = np.zeros(y_prob.shape)

  for k in range(y_prob.shape[0]):
    n = np.where(y_prob[k,:] == np.max(y_prob[k,:]))
    y_sel[k,n] = 1

  return np.sum(y_sel*np.broadcast_to(np.arange(1,Omega.shape[1]+1), y_sel.shape), axis=1) - 1

def binary_predict(x, omega):

  y_prob = model.sigmoid(np.matmul(x, omega, dtype=np.float32))

  y_sel = np.zeros([y_prob.shape[0],1])
  y_sel[y_prob > 0.5] = 1

  return y_sel

def one_hot_enc(y, n_classes):

  y_enc = np.zeros([y.shape[0], n_classes])

  for k in range(y.shape[0]):
    y_enc[k,y[k]] = 1

  return y_enc

def dct_matrix(N):

  n = np.arange(N)
  dct_mtx = np.zeros((N,N))

  for k in range(N):
    dct_mtx[k,:] = np.sqrt(2/N)*np.cos((np.pi/N)*(n+0.5)*k)

  dct_mtx[0,:] /= np.sqrt(2)

  return dct_mtx

def idct_matrix(N):

  n = np.arange(N)
  idct_mtx = np.zeros((N,N))

  for k in range(N):
    idct_mtx[:,k] = np.sqrt(2/N)*np.cos((np.pi/N)*(n+0.5)*k)

  idct_mtx[:,0] /= np.sqrt(2)

  return idct_mtx
