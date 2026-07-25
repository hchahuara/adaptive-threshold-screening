import numpy as np

from scipy import signal
from source.thresholding import skewint_threshold

def safeball_gapsafe_screening_metric(b, A, u, lmb, functions_dict):

  f       = functions_dict['f']
  g       = functions_dict['g']
  df_du   = functions_dict['df_du']
  fc      = functions_dict['fc']
  L       = functions_dict['L']
  lmb_max = functions_dict['lmb_max']

  w   = np.matmul(A, u, dtype=np.float32)
  res = -df_du(b, w)
  df  = np.matmul(A.T, res, dtype=np.float32)

  lmb_max = np.float32(np.linalg.norm(np.sum(df**2, axis=1)**0.5, ord=np.inf))
  theta   = res/np.maximum(lmb, lmb_max)

  eta = f(b, w) + lmb*g(u)
  nu  = -fc(b, -lmb*theta)
  gap = eta-nu

  radius = ((2*gap/L)**0.5)/lmb
  center = theta

  phi = np.sum(np.matmul(center.T, A, dtype=np.float32)**2, axis=0)**0.5 + radius*np.float32(np.linalg.norm(A, ord=2, axis=0))

  return np.reshape(phi, -1)

def strong_screening_metric(b, A, u, lmb, functions_dict):

  f     = functions_dict['f']
  df_du = functions_dict['df_du']

  w   = np.matmul(A, u, dtype=np.float32)
  res = -df_du(b, w)
  df  = np.matmul(A.T, res, dtype=np.float32)

  lmb_max = np.float32(np.linalg.norm(np.sum(df**2, axis=1)**0.5, np.inf))
  theta   = res/lmb

  phi = np.sum(np.matmul(theta.T, A, dtype=np.float32)**2, axis=0)**0.5 + (lmb_max/lmb-1)

  return np.reshape(phi, -1)

def refine_support(b, A, u, lmb, functions_dict, screening_dict, thresh_dict):

  f       = functions_dict['f']
  g       = functions_dict['g']
  df_du   = functions_dict['df_du']
  fc      = functions_dict['fc']
  L       = functions_dict['L']

  mode      = screening_dict['mode']
  n_bins    = screening_dict['n_bins']
  threshold = screening_dict['thresholding']

  if mode == 0:
    phi = safeball_gapsafe_screening_metric(b, A, u, lmb, functions_dict)
  else:
    phi = strong_screening_metric(b, A, u, lmb, functions_dict)

  if threshold == 0:
    t = 1
  elif threshold == 1:
    t = skewint_threshold(phi, n_bins, lmb, thresh_dict)
  else:
    t = 1

  return phi >= t
