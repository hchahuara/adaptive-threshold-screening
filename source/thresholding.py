import statistics
import numpy as np
import matplotlib.pyplot as plt

from math import tanh
from scipy.stats import skew, moment

def skewint_threshold(phi, n_bins, lmb, skewint_dict):

  skew_lim = skewint_dict['skew_lim']

  alpha_0  = skewint_dict['alpha_0']
  alpha_1  = skewint_dict['alpha_1']
  alpha_2  = skewint_dict['alpha_2']
  alpha_3  = skewint_dict['alpha_3']
  alpha_4  = skewint_dict['alpha_4']

  hist, edge_bins = np.histogram(phi, bins=n_bins)

  hist_skew = sample_skewness(phi)

  data_bins = (edge_bins[0:n_bins] + edge_bins[1:n_bins+1])/2

  bin_min  = 0
  hist_min = hist[bin_min]

  bin_mode  = np.argmax(hist)
  hist_mode = hist[bin_mode]

  bin_max  = len(hist)-1
  hist_max = hist[bin_max]

  delta_bin_left  = bin_mode-bin_min
  delta_bin_right = bin_max-bin_mode

  if delta_bin_left > 2:

    m_left = (hist_mode - hist_min)/(data_bins[bin_mode] - data_bins[bin_min])
    b_left = hist_mode - m_left*data_bins[bin_mode]

    bins_left  = range(1, bin_mode, 1)
    dists_left = np.abs(m_left*data_bins[bins_left] - hist[bins_left] + b_left)

    bin_rosin_left = np.argmax(dists_left)

  elif delta_bin_left == 2:
    bin_rosin_left = bin_min+1

  else:
    bin_rosin_left = bin_min

  if delta_bin_right > 2:

    m_right = (hist_mode-hist_max)/(data_bins[bin_mode]-data_bins[bin_max])
    b_right = hist_mode - m_right*data_bins[bin_mode]

    bins_right  = range(bin_mode+1, n_bins-1, 1)
    dists_right = np.abs(m_right*data_bins[bins_right] - hist[bins_right] + b_right)

    bin_rosin_right = np.argmax(dists_right) + bin_mode + 1

  elif delta_bin_right == 2:
    bin_rosin_right = bin_max-1

  else:
    bin_rosin_right = bin_max

  phi_min  = data_bins[bin_min]
  phi_l    = data_bins[bin_rosin_left]
  phi_mode = data_bins[bin_mode]
  phi_h    = data_bins[bin_rosin_right]
  phi_max  = data_bins[bin_max]

  if phi_max <= 1:
    t = 1

  elif phi_mode <= 1:
    t = alpha_0*max(1,phi_h) + (1-alpha_0)*phi_max

  elif hist_skew > skew_lim:
    t = alpha_1*phi_mode + (1-alpha_1)*phi_h

  elif abs(hist_skew) <= skew_lim:
    t = alpha_2*max(1,phi_l) + (1-alpha_2)*phi_mode

  elif phi_l <= 1:
    t = alpha_3 + (1-alpha_3)*phi_mode

  else:
    t = alpha_4*max(1,phi_min) + (1-alpha_4)*phi_l

  return t

def sample_skewness(data):

  mean   = statistics.mean(data)
  median = statistics.median(data)
  std    = statistics.stdev(data)
  
  return 3*(mean-median)/std
