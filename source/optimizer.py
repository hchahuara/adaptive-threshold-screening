import numpy as np

from datetime import datetime
from source.screening import refine_support

eps = np.finfo(float).eps

def apg_opt(b, A, lmb, u_0, params_dict, functions_dict, screening_dict, alphas_dict):

  max_num_screen   = params_dict['k_screening']
  skip_iter_screen = params_dict['k_skip']
  ss               = params_dict['ss']
  alpha_km1        = params_dict['alpha']
  max_iter         = params_dict['max_iter']
  tol              = params_dict['tol']
  c                = params_dict['c']
  stop_crit        = params_dict['stop_crit']

  df_du = functions_dict['df_du']

  meas  = np.zeros(max_iter, dtype=np.float32)
  F     = np.zeros(max_iter, dtype=np.float32)
  alpha = np.zeros(max_iter, dtype=np.float32)

  if stop_crit == 1:
    Fc = np.zeros(max_iter, dtype=np.float32)

  u_k   = u_0
  u_km1 = np.zeros(u_k.shape, dtype=np.float32)

  idx_set = np.arange(u_k.shape[0])

  vars_k_dict   = {}
  costs_k_dict  = {}
  vars_km1_dict = {}

  fun_costs = {}

  vars_k_dict['y']    = np.zeros(u_k.shape, dtype=np.float32)
  vars_k_dict['df']   = np.matmul(A.T, df_du(b, np.matmul(A, u_k)))
  vars_k_dict['t']    = 1
  vars_km1_dict['y']  = np.zeros(u_k.shape, dtype=np.float32)
  vars_km1_dict['df'] = np.zeros(u_k.shape, dtype=np.float32)

  costs_k_dict['F']  = 0
  costs_k_dict['Fc']  = 0

  k_screen = 0

  for k in range(max_iter):

    if k_screen < max_num_screen:
      if k % skip_iter_screen == 0:

        supp_k = refine_support(b, A, u_k, lmb, functions_dict, screening_dict, alphas_dict)

        if np.sum(supp_k) < u_k.shape[0]:

          idx_set = idx_set[supp_k]

          A = A[:,supp_k]

          u_k   = u_k[supp_k,:]
          u_km1 = u_km1[supp_k,:]

          y_k    = vars_k_dict['y']
          df_k   = vars_k_dict['df']
          y_km1  = vars_km1_dict['y']
          df_km1 = vars_km1_dict['df']

          vars_k_dict['y']    = y_k[supp_k,:]
          vars_k_dict['df']   = df_k[supp_k,:]
          vars_km1_dict['y']  = y_km1[supp_k,:]
          vars_km1_dict['df'] = df_km1[supp_k,:]

        k_screen = k_screen+1

    alpha_k = update_step_size(A, u_k, u_km1, alpha_km1, k, vars_k_dict, vars_km1_dict, params_dict)

    u_k, u_km1, vars_k_dict, vars_km1_dict = apg_opt_step(b, A, u_k, u_km1, lmb, alpha_k, vars_k_dict, vars_km1_dict, functions_dict)

    meas_k, costs_k_dict = compute_criterion_value(b, A, u_k, lmb, functions_dict, params_dict, costs_k_dict)

    meas[k] = meas_k
    F[k]    = costs_k_dict['F']

    if stop_crit == 1:
      Fc[k]   = costs_k_dict['Fc']

    if meas_k < tol:
      break

    alpha_km1 = alpha_k

    alpha[k] = alpha_k

  u_opt = np.zeros(u_0.shape, dtype=np.float32)

  u_opt[idx_set,:] = u_k

  fun_costs['F'] = F

  if stop_crit == 1:
    fun_costs['Fc'] = Fc

  fun_costs['alpha'] = alpha

  return u_opt, idx_set, meas, fun_costs

def update_step_size(A, u_k, u_km1, alpha, k, vars_k_dict, vars_km1_dict, params_dict):

  y_k  = vars_k_dict['y']
  df_k = vars_k_dict['df']

  y_km1  = vars_km1_dict['y']
  df_km1 = vars_km1_dict['df']

  c  = params_dict['c']
  ss = params_dict['ss']

  alpha_k = alpha

  if ss > 0:

    if k > 0:
      s_km1 = u_k - u_km1
      z_km1 = df_k - df_km1

      if ss < 3:

        if ss == 2:
          df_eff_k = df_k*(y_k > 0)

        else:
          df_eff_k = df_k

        alpha_k = c*(np.linalg.norm(df_eff_k, 'fro')**2)/(np.linalg.norm(np.matmul(A, df_eff_k), 'fro')**2 + eps)

      elif ss == 3:
        if k % 2 == 1:
          alpha_k   = (np.linalg.norm(df_k, 'fro')**2)/(np.linalg.norm(np.matmul(A, df_k), 'fro')**2 + eps)

      elif ss == 4:
        alpha_k = (np.linalg.norm(s_km1, 'fro')**2)/(np.matmul(s_km1.T, z_km1) + eps)

      elif ss == 5:
        alpha_k = np.matmul(s_km1.T, z_km1)/np.float32(np.linalg.norm(z_km1, 'fro')**2 + eps)

      else:
        alpha_k = np.float32(np.linalg.norm(s_km1, 'fro')/(np.linalg.norm(z_km1, 'fro') + eps))

    else:
      alpha_k = c*np.float32(np.linalg.norm(df_k, 'fro')**2)/(np.linalg.norm(np.matmul(A, df_k), 'fro')**2 + eps)

  else:
    alpha_k = alpha

  return np.float32(alpha_k)


def apg_opt_step(b, A, u_k, u_km1, lmb, alpha_k, vars_k_dict, vars_km1_dict, functions_dict):

  y_k  = vars_k_dict['y']
  t_k  = vars_k_dict['t']

  df_du  = functions_dict['df_du']
  prox_g = functions_dict['prox_g']

  df_k = vars_k_dict['df']

  u_km1 = u_k
  u_k   = prox_g(y_k-alpha_k*df_k, lmb*alpha_k)

  t_kp1   = 0.5 + 0.5*(1+4*t_k**2)**0.5
  gamma_k = (t_k-1)/t_kp1

  y_km1 = y_k
  y_k   = u_k + gamma_k*(u_k-u_km1)

  df_km1 = df_k
  df_k   = np.matmul(A.T, df_du(b, np.matmul(A, u_k)))

  vars_k_dict['y']  = y_k
  vars_k_dict['df'] = df_k
  vars_k_dict['t']  = t_kp1

  vars_km1_dict['y']  = y_km1
  vars_km1_dict['df'] = df_km1

  return u_k, u_km1, vars_k_dict, vars_km1_dict

def compute_criterion_value(b, A, u_k, lmb, functions_dict, params_dict, costs_k_dict):

  f       = functions_dict['f']
  g       = functions_dict['g']
  fc      = functions_dict['fc']
  df_du   = functions_dict['df_du']
  lmb_max = functions_dict['lmb_max']

  stop_crit = params_dict['stop_crit']

  F_km1 = costs_k_dict['F']

  F_k = f(b, np.matmul(A, u_k)) + lmb*g(u_k)

  if stop_crit == 1:
  
    v_k = -df_du(b, np.matmul(A, u_k))/max(lmb, lmb_max)
    Fc_k = -fc(b, -lmb*v_k)

    meas_k = F_k - Fc_k

  else:
    meas_k = abs((F_k - F_km1)/(F_km1 + eps))

  costs_k_dict['F']  = F_k

  if stop_crit == 1:
    costs_k_dict['Fc'] = Fc_k

  return meas_k, costs_k_dict
