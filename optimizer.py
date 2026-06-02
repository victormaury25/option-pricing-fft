import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize, brentq
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

today = datetime.today()
target_30 = today + timedelta(days=30)
target_60 = today + timedelta(days=60)
target_90 = today + timedelta(days=90)

ticker = yf.Ticker("^SPX")
expirations = ticker.options
expirations_dt = [datetime.strptime(e, '%Y-%m-%d') for e in expirations]

S = yf.Ticker("^SPX").history(period="1d")["Close"].iloc[-1]
tbill = yf.Ticker("^IRX")
r = tbill.history(period="1d")["Close"].iloc[-1] / 100

# Finding the nearest maturity for each target date
def nearest_expiry(target, expirations_dt, expirations):
    idx = min(range(len(expirations_dt)), key=lambda i: abs((expirations_dt[i] - target).days))
    return expirations[idx], expirations_dt[idx]

exp_30, dt_30 = nearest_expiry(target_30, expirations_dt, expirations)
exp_60, dt_60 = nearest_expiry(target_60, expirations_dt, expirations)
exp_90, dt_90 = nearest_expiry(target_90, expirations_dt, expirations)

T_30d = (dt_30 - today).days / 365
T_60d = (dt_60 - today).days / 365
T_90d = (dt_90 - today).days / 365

print(f"30d → {exp_30} (T={T_30d:.4f})")
print(f"60d → {exp_60} (T={T_60d:.4f})")
print(f"90d → {exp_90} (T={T_90d:.4f})")

def BS_call(S0, K, r, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def get_iv(row, T_i):
    K = row["strike"] / S
    mid = row["mid_price"] / S
    try:
        iv = brentq(lambda sigma: BS_call(1, K, r, T_i, sigma) - mid, 1e-6, 5.0)
        return iv
    except:
        return np.nan


df_30d = ticker.option_chain(exp_30).calls
df_60d = ticker.option_chain(exp_60).calls
df_90d = ticker.option_chain(exp_90).calls
S = yf.Ticker("^SPX").history(period="1d")["Close"].iloc[-1]

df_filtered_30d = df_30d[(df_30d['bid']>0) & (df_30d['strike']/S > 0.9) & (df_30d['strike']/S < 1.10) & (np.log(df_30d['strike']/S) > -0.05) & (np.log(df_30d['strike']/S) < 0.05)].copy()
df_filtered_60d = df_60d[(df_60d['bid']>0) & (df_60d['strike']/S > 0.9) & (df_60d['strike']/S < 1.10) & (np.log(df_60d['strike']/S) > -0.05) & (np.log(df_60d['strike']/S) < 0.05)].copy()
df_filtered_90d = df_90d[(df_90d['bid']>0) & (df_90d['strike']/S > 0.9) & (df_90d['strike']/S < 1.10) & (np.log(df_90d['strike']/S) > -0.05) & (np.log(df_90d['strike']/S) < 0.05)].copy()

df_filtered_30d['mid_price'] = (df_filtered_30d['ask'] + df_filtered_30d['bid'])/2
df_filtered_60d['mid_price'] = (df_filtered_60d['ask'] + df_filtered_60d['bid'])/2
df_filtered_90d['mid_price'] = (df_filtered_90d['ask'] + df_filtered_90d['bid'])/2

df_filtered_30d['iv_market'] = df_filtered_30d.apply(lambda row: get_iv(row, T_30d), axis=1)
df_filtered_60d['iv_market'] = df_filtered_60d.apply(lambda row: get_iv(row, T_60d), axis=1)
df_filtered_90d['iv_market'] = df_filtered_90d.apply(lambda row: get_iv(row, T_90d), axis=1)

df_filtered_30d['log(K/S)'] = np.log(df_filtered_30d['strike']/S)
df_filtered_60d['log(K/S)'] = np.log(df_filtered_60d['strike']/S)
df_filtered_90d['log(K/S)'] = np.log(df_filtered_90d['strike']/S)

T = [T_30d, T_60d, T_90d]
df_filtered_tot = [df_filtered_30d, df_filtered_60d, df_filtered_90d]


N = 8192
eta = 0.05
alpha = 1.5
def C_ITM(r, T, N, S_0, params, eta, alpha, phi):
    lambda_1 = (2*np.pi)/(N*eta)
    b = 0.5 * N * lambda_1
    k = np.array([-b + lambda_1*(u - 1) for u in range(1,N+1)])
    nu = np.array([eta*(j-1) for j in range(1, N+1)])
    weights = (3 + (-1)**np.arange(N)).astype(float)
    weights[0] = 1
    weights[-1] = 1

    psi = ((np.exp(-r*T))*phi(r, T, S_0, params, nu - (alpha + 1)*1j))/(alpha**2 + alpha - nu**2 + 1j*(2*alpha + 1)*nu)
    x = np.exp(1j*b*nu)*psi*(eta/3) * weights
    return k, (np.exp(-alpha * k)/np.pi) * np.real(np.fft.fft(x))

def phi_VG(r, T, S_0, params, u):
    theta, sigma, v = params
    omega = (1/v)*np.log(1 - theta * v - 0.5 * sigma**2 * v)
    return np.exp(1j*u*(np.log(S_0) + (r+omega)*T))*(1 - 1j * theta*v*u + 0.5 * sigma**2 * u**2 * v)**(-T/v)

def phi_BSM(r, T, S_0, params, u):
    sigma, = params
    return np.exp(1j*u*(np.log(S_0) + (r - 0.5*sigma**2)*T) - 0.5*sigma**2*u**2*T)

def make_ivrmse(T_i, df_i, phi):
    def ivrmse(params):
        iv_VG = []
        k, C = C_ITM(r, T_i, N, 1, params, eta, alpha, phi)
        mask = (k > -0.5) & (k < 0.5)
        cs = CubicSpline(k[mask], C[mask])

        for i in df_i['strike']:
            k_target = np.log(i / S)
            price = cs(k_target)
            try:
                iv = brentq(lambda sv: BS_call(1, i/S, r, T_i, sv) - price, 1e-6, 5.0)
            except:
                iv = np.nan
            iv_VG.append(iv)

        diff = np.array(iv_VG) - np.array(df_i['iv_market'])
        return np.sqrt(np.nanmean(diff**2))
    return ivrmse

def phi_heston(r, T, S_0, params, u):
    eta, kappa, lambda_1, rho, nu_0 = params
    d = np.sqrt((rho*lambda_1*u*1j - kappa)**2 + lambda_1**2*(u*1j + u**2))
    g_2 = (kappa - rho*lambda_1*u*1j - d)/(kappa - rho*lambda_1*u*1j + d)
    return np.exp(1j*u*(np.log(S_0) + r*T))*np.exp(eta*kappa/lambda_1**2*((kappa - rho*lambda_1*u*1j - d)*T - 2*np.log((1-g_2*np.exp(-d*T))/(1-g_2))))*np.exp(nu_0**2/lambda_1**2*(kappa - rho*lambda_1*u*1j - d)*(1-np.exp(-d*T))/(1-g_2*np.exp(-d*T)))

def ivrmse_joint(params):
    total = 0
    for T_i, df_i in zip(T, df_filtered_tot):
        total += make_ivrmse(T_i, df_i, phi_heston)(params)**2
    return np.sqrt(total / len(T))

bounds_heston = [(0.001, 0.5), (0.1, 10), (0.01, 2), (-0.99, 0), (0.01, 1)]

def run_minimize(x0):
    from scipy.optimize import minimize
    return minimize(ivrmse_joint, x0, method='L-BFGS-B', bounds=bounds_heston)

def run_minimize_permat(args):
    T_i, df_i, x0 = args
    from scipy.optimize import minimize
    return minimize(make_ivrmse(T_i, df_i, phi_heston), x0, method='L-BFGS-B', bounds=bounds_heston)