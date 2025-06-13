# -*- coding: utf-8 -*-
"""SEFE_final_ass.ipynb
# Mudules
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import  minimize
from scipy.stats import lognorm
from scipy.special import beta
from scipy.stats import skew
from scipy.interpolate import interp1d
import math
import yfinance as yf
from arch import arch_model
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid

"""# Function"""

def plot_func (x, y, label, title, xlabel, ylabel, plot_name, grid = True):
  plt.figure(figsize=(8, 5))
  plt.plot(x, y, label=label)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(grid)
  plt.legend()
  if plot_name != None: plt.savefig(plot_name)
  plt.show()
  return None

"""# Data"""

"""### 2025/04/30 UTC+8
1. Underlying asset: ^TWII
2. Future: TX ($F=S_0e^{rT}$)
3. Option: TXO
4. Risk-free rate: TAIBOR 1M

"""

df = pd.read_csv('20250430TXO.csv')
S = 20235     # 現貨價格
r = 0.016     # 無風險利率
T = 21 / 252    # 到期時間（年）
K = df['K'].values
C_mkt = df['Cmkt'].values
P_mkt = df['Pmkt'].values
#F = S * np.exp(r * T)  # 計算期貨價格
F = 20149

"""# 1. Estimate RND via option prices

## 1.1 RND - MLN
"""
def MLN_func (K, C_mkt, P_mkt, S, r, T, F, n = 30):

  def mln_density(x, w, mu1, sigma1, mu2, sigma2):
    pdf1 = lognorm.pdf(x, s=sigma1, scale=np.exp(mu1))
    pdf2 = lognorm.pdf(x, s=sigma2, scale=np.exp(mu2))
    return w * pdf1 + (1 - w) * pdf2
  def call_price(K, w, mu1, sigma1, mu2, sigma2):
    integrand = lambda x: np.maximum(x - K, 0) * mln_density(x, w, mu1, sigma1, mu2, sigma2)
    result, _ = quad(integrand, K, 5 * F)
    return np.exp(-r * T) * result
  def put_price(K, w, mu1, sigma1, mu2, sigma2):
    integrand = lambda x: np.maximum(K - x, 0) * mln_density(x, w, mu1, sigma1, mu2, sigma2)
    result, _ = quad(integrand, 0, K)
    return np.exp(-r * T) * result
  def loss_fn(params):
    w, mu1, sigma1, mu2, sigma2 = params
    if not (0 < w < 1 and sigma1 > 0 and sigma2 > 0):
        return np.inf
    call_model = np.array([call_price(k, w, mu1, sigma1, mu2, sigma2) for k in K])
    put_model  = np.array([put_price(k, w, mu1, sigma1, mu2, sigma2) for k in K])
    loss = np.mean((call_model - C_mkt)**2 + (put_model - P_mkt)**2)
    return loss

  if F == None: F = S * np.exp(r * T)

  initial_guess = [0.5, np.log(F), 0.2, np.log(F), 0.4]  # 初始值
  bounds = [(0, 1), (-np.inf, np.inf), (1e-3, None), (-np.inf, np.inf), (1e-3, None)]

  result = minimize(loss_fn, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': n})
  w_opt, mu1_opt, sigma1_opt, mu2_opt, sigma2_opt = result.x
  x_MLN = np.linspace(0.8 * F, 1.2 * F, 500)
  fq_MLN = mln_density(x_MLN, w_opt, mu1_opt, sigma1_opt, mu2_opt, sigma2_opt)
  return x_MLN, fq_MLN

x_MLN, fq_MLN = MLN_func(K, C_mkt, P_mkt, S, r, T, F, n = 30)

plot_func (x_MLN,
      fq_MLN,
      label = "RND (MLN)",
      title = "Estimated RND using MLN, 2025/04/30",
      xlabel = "Index Level, $x$",
      ylabel = "Density, $f_q(x)$",
      plot_name = "RND_MLN_20250430",
      grid = True)

"""## 1.2 RND - GB2"""

def GB2_func (S, r, T, K, C_mkt):
    # Step 1: 定義 GB2 pdf
    def gb2_pdf(x, a, b, p, q):
        return (a * x**(a*p - 1)) / (b**(a*p) * beta(p, q) * (1 + (x / b)**a)**(p + q))
    # Step 2: 用 GB2 理論買權定價公式（風險中性期望）
    def gb2_call_price(K, a, b, p, q, r, T):
        x = np.linspace(K.min() * 0.8, K.max() * 1.2, 1000)
        dx = x[1] - x[0]
        f_q = gb2_pdf(x, a, b, p, q)
        call_payoff = np.maximum(x[:, None] - K[None, :], 0)
        price = np.exp(-r * T) * np.sum(f_q[:, None] * call_payoff, axis=0) * dx
        return price
    # Step 3: 最小化誤差
    def loss(params):
        a, b, p, q = params
        if any(x <= 0 for x in [a, b, p, q]):
            return 1e6  # 避免無效參數
        C_model = gb2_call_price(K, a, b, p, q, r, T)
        return np.sum((C_model - C_mkt)**2)
    def drop_trailing_zeros(x, fq):
        """
        去除 fq 尾部連續為 0 的點，並同步刪除 x 對應點
        """
        # 反向尋找第一個非零點索引
        nonzero_indices = np.where(fq != 0)[0]
        if len(nonzero_indices) == 0:
            # 全為零，直接回傳空陣列
            return np.array([]), np.array([])
        last_nonzero_idx = nonzero_indices[-1]
        # 截取至最後非零點
        return x[:last_nonzero_idx+1], fq[:last_nonzero_idx+1]
        
    init_params = [2, S, 2, 2] # 初始值 [a, b, p, q]
    result = minimize(loss, init_params, method='Nelder-Mead') # 執行最佳化
    a_hat, b_hat, p_hat, q_hat = result.x
    x = np.linspace(K.min() * 0.9, K.max() * 1.1, 1000)
    fq = gb2_pdf(x, a_hat, b_hat, p_hat, q_hat)
    x_GB2, fq_GB2 = drop_trailing_zeros(x, fq)
    return x_GB2, fq_GB2

x_GB2, fq_GB2 = GB2_func (S, r, T, K, C_mkt)

plot_func (x_GB2,
      fq_GB2,
      label = "RND (GB2)",
      title = "Estimated RND using GB2, 2025/04/30",
      xlabel = "Index Level, $x$",
      ylabel = "Density, $f_q(x)$",
      plot_name = "RND_GB2_20250430",
      grid = True)

"""# 2
#### Utility function
"""

def analyze_real_world_density(x, f_q_input, gammas=[-2, 0, 4], plot_name="Q and P measure, MLN"):
    """
    目的: 將 RND f_q(x) 轉換為 real-world density f_p(x) 並進行圖形與統計分析。

    參數:
        x (np.array): 價格範圍
        f_q_input (np.array): 對應於 x 的風險中立密度
        gammas (list): CRRA risk aversion 值
        plot_title (str): 圖表標題
    回傳值: mean, skewness, f_p (dict)
    """
    def compute_fp(f_q, x, gamma):
        weight = x ** (-gamma)
        f_p_unnorm = f_q * weight
        f_p = f_p_unnorm / np.trapz(f_p_unnorm, x)
        return f_p

    results = {}

    plt.figure(figsize=(12, 6))
    plt.plot(x, f_q_input, label='RND (f_q)', linewidth=2, color='black')

    for gamma in gammas:
        f_p = compute_fp(f_q_input, x, gamma)
        mean_p = np.trapz(x * f_p, x)
        skew_p = skew(np.random.choice(x, p=f_p / np.sum(f_p), size=10000))
        results[gamma] = {
            'mean': mean_p,
            'skewness': skew_p,
            'fp': f_p
        }
        plt.plot(x, f_p, label=f'γ = {gamma}, μ ≈ {mean_p:.1f}, skew ≈ {skew_p:.2f}')

    plt.title("RND to Real-World Densities")
    plt.xlabel("index level")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()

    # 計算 VaR (γ = 4)
    if 4 in gammas:
        f_p_4 = results[4]['fp']
        cdf_p_4 = np.cumsum(f_p_4)
        cdf_p_4 /= cdf_p_4[-1]  # normalize to 1
        inv_cdf = interp1d(cdf_p_4, x)
        VaR_95 = inv_cdf(0.05)
        print(f"\nValue at Risk (VaR) at 95% confidence when γ = 4: {VaR_95:.2f}")
        results[4]['VaR_95'] = VaR_95
    else:
        print("\nNo VaR calculated (γ=4 not in list).")

    return results

P_measure_MLN = analyze_real_world_density(x_MLN, fq_MLN, gammas=[-2, 0, 4], plot_name = "Q and P measure, MLN")
P_measure_GB2 = analyze_real_world_density(x_GB2, fq_GB2, gammas=[-2, 0, 4], plot_name="Q and P measure, RND")

"""# 3"""

def simulate_arch_paths (start = "2015-04-30", end = "2025-04-30"):
  # 下載台灣加權指數（或替換為 TXF 等期貨收盤價）
  df = pd.read_csv("TWII_20150101_20250601.csv", index_col=0)
  data = df.loc[start:end]
  price = data['Close'].dropna().astype(float)
  returns = 100 * np.log(price / price.shift(1)).dropna()

  # GARCH(1,1)
  #model = arch_model(returns, vol='Garch', p=1, q=1)
  model = arch_model(returns, vol='GARCH', p=1, o=1, q=1)
  res = model.fit(disp='off') # 在最佳化過程中，不顯示逐步訊息
  print(res.summary())

  # 模擬參數
  S0 = price.iloc[-1].item()
  T_days = 21
  N_sim = 1000

  # 模擬 N_sim 條路徑
  simulated_returns = np.zeros((N_sim, T_days))
  for i in range(N_sim):
    sim_path = model.simulate(res.params, nobs=T_days)
    sim_ret = sim_path['data'] / 100  # 百分比轉換為小數報酬
    simulated_returns[i, :] = sim_ret

  # 累積報酬 → 最終價格 S_T
  #cum_returns = np.exp(np.sum(simulated_returns, axis=1))
  cum_returns = np.prod(1 + simulated_returns, axis=1)
  S_T = S0 * cum_returns

  return S_T

def compare_density_plot(x, fq, S_T, Type = "log"):
    fq = fq / np.trapz(fq, x) # 標準化 RND
    kde = gaussian_kde(S_T, bw_method='scott') # KDE 擬合實際世界密度 f_p(x)
    fp = kde(x)
    fp /= np.trapz(fp, x)
    log_diff = np.log(np.maximum(fq, 1e-12) / np.maximum(fp, 1e-12))  # 加 ε 防止除以 0，計算 log-difference
    # 繪圖
    plt.figure(figsize=(12, 6))
    if Type == "log":
      plt.plot(x, log_diff, label="log(f_q / f_p)", linestyle='--', color='green')
    elif Type == "normal":
      plt.plot(x, fq, label="RND (MLN)", color='black')
      plt.plot(x, fp, label="Real-World Density (GARCH)", color='blue')
      plt.axvline(np.percentile(S_T, 5), color='red', linestyle='--', label="5% VaR")
    else:
      print("Error Type")
      return None
    plt.title("Comparison of Risk-Neutral and Real-World Densities")
    plt.xlabel("Terminal Price")
    plt.ylabel("Density / log ratio")
    plt.legend()
    plt.grid(True)
    plt.show()
    return {'fq': fq, 'fp': fp, 'log_diff': log_diff}

S_T = simulate_arch_paths (start = "2015-04-30", end = "2025-04-30")

result_MLN_normal = compare_density_plot(x = x_MLN,
                                         fq = fq_MLN,
                                         S_T = S_T,
                                         Type = "normal")
result_MLN_log = compare_density_plot(x = x_MLN,
                                      fq = fq_MLN,
                                      S_T = S_T,
                                      Type = "log")
result_GB2_normal = compare_density_plot(x = x_GB2,
                                         fq = fq_GB2,
                                         S_T = S_T,
                                         Type = "normal")
result_GB2_log = compare_density_plot(x = x_GB2,
                                      fq = fq_GB2,
                                      S_T = S_T,
                                      Type = "log")

"""# 4.
Compare the empirical ratios before and after an event or in high and low volatility dates.
1. Event: crash in 2025-04-07

#### 4.1 股價資料
"""

df = pd.read_csv('20250402TXO.csv')
S = 21298.221
r = 0.0146944
T = 14 / 252
F = 21304
K = df['K'].values
C_mkt = df['C_mkt'].values
P_mkt = df['P_mkt'].values

x_MLN_1, fq_MLN_1 = MLN_func (K, C_mkt, P_mkt, S, r, T, F, n = 30)
x_GB2_1, fq_GB2_1 = GB2_func (S, r, T, K, C_mkt)

plot_func (x_MLN_1,
      fq_MLN_1,
      label = "RND (MLN)",
      title = "Estimated RND using MLN, 2025/04/02",
      xlabel = "Index Level, $x$",
      ylabel = "Density, $f_q(x)$",
      plot_name = "RND_MLN_20250402",
      grid = True)

plot_func (x_GB2_1,
      fq_GB2_1,
      label = "RND (GB2)",
      title = "Estimated RND using GB2, 2025/04/02",
      xlabel = "Index Level, $x$",
      ylabel = "Density, $f_q(x)$",
      plot_name = "RND_GB2_20250402",
      grid = True)

df = pd.read_csv('20250407TXO.csv')
S = 19232.349
r = 0.01381
T = 9 / 252
F = 19167
K = df['K'].values
C_mkt = df['Cmkt'].values
P_mkt = df['Pmkt'].values

x_MLN_2, fq_MLN_2 = MLN_func (K, C_mkt, P_mkt, S, r, T, F, n = 30)
x_GB2_2, fq_GB2_2 = GB2_func (S, r, T, K, C_mkt)

plot_func (x_MLN_2,
      fq_MLN_2,
      label = "RND (MLN)",
      title = "Estimated RND using MLN, 2025/04/07",
      xlabel = "Index Level, $x$",
      ylabel = "Density, $f_q(x)$",
      plot_name = "RND_MLN_20250407",
      grid = True)

plot_func (x_GB2_2,
      fq_GB2_2,
      label = "RND (GB2)",
      title = "Estimated RND using GB2, 2025/04/07",
      xlabel = "Index Level, $x$",
      ylabel = "Density, $f_q(x)$",
      plot_name = "RND_GB2_20250407",
      grid = True)

"""### 4.2: 以utility function 轉換"""

P_measure_MLN_1 = analyze_real_world_density(x_MLN_1,
                      fq_MLN_1,
                      gammas=[-2, 0, 4],
                      plot_name="RND_to_RealWorld_Densities_MLN_2025_04_02")
P_measure_GB2_1 = analyze_real_world_density(x_GB2_1,
                      fq_GB2_1,
                      gammas=[-2, 0, 4],
                      plot_name="RND_to_RealWorld_Densities_GB2_2025_04_02")

P_measure_MLN_2 = analyze_real_world_density(x_MLN_2,
                      fq_MLN_2,
                      gammas=[-2, 0, 4],
                      plot_name="RND_to_RealWorld_Densities_MLN_2025_04_07")
P_measure_GB2_2 = analyze_real_world_density(x_GB2_2,
                      fq_GB2_2,
                      gammas=[-2, 0, 4],
                      plot_name="RND_to_RealWorld_Densities_GB2_2025_04_07")

"""### 4.3: 與歷史價格分配比較"""

S_T = simulate_arch_paths (start = "2015-04-02", end = "2025-04-02")

result_MLN_normal = compare_density_plot(x = x_MLN_1,
                                         fq = fq_MLN_1,
                                         S_T = S_T,
                                         Type = "normal")
result_MLN_log = compare_density_plot(x = x_MLN_1,
                                      fq = fq_MLN_1,
                                      S_T = S_T,
                                      Type = "log")
result_GB2_normal = compare_density_plot(x = x_GB2_1,
                                         fq = fq_GB2_1,
                                         S_T = S_T,
                                         Type = "normal")
result_GB2_log = compare_density_plot(x = x_GB2_1,
                                      fq = fq_GB2_1,
                                      S_T = S_T,
                                      Type = "log")

S_T = simulate_arch_paths (start = "2015-04-07", end = "2025-04-07")

result_MLN_normal = compare_density_plot(x = x_MLN_2,
                                         fq = fq_MLN_2,
                                         S_T = S_T,
                                         Type = "normal")
result_MLN_log = compare_density_plot(x = x_MLN_2,
                                      fq = fq_MLN_2,
                                      S_T = S_T,
                                      Type = "log")
result_GB2_normal = compare_density_plot(x = x_GB2_2,
                                         fq = fq_GB2_2,
                                         S_T = S_T,
                                         Type = "normal")
result_GB2_log = compare_density_plot(x = x_GB2_2,
                                      fq = fq_GB2_2,
                                      S_T = S_T,
                                      Type = "log")

"""# -------------------------------------"""

