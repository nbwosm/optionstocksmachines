# Risk-constrained optimization

# Built using R 4.0.3, and Python 3.8.3

# [R]
# Only run this if working in Rmarkdown
# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidyquant)
  library(reticulate)
})

# [Python]
# Load libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/user_name/Anaconda3/Library/plugins/platforms'
# This is mainly if working in windows. Probably different for other machines.

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,6)

# Directory to save images
# Most of the graphs are now imported as png. We've explained why in some cases that was necessary due to the way reticulate plays with Python. But we've also found that if we don't use pngs, the images don't get imported into the emails that go to subscribers. 

DIR = "your/image/directory" 

def save_fig_blog(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(DIR, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dip=resolution)

# Import OSM modules and functions
# https://github.com/nbwosm/optionstocksmachines/blob/main/Port_sim_functions.py
# https://github.com/nbwosm/optionstocksmachines/blob/main/Port_risk_functions.py
from Port_sim_functions import Port_sim as ps
from Portfolio_risk_functions import rsq_func_prem, factor_beta_risk_premium_calc, betas_plot, factor_port_var, port_var_plot

# Load factors
# Prior post (https://www.optionstocksmachines.com/post/2021-01-12-port-22/more-factors-more-variance/) shows how they were downloaded and cleaned.
tf_change = total_factors.copy()
tf_change.loc[:,['PMI', 'ICSA', 'PERMIT', 'BOGMBASE']] = tf_change.loc[:,['PMI', 'ICSA', 'PERMIT', 'BOGMBASE']].pct_change()
tf_change.loc[:,['USSLIND','T10Y2Y', 'CORP']] = tf_change.loc[:,['USSLIND','T10Y2Y', 'CORP']]*.01

# Create time period
fact_gen = tf_change.loc['1990':'2010',[x for x in tf_change.columns.to_list() if x not in ['mkt-rfr', 'rfr']]]

# Generate moments
mu_gen, sig_gen = fact_gen.mean(), fact_gen.std()

# Simulate factors
np.random.seed(123)
fake_factors = pd.DataFrame(np.zeros((60,10)))
for i in range(10):
    fake_factors.iloc[:,i] = np.random.normal(mu_gen[i], sig_gen[i], 60)
    
fake_factors.columns = [f'fake_{i+1}' for i in range(10)]

# Plot fake factors
(fake_factors*100).plot(cmap = 'Blues')
plt.title("Fake factor returns")
plt.xlabel("Month")
plt.ylabel("Returns (%)")
plt.legend(loc = 'upper center', ncol = 5)
plt.show()

# Generate asset returns
np.random.seed(42)
assets = pd.DataFrame(np.zeros((60,30)))
rand_beta = np.array([np.random.normal(0.05,0.3,10) for i in range(30)])
spec_var = np.round(np.random.uniform(0.5,0.9,30),2) #np.repeat(0.9, 30)

for i in range(len(assets.columns)):
    
    assets.iloc[:,i] = spec_var[i]*(np.random.normal(0,0.01/12) + rand_beta[i] @ fake_factors.T + np.random.normal(0.0,0.02,60)) + \
                        (1-spec_var[i])*np.random.normal(0.0, 0.05, 60)
    
assets.columns = [f'asset_{i+1}' for i in range(len(assets.columns))]

# Calculate betas
betas, _, error = factor_beta_risk_premium_calc(fake_factors, assets)

# Create portfolio simulations
np.random.seed(42)
port, wts, sharpe  = ps.calc_sim_lv(df = assets, sims = 10000, cols=30)

# Find relevant portfolios and graph simulation
max_sharp = port[np.argmax(sharpe)]
min_vol = port[np.argmin(port[:,1])]
max_ret_eff = port[pd.DataFrame(np.c_[port,sharpe], columns = ['ret', 'risk', 'sharpe']).sort_values(['ret', 'sharpe'], ascending=False).index[0]]

fig = plt.figure()
ax = fig.add_subplot(1,1, 1)
sim = ax.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
ax.scatter(max_sharp[1]*np.sqrt(12)*100, max_sharp[0]*1200,marker=(4,1,0),color='r',s=500)
ax.scatter(min_vol[1]*np.sqrt(12)*100,min_vol[0]*1200,marker=(4,1,0),color='purple',s=500)
ax.scatter(max_ret_eff[1]*np.sqrt(12)*100,max_ret_eff[0]*1200,marker=(4,1,0),color='blue',s=500)
ax.set_title('Simulated portfolios', fontsize=20)
ax.set_xlabel('Risk (%)')
ax.set_ylabel('Return (%)')

cbaxes = fig.add_axes([0.15, 0.65, 0.01, 0.2])
clb = fig.colorbar(sim, cax = cbaxes)
clb.ax.set_title(label='Sharpe', fontsize=10)
# save_fig_blog('sim_ports_23', tight_layout=False)
plt.show()


# Create efficient factor frontier function
# We found this document helpful in creating the function. If you're curious why we built this using scipy instead of CVXPY or Pyportfolioopt, send us an email and we'll explain.
# https://pyportfolioopt.readthedocs.io/en/latest/index.html

def eff_ret(df_returns, betas=None, factors=None, error=None, gamma=1, optim_out = 'mvo', short = False, l_reg = 0, min_weight = 0):
    
    n = len(df_returns.columns)
    
    def get_data(weights):
        weights = np.array(weights)
        returns = np.sum(df_returns.mean() * weights)
        risk = np.sqrt(np.dot(weights.T, np.dot(df_returns.cov(), weights)))
        sharpe = returns/risk
        return np.array([returns,risk,sharpe])

    def check_sum(weights):
        return np.sum(np.abs(weights)) - 1

    def factor_var_func(weights):
    
        B = np.array(betas)
        F = np.array(factors.cov())
        S = np.diag(np.array(error.var()))

        factor_var = weights.dot(B.dot(F).dot(B.T)).dot(weights.T)
        specific_var = weights.dot(S).dot(weights.T)
         
        return factor_var, specific_var, factor_var/(factor_var + specific_var)
             
    def mvo(weights):
        return -(get_data(weights)[0] - gamma*(get_data(weights)[1]**2 + l_reg*weights.T.dot(weights)))
    
    def factor(weights):
        return -(get_data(weights)[0] - gamma*(factor_var_func(weights)[0] + factor_var_func(weights)[1] + l_reg* weights.T.dot(weights)))
                 
    # Inputs
    init_guess = np.repeat(1/n,n)
    bounds = ((-1 if short else 0, 1),) * n
    
    func_dict = {'mvo': mvo,
                 'factor': factor}
                 
    constraints = {'mvo': ({'type':'eq','fun': check_sum},
                           {'type':'ineq', 'fun': lambda x: x - min_weight}),
                   'factor': ({'type':'eq','fun': check_sum})}
        
    result = minimize(func_dict[optim_out],init_guess,method='SLSQP',bounds=bounds,constraints=constraints[optim_out])
        
    return result.x
    
from scipy.optimize import minimize

# create efficient frontier function
def create_frontier(asset_returns, betas=None, factors=None, error=None, optim_out = 'mvo', short = False, l_reg = 0, min_weight=0):
    
    gammas = np.logspace(-3,3,20)
    gam_wt = []
    for gamma in gammas:
        gam_wt.append(eff_ret(asset_returns, betas, factors, error, gamma, optim_out, short, l_reg, min_weight))

    df_mean = asset_returns.mean()
    df_cov = asset_returns.cov()

    ret_g, risk_g = [], []
    for g in gam_wt:
        ret_g.append(g.T @ df_mean)
        risk_g.append(np.sqrt(g.T @ df_cov @ g))

    return np.array(gam_wt), np.array(ret_g), np.array(risk_g)

def quick_plot(ret_g, risk_g):
    plt.figure()
    plt.plot(risk_g, ret_g, 'bo-')
    plt.show()
    
# Create graph functions to show results
def frontier_plot(portfolios, sharpe_ratios, returns_frontier, risk_frontier, max_sharp, max_ret_eff, min_vol, axes = [0.85, 0.6, 0.01, 0.2],
                 save_fig = False, fig_name=None):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    ax.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
    ax.scatter(max_sharp[1]*np.sqrt(12)*100, max_sharp[0]*1200,marker=(4,1,0),color='r',s=500)
    ax.scatter(max_ret_eff[1]*np.sqrt(12)*100,max_ret_eff[0]*1200,marker=(4,1,0),color='blue',s=500)
    ax.scatter(min_vol[1]*np.sqrt(12)*100,min_vol[0]*1200,marker=(4,1,0),color='purple',s=500)
    ax.plot(risk_frontier*np.sqrt(12)*100, returns_frontier*1200, 'bo-', linewidth=2)
    ax.set_title('Simulated portfolios', fontsize=20)
    ax.set_xlabel('Risk (%)')
    ax.set_ylabel('Return (%)')

    cbaxes = fig.add_axes(axes)
    clb = fig.colorbar(sim, cax = cbaxes)
    clb.ax.set_title(label='Sharpe', fontsize=10)
    
    if save_fig:
        save_fig_blog(fig_name, tight_layout=False)

    plt.show()
    
def frontier_wt_plot(frontier_weights, multiplier, nudge, save_fig = False, fig_name=None):
    
    wt_dict = {}
    for num, wts in enumerate(frontier_weights):
        min = 1
        ix = 0
        for idx, wt in enumerate(wts):
            if (wt < min) & (wt > 0):
                min = wt
                ix = idx
        wt_dict[num] = [ix,min] 

    plt.figure()
    vals = [x[1]*multiplier for x in wt_dict.values()]
    labs = [str(x+1) for x in wt_dict.keys()]
    asst = [x[0]+1 for x in wt_dict.values()]

    plt.bar(labs, vals, color = 'blue')

    for i in range(len(wt_dict)):
        plt.annotate(asst[i], xy = (i-0.2, vals[i]+nudge))

    
    plt.xlabel('Efficient portfolios')
    plt.ylabel(f'Weights % times {multiplier/100:.1e}')
    plt.title('Weighting of minimum weighted asset by efficient portfolio \nAnnotation: asset with the lowest weighting')

    if save_fig:
        save_fig_blog(fig_name)

    plt.show()

# Generate first efficient frontier and graph
wt_1, rt_1, rk_1 = create_frontier(assets, 'mvo', l_reg = 0)
max_sharp1 = port[np.argmax(sharpe)]
max_ret_eff1 = port[pd.DataFrame(np.c_[port,sharpe], columns = ['ret', 'risk', 'sharpe']).sort_values(['ret', 'sharpe'], ascending=False).index[0]]
min_vol1 = port[np.argmin(port[:,1])]

frontier_plot(port, sharpe, rt_1, rk_1, max_sharp1, max_ret_eff1, min_vol1, save_fig=False, fig_name = 'eff_front_1_23')

# Generate risk factor efficient frontier and graph
wt_f1, rt_f1, rk_f1 = create_frontier(assets, betas, fake_factors, error, optim_out = 'factor', short = False, l_reg = 0)

frontier_plot(port, sharpe, rt_f1, rk_f1, max_sharp1, max_ret_eff1, min_vol1, save_fig=False, fig_name='risk_front_1_24', 
              plot_title='Risk factor derived efficient frontier')


## Create new risk factor constrained optimization function
# We found these links useful:
  # https://nbviewer.jupyter.org/github/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%202.ipynb
  # https://riskfolio-lib.readthedocs.io/en/latest/constraints.html
# Note: we're calling it test, becuase it is still a work in progress. If it doesn't produce the results you see on the blog, let us know. So many iterations, so little time to check.

def eff_test(df_returns, betas=None, factors=None, error=None, gamma=1, optim_out = 'mvo', 
             short = False, l_reg = 0, min_weight = 0, wt_cons = [[0],[0]]):
    
    n = len(df_returns.columns)
    
    def get_data(weights):
        weights = np.array(weights)
        returns = np.sum(df_returns.mean() * weights)
        risk = np.sqrt(np.dot(weights.T, np.dot(df_returns.cov(), weights)))
        sharpe = returns/risk
        return np.array([returns,risk,sharpe])

    def check_sum(weights):
        return np.sum(np.abs(weights)) - 1

    def factor_var_func(weights):
    
        B = np.array(betas)
        F = np.array(factors.cov())
        S = np.diag(np.array(error.var()))

        factor_var = weights.dot(B.dot(F).dot(B.T)).dot(weights.T)
        specific_var = weights.dot(S).dot(weights.T)
         
        return factor_var, specific_var

    def factor_cons_func(weights):
    
        B = np.array(betas)
        F = np.diag(np.array(factors.var()))
        S = np.diag(np.array(error.var()))

        factor_var = weights.dot(B.dot(F).dot(B.T)).dot(weights.T)
        specific_var = weights.dot(S).dot(weights.T)
         
        return factor_var, specific_var

    def mvo(weights):
        return -(get_data(weights)[0] - gamma*(get_data(weights)[1]**2 + l_reg*weights.T.dot(weights)))
    
    def factor(weights):
        return -(get_data(weights)[0] - gamma*(factor_var_func(weights)[0] + factor_var_func(weights)[1] + l_reg* weights.T.dot(weights)))
    
    def factor_cons(weights):
        return -(get_data(weights)[0] - gamma*(factor_cons_func(weights)[0] + factor_cons_func(weights)[1] + l_reg* weights.T.dot(weights)))
    
    def factor_load(weights):
        B = np.array(betas)
        return weights.dot(B)
    
                 
    # Inputs
    init_guess = np.repeat(1/n,n)
    bounds = ((-1 if short else 0, 1),) * n
    
    func_dict = {'mvo': mvo,
                 'factor': factor,
                 'factor_cons_eq': factor,
                 'factor_cons_ineq': factor}
                 
    constraints = {'mvo': ({'type':'eq','fun': check_sum},
                           {'type':'ineq', 'fun': lambda x: x - min_weight}),
                   'factor': ({'type':'eq','fun': check_sum},
                              {'type':'ineq', 'fun': lambda x: x - min_weight}),
                   'factor_cons_eq': ({'type':'eq','fun': check_sum},
                              {'type':'eq', 'fun': lambda x: factor_load(x)[wt_cons[0]] - wt_cons[1]}),
                    'factor_cons_ineq': ({'type':'eq','fun': check_sum},
                              {'type':'ineq', 'fun': lambda x: factor_load(x)[wt_cons[0]]  - wt_cons[1]})}}
        
    result = minimize(func_dict[optim_out],init_guess,method='SLSQP',bounds=bounds,constraints=constraints[optim_out])
        
    return result.x
    
## Adjust frontier function
def front_test(asset_returns, betas=None, factors=None, error=None, optim_out = 'mvo', short = False, l_reg = 0, min_weight=0, wt_cons = [[0],[0]]):
    
    gammas = np.logspace(-3,3,20)
    gam_wt = []
    for gamma in gammas:
        gam_wt.append(eff_test(asset_returns, betas, factors, error, gamma, optim_out, short, l_reg, min_weight, wt_cons))

    df_mean = asset_returns.mean()
    df_cov = asset_returns.cov()

    ret_g, risk_g = [], []
    for g in gam_wt:
        ret_g.append(g.T @ df_mean)
        risk_g.append(np.sqrt(g.T @ df_cov @ g))

    return np.array(gam_wt), np.array(ret_g), np.array(risk_g)
    
    
## Calculate and graph factor loadings
    
pd.DataFrame(zip(betas.min(), betas.max()), columns = ['Min', 'Max']).plot(kind='bar', color=['grey', 'blue'],figsize=(12,6))
plt.xticks(np.arange(10), labels = [str(x+1) for x in np.arange(10)], rotation=0)
plt.axhline(betas.mean().mean(), color='red', linestyle='--')
plt.xlabel("Fake factors")
plt.ylabel(r"$\beta$s")
plt.title(r'Minimum and maximum loadings ($\beta$) by factor')
save_fig_blog('factor_loadings_24')
plt.show()

## Calculate and graph factor loadings for original MVO max Sharpe portfolio
idx = np.argmax(rt_1/rk_1)

X1 = sm.add_constant(fake_factors.values)
y = assets @ wt_1[idx]
results = sm.OLS(y, X1).fit()
coefs = results.params

plt.figure()
plt.bar(np.arange(1,11), coefs[1:], color="blue")
plt.xticks(np.arange(1,11), [str(x) for x in np.arange(1,11)])
plt.xlabel("Fake factor")
plt.ylabel(r'$\beta$')
plt.title(r'$\beta$s by factor for MVO-derived Maximum Sharpe ratio portfolio')
save_fig_blog('sharpe_betas_24')
plt.show()

## Create constraints on factor 4 and run risk constrained optimization
wt_cons2 = [[3], [0.05]]
wt_t1, rt_t1, rk_t1 = front_test(assets, betas, fake_factors, error, optim_out = 'factor_cons_eq', wt_cons = wt_cons2)


idx1 = np.argmax(rt_1/rk_1)
old_sharpe = np.array([rt_1[idx1], rk_1[idx1]])

idx2 = np.argmax(rt_t1/rk_t1)
new_sharpe = np.array([rt_t1[idx2], rk_t1[idx2]])

fig = plt.figure()
ax = fig.add_subplot(1,1, 1)
ax.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
ax.scatter(new_sharpe[1]*np.sqrt(12)*100, new_sharpe[0]*1200,marker=(4,1,0),color='r',s=500)
ax.scatter(old_sharpe[1]*np.sqrt(12)*100,old_sharpe[0]*1200,marker=(4,1,0),color='purple',s=500)
ax.scatter(max_ret_eff1[1]*np.sqrt(12)*100,max_ret_eff1[0]*1200,marker=(4,1,0),color='blue',s=500)
ax.plot(rk_t1*np.sqrt(12)*100, rt_t1*1200, 'bo-', linewidth=2)

ax.set_xlabel('Risk (%)')
ax.set_ylabel('Return (%)')

ax.set_title('Efficient frontier with new maximum Sharpe portfolio', fontsize=20)

save_fig_blog('new_sharpe_24', tight_layout=True)

plt.show()

## Create large constraint and run optimization
wt_cons1 = [[np.arange(10)], np.repeat(0.05, 10)]

wt_t, rt_t, rk_t = front_test(assets, betas, fake_factors, error, optim_out = 'factor_cons_ineq', wt_cons = wt_cons1)

idx1 = np.argmax(rt_1/rk_1)
old_sharpe = np.array([rt_1[idx1], rk_1[idx1]])

idx2 = np.argmax(rt_t1/rk_t1)
new_sharpe = np.array([rt_t1[idx2], rk_t1[idx2]])

idx3 = np.argmax(rt_t/rk_t)
cons_sharpe = np.array([rt_t[idx3], rk_t[idx3]])

fig = plt.figure()
ax = fig.add_subplot(1,1, 1)
ax.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
ax.scatter(new_sharpe[1]*np.sqrt(12)*100, new_sharpe[0]*1200,marker=(4,1,0),color='r',s=200)
ax.scatter(old_sharpe[1]*np.sqrt(12)*100,old_sharpe[0]*1200,marker=(4,1,0),color='purple',s=200)

ax.scatter(max_ret_eff1[1]*np.sqrt(12)*100,max_ret_eff1[0]*1200,marker=(4,1,0),color='blue',s=200)
ax.plot(rk_t*np.sqrt(12)*100, rt_t*1200, 'bo-', linewidth=2)
ax.scatter(cons_sharpe[1]*np.sqrt(12)*100,cons_sharpe[0]*1200,marker=(4,1,0),color='grey',s=400, zorder=4)

ax.set_xlabel('Risk (%)')
ax.set_ylabel('Return (%)')

ax.set_title('Risk constrained efficient frontier', fontsize=20)

save_fig_blog('risk_cons_front_24', tight_layout=True)

plt.show()

## Calculate and graph betas for risk constrained optimization
X1 = sm.add_constant(fake_factors.values)
y = assets @ wt_t[14]
results = sm.OLS(y, X1).fit()
coefs = results.params

plt.figure()
plt.bar(np.arange(1,11), coefs[1:], color="blue")
plt.xticks(np.arange(1,11), [str(x) for x in np.arange(1,11)])
plt.xlabel("Fake factor")
plt.ylabel(r'$\beta$')
plt.title(r'$\beta$s by factor after risk constrained optimization')
save_fig_blog('risk_cons_beta_24')

plt.show()

## R-squareds for fake factors on all assets

plt.figure()
plt.bar([str(x) for x in np.arange(1,31)], rsq, color='blue' )
plt.axhline(np.array(rsq).mean(), color='red', linestyle="--")
plt.title("$R^2$ with fake factors by fake asset number")
plt.ylabel("$R^2$ (%)")
# save_fig_blog('fake_asset_rsq_23')
plt.xlim([-1, len(rsq)-0.025])
plt.show()