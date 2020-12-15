# Explaining variance
# Published 12/14/20

# Built using R 4.0.3 and Python 3.8.3

### R
## Load packages
suppressPackageStartupMessages({
  library(tidyquant) # Not really necessary, but force of habit
  library(tidyverse) # Not really necessary, but force of habit
  library(reticulate) # development version
})

# Allow variables in one python chunk to be used by other chunks.
knitr::knit_engines$set(python = reticulate::eng_python) 

### Python from here on!
# Load libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import os

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/usr/Anaconda3/Library/plugins/platforms'
plt.style.use('ggplot')

## Load asset data
df = pd.read_pickle('port_const.pkl') # Check out http://www.optionstocksmachines.com/ for how we pulled in the data.
df.iloc[0,3] = 0.006 # Interpolation

## Load ff data
ff_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_3_Factors_CSV.zip"
col_names = ['date', 'mkt-rfr', 'smb', 'hml', 'rfr']
ff = pd.read_csv(ff_url, skiprows=6, header=0, names = col_names)
ff = ff.iloc[:364,:]

from pandas.tseries.offsets import MonthEnd
ff['date'] = pd.to_datetime([str(x[:4]) + "/" + str.rstrip(x[4:]) for x in ff['date']], format = "%Y-%m") + MonthEnd(1)
ff.iloc[:,1:] = ff.iloc[:,1:].apply(pd.to_numeric)

momo_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
momo = pd.read_csv(momo_url, skiprows=13, header=0, names=['date', 'mom'])
momo = momo.iloc[:1125,:]

momo['date'] = pd.to_datetime([str(x[:4]) + "/" + str(x[4:]) for x in momo['date']], format = "%Y-%m") + MonthEnd(1)
momo['mom'] = pd.to_numeric(momo['mom'])

ff_mo = pd.merge(ff, momo, how = 'left', on='date')
col_ord = [x for x in ff_mo.columns.to_list() if x not in ['rfr']] + ['rfr']
ff_mo = ff_mo.loc[:,col_ord]
ff_mo = ff_mo[(ff_mo['date']>="1987-01-31") & (ff_mo['date']<="2019-12-31")].reset_index(drop=True)

## Plot ff
ff_factors =  ['Risk premium', 'SMB', 'HML', 'Momemtum']

fig, axes = plt.subplots(4,1, figsize=(10,8))
for idx, ax in enumerate(fig.axes):
    ax.plot(ff_mo.iloc[:60,0], ff_mo.iloc[:60,idx+1], linestyle = "dashed", color='blue')
    ax.set_title(ff_factors[idx], fontsize=10, loc='left')
    if idx % 2 != 0:
        ax.set_ylabel("Returns (%)")
        
fig.tight_layout(pad = 0.5)
plt.tight_layout()
plt.show()

## Abbreviated Simulation function
class Port_sim:
    import numpy as np
    import pandas as pd

    def calc_sim_lv(df, sims, cols):
        wts = np.zeros(((cols-1)*sims, cols))
        count=0

        for i in range(1,cols):
            for j in range(sims):
                a = np.random.uniform(0,1,(cols-i+1))
                b = a/np.sum(a)
                c = np.random.choice(np.concatenate((b, np.zeros(i-1))),cols, replace=False)
                wts[count,:] = c
                count+=1

        mean_ret = df.mean()
        port_cov = df.cov()

        rets=[]
        vols=[]
        for i in range((cols-1)*sims):
            rets.append(np.sum(wts[i,:]*mean_ret))
            vols.append(np.sqrt(np.dot(np.dot(wts[i,:].T,port_cov), wts[i,:])))

        port = np.c_[rets, vols]

        sharpe = port[:,0]/port[:,1]*np.sqrt(12)

        return port, wts, sharpe

      
## Simulate portfolios
port1, wts1, sharpe1  = Port_sim.calc_sim_lv(df.iloc[1:60, 0:4], 10000,4)

## Plot simulated portfolios
max_sharp1 = port1[np.argmax(sharpe1)]
min_vol1 = port1[np.argmin(port1[:,1])]

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1, 1)
sim = ax.scatter(port1[:,1]*np.sqrt(12)*100, port1[:,0]*1200, marker='.', c=sharpe1, cmap='Blues')
ax.scatter(max_sharp1[1]*np.sqrt(12)*100, max_sharp1[0]*1200,marker=(4,1,0),color='r',s=500)
ax.scatter(min_vol1[1]*np.sqrt(12)*100,min_vol1[0]*1200,marker=(4,1,0),color='purple',s=500)
ax.set_title('Simulated portfolios', fontsize=20)
ax.set_xlabel('Risk (%)')
ax.set_ylabel('Return (%)')

cbaxes = fig.add_axes([0.15, 0.6, 0.01, 0.2])
clb = fig.colorbar(sim, cax = cbaxes)
clb.ax.set_title(label='Sharpe', fontsize=10)

plt.tight_layout()
plt.show()

## Calculate betas for asset classes
X = sm.add_constant(ff_mo.iloc[:60,1:5])

rsq = []
for i in range(4):
    y = df.iloc[:60,i].values - ff_mo.loc[:59, 'rfr'].values
    mod = sm.OLS(y, X).fit().rsquared*100
    rsq.append(mod)

asset_names = ['Stocks', 'Bonds', 'Gold', 'Real estate']

fact_plot = pd.DataFrame(zip(asset_names,rsq), columns = ['asset_names', 'rsq'])

## Plot betas
ax = fact_plot['rsq'].plot(kind = "bar", color='blue', figsize=(12,6))
ax.set_xticklabels(asset_names, rotation=0)
ax.set_ylabel("$R^{2}$")
ax.set_title("$R^{2}$ for Fama-French Four Factor Model")
ax.set_ylim([0,45])

## Iterate through annotation
for i in range(4):
    plt.annotate(str(round(rsq[i]))+'%', xy = (fact_plot.index[i]-0.05, rsq[i]+1))

plt.tight_layout()
plt.show()

## Note: reticulate does not like plt.annotate() and throws errors left, right, and center if you
## don't ensure that the x ticks are numeric, which means you have to label the xticks separately
## through the axes setting. Very annoying!

# Find factor exposures
assets = df.iloc[:60,:4]
betas = pd.DataFrame(index=assets.columns)
error = pd.DataFrame(index=assets.index)

# Create betas and error 
# Code derived from Quantopian 
X = sm.add_constant(ff_mo.iloc[:60,1:5])

for i in assets.columns:
    y = assets.loc[:,i].values - ff_mo.loc[:59,'rfr'].values
    result = sm.OLS(y, X).fit()

    betas.loc[i,"mkt_beta"] = result.params[1]
    betas.loc[i,"smb_beta"] = result.params[2]
    betas.loc[i,"hml_beta"] = result.params[3]
    betas.loc[i,'momo_beta'] = result.params[4]
    
    # We don't show the p-values in the post, but did promise to show how we coded it.
    pvalues.loc[i,"mkt_p"] = result.pvalues[1] 
    pvalues.loc[i,"smb_p"] = result.pvalues[2]
    pvalues.loc[i,"hml_p"] = result.pvalues[3]
    pvalues.loc[i,'momo_p'] = result.pvalues[4]
        
    error.loc[:,i] = (y - X.dot(result.params)).values

 # Plot the betas
(betas*100).plot(kind='bar', width = 0.75, color=['darkblue', 'blue', 'grey', 'darkgrey'], figsize=(12,6))
plt.legend(['Risk premium', 'SMB', 'HML', 'Momentum'], loc='upper right')
plt.xticks([0,1,2,3], ['Stock', 'Bond', 'Gold', 'Real estate'], rotation=0)
plt.ylabel(r'Factor $\beta$s ')
plt.title('')
plt.tight_layout()
plt.show()

# Create variance contribution function
def factor_port_var(betas, factors, weights, error):
    
        B = np.array(betas)
        F = np.array(factors.cov())
        S = np.diag(np.array(error.var()))

        factor_var = weights.dot(B.dot(F).dot(B.T)).dot(weights.T)
        specific_var = weights.dot(S).dot(weights.T)
            
        return factor_var, specific_var

# Iterate variance calculation through portfolios                
facts = ff_mo.iloc[:60, 1:5]
fact_var = []
spec_var = []

for i in range(len(wts1)):
    out = factor_port_var(betas, facts, wts1[i], error)
    fact_var.append(out[0])
    spec_var.append(out[1])    

vars = np.array([fact_var, spec_var])

## Find max sharpe and min vol portfolio
max_sharp_var = [exp_var[np.argmax(sharpe1)], port1[np.argmax(sharpe1)][1]]
min_vol_var = [exp_var[np.argmin(port1[:,1])], port1[np.argmin(port1[:,1])][1]]

## Plot variance explained vs. volatility
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1, 1)
sim = ax.scatter(port1[:,1]*np.sqrt(12)*100, exp_var, marker='.', c=sharpe1, cmap='Blues')
ax.scatter(max_sharp_var[1]*np.sqrt(12)*100, max_sharp_var[0],marker=(4,1,0),color='r',s=500)
ax.scatter(min_vol_var[1]*np.sqrt(12)*100,min_vol_var[0],marker=(4,1,0),color='purple',s=500)
ax.set_title('Portfolio variance due to risk factors vs. portfolio volatility ', fontsize=20)
ax.set_xlabel('Portfolio Volatility (%)')
ax.set_ylabel('Risk factor variance contribution (%)')
ax.set_xlim([0,13])

cbaxes = fig.add_axes([0.15, 0.6, 0.01, 0.2])
clb = fig.colorbar(sim, cax = cbaxes)
clb.ax.set_title(label='Sharpe', fontsize=10)

plt.tight_layout()
plt.show()

## Create ranking data frame
rank = pd.DataFrame(zip(port1[:,1], exp_var), columns=['vol', 'exp_var'])
rank = rank.sort_values('vol')
rank['decile'] = pd.qcut(rank['vol'], 10, labels = False)
vol_rank = rank.groupby('decile')[['vol','exp_var']].mean()

vols = (vol_rank['vol'] * np.sqrt(12)*100).values

## Plot explained variance vs. ranking
ax = vol_rank['exp_var'].plot(kind='bar', color='blue', figsize=(12,6))
ax.set_xticklabels([x for x in np.arange(1,11)], rotation=0)
ax.set_xlabel('Decile')
ax.set_ylabel('Risk factor explained variance (%)')
ax.set_title('Variance explained by risk factor grouped by volatility decile\nwith average volatility by bar')
ax.set_ylim([20,40])

for i in range(10):
    plt.annotate(str(round(vols[i],1))+'%', xy = (vol_rank.index[i]-0.2, vol_rank['exp_var'][i]+1))

plt.tight_layout()
plt.show()


## Show grouping of portfolios
## Note we could not get this to work within reticulate, so simply saved the graph as a png.
## This did work in jupyter, however.
wt_df = pd.DataFrame(wts1, columns = assets.columns)

indices = []

for asset in assets.columns:
    idx = np.array(wt_df[wt_df[asset] > 0.5].index)
    indices.append(idx)

eq_wt = []
for i, row in wt_df.iterrows():
    if row.max() < 0.3:
        eq_wt.append(i)

exp_var_asset = []

for i in range(4):
    out = np.mean(exp_var[indices[i]])
    exp_var_asset.append(out)

exp_var_asset.append(np.mean(exp_var[eq_wt]))

mask = np.concatenate((np.concatenate(indices), np.array(eq_wt)))
exp_var_asset.append(np.mean(exp_var[~mask]))

plt.figure(figsize=(12,6))
asset_names = ['Stocks', 'Bonds', 'Gold', 'Real estate']
plt.bar(['All'] + asset_names + ['Equal', 'Remainder'], exp_var_asset, color = "blue")

for i in range(len(exp_var_asset)):
    plt.annotate(str(round(exp_var_asset[i])) + '%', xy = (i-0.05, exp_var_asset[i]+1))
    
plt.title('Portfolio variance explained by factor model for asset and equal-weighted models')
plt.ylabel('Variance explained (%)')
plt.ylim([10,50])

plt.tight_layout()
plt.show()

# This is the error we'd get every time we ran the code in blogdown.
# Error in py_call_impl(callable, dots$args, dots$keywords) : 
#   TypeError: only integer scalar arrays can be converted to a scalar index
# 
# Detailed traceback: 
#   File "<string>", line 2, in <module>
# Calls: local ... py_capture_output -> force -> <Anonymous> -> py_call_impl
# Execution halted
# Error in render_page(f) : 
#   Failed to render 'content/post/2020-12-01-port-20/index.Rmd'

## Instantiate original four portfolio weights 
satis_wt = np.array([0.32, 0.4, 0.2, 0.08])
equal_wt = np.repeat(0.25,4)
max_sharp_wt = wts1[np.argmax(sharpe1)]
max_ret_wt = wts1[pd.DataFrame(np.c_[port1,sharpe1], columns = ['ret', 'risk', 'sharpe']).sort_values(['ret', 'sharpe'], ascending=False).index[0]]

## Loop through weights to calculate explained variance
wt_list = [satis_wt, equal_wt, max_sharp_wt, max_ret_wt]
port_exp=[]

for wt in wt_list:
    out = factor_port_var(betas, facts, wt, error)
    port_exp.append(out[0]/(out[0] + out[1]))

port_exp = np.array(port_exp)

## Graph portfolio
## We didn't even bother trying to make this work in blogdown and just saved direct to a png.
port_names = ['Satisfactory', 'Naive', 'Max Sharpe', 'Max Return']
plt.figure(figsize=(12,6))
plt.bar(port_names, port_exp*100, color='blue')

for i in range(4):
    plt.annotate(str(round(port_exp[i]*100)) + '%', xy = (i-0.05, port_exp[i]*100+0.5))
    
plt.title('Original four portfolios variance explained by factor models')
plt.ylabel('Variance explained (%)')
plt.ylim([10,50])
plt.show()