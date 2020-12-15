## Round about the kernel
## Published 11/12/20

# Built using Python 3.7.4

## Import packages
import numpy as np
import pandas as pd
import pandas_datareader as dr
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,6)
plt.style.use('ggplot')

## Load data
prices = pd.read_pickle('xli_prices.pkl')
xli = pd.read_pickle('xli_etf.pkl')

returns = prices.drop(columns = ['OTIS', 'CARR']).pct_change()
returns.head()

xli_rets = xli.pct_change(60).shift(-60)

## Import cause_lists created using R
# See R code above to create
cause_lists = pd.read_csv("cause_lists.csv",header=None)
cause_lists = cause_lists.iloc[1:,1:]

## Define correlation function
def mean_cor(df):
    corr_df = df.corr()
    np.fill_diagonal(corr_df.values, np.nan)
    return np.nanmean(corr_df.values)
    
## Create data frames and train-test splits    
corr_comp = pd.DataFrame(index=returns.index[59:])
corr_comp['corr'] = [mean_cor(returns.iloc[i-59:i+1,:]) for i in range(59,len(returns))]
xli_rets = xli.pct_change(60).shift(-60)
total_60 = pd.merge(corr_comp, xli_rets, how="left", on="Date").dropna()
total_60.columns = ['corr', 'xli']

split = round(len(total_60)*.7)
train_60 = total_60.iloc[:split,:]
test_60 = total_60.iloc[split:, :]

tot_returns = pd.merge(xli,prices.drop(columns = ["CARR", "OTIS"]), "left", "Date")
tot_returns = tot_returns.rename(columns = {'Adj Close': 'xli'})
tot_returns = tot_returns.pct_change()
tot_split = len(train_60)+60
train = tot_returns.iloc[:tot_split,:]
test = tot_returns.iloc[tot_split:len(tot_returns),:]
train.head()

## Create period indices to run pairwise correlations and forward returns for regressions
cor_idx = np.array((np.arange(190,500), np.arange(440,750), np.arange(690,1000), np.arange(940,1250),
                np.arange(1190,1500), np.arange(1440,1750), np.arange(1690,2000), np.arange(1940,2250),
                np.arange(2190,2500)))

# Add 1 since xli is price while train is ret so begin date is off by 1 biz day
ret_idx = np.array((np.arange(250,561), np.arange(500,811), np.arange(750,1061), np.arange(1000,1311),
                np.arange(1250,1561), np.arange(1500,1811), np.arange(1750,2061), np.arange(2000,2311),
                np.arange(2250,2561)))

# Create separate data arrays using cause_lists and indices
# Causal subset
merge_list = [0]*9
for i in range(len(cor_idx)):
        dat = train.reset_index().loc[cor_idx[i],cause_lists.iloc[i,:].dropna()]
        corr = [mean_cor(dat.iloc[i-59:i+1,:]) for i in range(59,len(dat))]
        ret1 = xli.reset_index().iloc[ret_idx[i],1]
        ret1 = ret1.pct_change(60).shift(-60).values
        ret1 = ret1[~np.isnan(ret1)]
        merge_list[i] = np.c_[corr, ret1]
        
# Non-causal subset        
non_cause_list = [0] * 9
for i in range(len(cor_idx)):
    non_c = [x for x in list(train.columns[1:]) if x not in cause_lists.iloc[3,:].dropna().to_list()]
    dat = train.reset_index().loc[cor_idx[i], non_c]
    corr = [mean_cor(dat.iloc[i-59:i+1,:]) for i in range(59,len(dat))]
    ret1 = xli.reset_index().iloc[ret_idx[i],1]
    ret1 = ret1.pct_change(60).shift(-60).values
    ret1 = ret1[~np.isnan(ret1)]
    non_cause_list[i] = np.c_[corr, ret1]
    
    
# Create single data set for example
cause_ex = np.c_[merge_list[2],non_cause_list[2][:,0]]

# Run linear regression
from sklearn.linear_model import LinearRegression
X = cause_ex[:,0].reshape(-1,1)
y = cause_ex[:,1]
lin_reg = LinearRegression().fit(X,y)
y_pred = lin_reg.predict(X)

# Graph scatterplot with lowess and linear regression
import seaborn as sns
sns.regplot(cause_ex[:,0]*100, cause_ex[:,1]*100, color = 'blue', lowess=True, line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.4})
plt.plot(X*100, y_pred*100, color = 'darkgrey', linestyle='dashed')
plt.xlabel("Correlation (%)")
plt.ylabel("Return (%)")
plt.title("Return (XLI) vs. correlation (causal subset)")
plt.show()

# Run linear regression on non-causal component of cause_ex data frame
from sklearn.linear_model import LinearRegression
X_non = cause_ex[:,2].reshape(-1,1)
y = cause_ex[:,1]
lin_reg_non = LinearRegression().fit(X_non,y)
y_pred_non = lin_reg_non.predict(X_non)

# Graph scatter plot
sns.regplot(cause_ex[:,2]*100, cause_ex[:,1]*100, color = 'blue', lowess=True, line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.4})
plt.plot(X_non*100, y_pred_non*100, color = 'darkgrey', linestyle='dashed')
plt.xlabel("Correlation (%)")
plt.ylabel("Return (%)")
plt.title("Return (XLI) vs. correlation (non-causal subset)")
plt.show()

## Run regressions on cause_ex
from sklearn_extensions.kernel_regression import KernelRegression
import statsmodels.api as sm

x = cause_ex[:,0]
X = sm.add_constant(x)
x_non = cause_ex[:,2]
X_non = sm.add_constant(x_non)
y = cause_ex[:,1]
lin_c = sm.OLS(y,X).fit().rsquared*100
lin_nc = sm.OLS(y,X_non).fit().rsquared*100

# Note KernelRegressions() returns different results than kern() from generalCorr
kr = KernelRegression(kernel='rbf', gamma=np.logspace(-5,5,10)) 
kr.fit(X,y)
kr_c = kr.score(X,y)*100

kr.fit(X_non, y)
kr_nc = kr.score(X_non, y)*100

print(f"R-squared for kernel regression causal subset: {kr_c:0.01f}")
print(f"R-squared for kernel regression non-causal subset: {kr_nc:0.01f}")
print(f"R-squared for linear regression causal subset: {lin_c:0.01f}")
print(f"R-squared for linear regression non-causal subset: {lin_nc:0.01f}")


## Run regressions on data lists
import statsmodels.api as sm

# Causal subset linear model
lin_mod = []
for i in range(len(merge_list)):
    x = merge_list[i][:,0]
    X = sm.add_constant(x)
    y = merge_list[i][:,1]
    mod_reg = sm.OLS(y,X).fit()
    lin_mod.append(mod_reg.rsquared)

start = train.index[np.arange(249,2251,250)].year
end = train.index[np.arange(499,2500,250)].year

model_dates = [str(x)+"-"+str(y) for x,y in zip(start,end)]

# Non-causal subset linear model
non_lin_mod = []
for i in range(len(non_cause_list)):
    x = non_cause_list[i][:,0]
    X = sm.add_constant(x)
    y = non_cause_list[i][:,1]
    mod_reg = sm.OLS(y,X).fit()
    non_lin_mod.append(mod_reg.rsquared)
    
    
# Causal subset kernel regression
from sklearn_extensions.kernel_regression import KernelRegression

kern = []
for i in range(len(merge_list)):
    X = merge_list[i][:,0].reshape(-1,1)
    y = merge_list[i][:,1]
    kr = KernelRegression(kernel='rbf', gamma=np.logspace(-5,5,10))
    kr.fit(X,y)
    kern.append(kr.score(X,y))

    
## Plot R-squared comparisons

# Causal kernel vs. linear    
df = pd.DataFrame(np.c_[np.array(kern)*100, np.array(lin_mod)*100], columns = ['Kernel', 'Linear'])
df.plot(kind='bar', color = ['blue','darkgrey'])
plt.xticks(ticks = df.index, labels=model_dates, rotation=0)
plt.legend(loc = 'upper left')
plt.show()

# Causal kerner vs causal & non-causal linear
df = pd.DataFrame(np.c_[np.array(kern)*100, np.array(lin_mod)*100, np.array(non_lin_mod)*100], 
                  columns = ['Kernel', 'Linear-causal', 'Linear--non-causal'])
df.plot(kind='bar', color = ['blue','darkgrey', 'darkblue'], width=.85)
plt.xticks(ticks = df.index, labels=model_dates, rotation=0)
plt.legend(bbox_to_anchor=(0.3, 0.9), loc = 'center')
plt.ylabel("R-squared (%)")
plt.title("R-squared output for regression results by period and model")
plt.show()

## Create RMSE lists
lin_rmse = []
for i in range(len(merge_list)):
    x = merge_list[i][:,0]
    X = sm.add_constant(x)
    y = merge_list[i][:,1]
    mod_reg = sm.OLS(y,X).fit()
    lin_rmse.append(np.sqrt(mod_reg.mse_resid))
    
lin_non_rmse = []
for i in range(len(non_cause_list)):
    x = non_cause_list[i][:,0]
    X = sm.add_constant(x)
    y = non_cause_list[i][:,1]
    mod_reg = sm.OLS(y,X).fit()
    lin_non_rmse.append(np.sqrt(mod_reg.mse_resid))
    
kern_rmse = []
for i in range(len(merge_list)):
    X = merge_list[i][:,0].reshape(-1,1)
    y = merge_list[i][:,1]
    kr = KernelRegression(kernel='rbf', gamma=np.logspace(-5,5,10)) 
    kr.fit(X,y)
    rmse = np.sqrt(np.mean((kr.predict(X)-y)**2))
    kern_rmse.append(rmse)
    
## Graph RMSE comparisons
df = pd.DataFrame(np.c_[np.array(kern_rmse)*100, np.array(lin_rmse)*100, np.array(lin_non_rmse)*100], 
                  columns = ['Kernel', 'Linear-causal', 'Linear--non-causal'])
df.plot(kind='bar', color = ['blue','darkgrey', 'darkblue'], width=.85)
plt.xticks(ticks = df.index, labels=model_dates, rotation=0)
plt.legend(loc = 'upper left')
plt.ylabel("RMSE (%)")
plt.title("RMSE results by period and model")
plt.show()


## Graph RMSE differences
kern_lin = [x-y for x,y in zip(lin_rmse, kern_rmse)]
kern_non = [x-y for x,y in zip(lin_non_rmse, kern_rmse)]
df = pd.DataFrame(np.c_[np.array(kern_lin)*100, np.array(kern_non)*100], 
                  columns = ['Kernel - Linear-causal', 'Kernel - Linear--non-causal'])
df.plot(kind='bar', color = ['darkgrey', 'darkblue'], width=.85)
plt.xticks(ticks = df.index, labels=model_dates, rotation=0)
plt.legend(loc = 'upper left')
plt.ylabel("RMSE (%)")
plt.title("RMSE differences by period and model")
plt.show()

## Graph XLI
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(xli["2010":"2014"], color='blue')
ax.set_label("")
ax.set_ylabel("Price (US$)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(ScalarFormatter())
ax.set_title("XLI price log-scale")
plt.show()