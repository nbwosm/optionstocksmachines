# Portfolio risk functions

# Load libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def rsq_func_prem(ind_df, dep_df, look_forward = None, risk_premium = False, period = 60, start_date=0, \
             plot=True, asset_names = True, print_rsq = True, chart_title = None,\
             y_lim = None, save_fig = False, fig_name = None):
    """ Assumes ind_df starts from the same date as dep_df.
        Dep_df has only as many columns as interested for modeling. """

    xs = ind_df[0:start_date+period]
    
    assets = dep_df.columns.to_list()
    
    rsq = []
    
    if look_forward:
        start = start_date + look_forward
        end = start_date + look_forward + period
    else:
        start = start_date 
        end = start_date + period
    
    if risk_premium:
        # create mask to remove market risk premium from stock regression
        mask = [x for x in ind_df.columns.to_list() if x != 'mkt-rfr']
        for asset in assets:
            if asset == 'stock':
                X = sm.add_constant(xs.loc[:,mask])
            else:
                X = sm.add_constant(xs)
            y = dep_df[asset][start:end].values
            mod = sm.OLS(y, X).fit().rsquared*100
            rsq.append(mod)
            if print_rsq:
                print(f'R-squared for {asset} is {mod:0.03f}')      
    else:
        X = sm.add_constant(xs)
        for asset in assets:
            y = dep_df[asset][start:end].values
            mod = sm.OLS(y, X).fit().rsquared*100
            rsq.append(mod)
            if print_rsq:
                print(f'R-squared for {asset} is {mod:0.03f}')


    if plot:
        if asset_names:
            x_labels = ['Stocks', 'Bonds', 'Gold', 'Real estate']
        else:
            x_labels = asset_names

        plt.figure(figsize=(12,6))
        plt.bar(x_labels, rsq, color='blue')

        for i in range(len(x_labels)):
            plt.annotate(str(round(rsq[i]),), xy = (x_labels[i], rsq[i]+0.5))

   
        plt.ylabel("$R^{2}$")
        
        if chart_title:
            plt.title(chart_title)
        else:
            plt.title("$R^{2}$ for Macro Risk Factor Model")
        
        plt.ylim(y_lim)
        
        if save_fig:
            save_fig_blog(fig_name)
        else:
            plt.tight_layout()
                    
        plt.show()
        
    return rsq
    

def factor_beta_risk_premium_calc(ind_df, dep_df, risk_premium = False):
    
    xs =  ind_df
    factor_names = [x.lower() for x in ind_df.columns.to_list()]
    assets = dep_df.columns.to_list()
    
    betas = pd.DataFrame(index=dep_df.columns)
    pvalues = pd.DataFrame(index=dep_df.columns)
    error = pd.DataFrame(index=dep_df.index)

    if risk_premium:
        mask = [x for x in ind_df.columns.to_list() if x != 'mkt-rfr'] # remove market risk premium from independent variables
        zero_val = np.where(ind_df.columns == 'mkt-rfr')[0][0] # identify index of market risk premium
        
        for asset in assets:
            if asset == 'stock':
                X = sm.add_constant(xs.loc[:,mask])
                y = dep_df[asset].values
                result = sm.OLS(y, X).fit()
                # pad results for missing market risk premium
                results = np.array([x for x in result.params[1:zero_val+1]] + [0.0] + [x for x in result.params[zero_val+1:]])
                
                for j in range(len(results)):
                    # results and factor names have same length
                    betas.loc[asset, factor_names[j]] = results[j]
                    pvalues.loc[asset, factor_names[j]] = results[j]
                    
            else:
                X = sm.add_constant(xs)
                y = dep_df[asset].values
                result = sm.OLS(y, X).fit()
            
                for j in range(1, len(result.params)):
                        # result.params equals length of factor_names + 1 due to intercept so start at 1
                        betas.loc[asset, factor_names[j-1]] = result.params[j]
                        pvalues.loc[asset, factor_names[j-1]] = result.pvalues[j]
                                        
            # Careful of error indentation: lopping through assets
            error.loc[:,asset] = (y - X.dot(result.params))
        
    else:
        X = sm.add_constant(xs)
        for asset in assets:
            y = dep_df[asset].values
            result = sm.OLS(y, X).fit()
            
            for j in range(1, len(result.params)):
                betas.loc[asset, factor_names[j-1]] = result.params[j]
                pvalues.loc[asset, factor_names[j-1]] = result.pvalues[j]

            error.loc[:,asset] = (y - X.dot(result.params))
          
    return betas, pvalues, error

def betas_plot(beta_df, colors, legend_names, save_fig = False, fig_name=None):
    beta_df.plot(kind='bar', width = 0.75, color= colors, figsize=(12,6))
    plt.legend(legend_names)
    plt.xticks([0,1,2,3], ['Stock', 'Bond', 'Gold', 'Real estate'], rotation=0)
    plt.ylabel(r'Factor $\beta$s')
    plt.title(r'Factor $\beta$s by asset class')
    if save_fig:
        save_fig_blog(fig_name)
    plt.show()
    
    
def factor_port_var(betas, factors, weights, error):
    
        B = np.array(betas)
        F = np.array(factors.cov())
        S = np.diag(np.array(error.var()))

        factor_var = weights.dot(B.dot(F).dot(B.T)).dot(weights.T)
        specific_var = weights.dot(S).dot(weights.T)
            
        return factor_var, specific_var
    
    
def port_var_plot(port_exp, port_names=None, y_lim=None, save_fig=False, fig_name=None, fig_title = False, fig_title_name = None):
    if not port_names:
        port_names = ['Satisfactory', 'Naive', 'Max Sharpe', 'Max Return']
    else:
        port_names = port_names
        
    plt.figure(figsize=(12,6))
    plt.bar(port_names, port_exp*100, color='blue')

    for i in range(4):
        plt.annotate(str(round(port_exp[i]*100,1)) + '%', xy = (i-0.05, port_exp[i]*100+0.5))

    if fig_title:
        plt.title(fig_title_name)
    else:
        plt.title('Original four portfolios\' variance explained by Macro risk factor model')
    
    plt.ylabel('Variance explained (%)')
    plt.ylim(y_lim)
    if save_fig:
        save_fig_blog(fig_name)
    plt.show()