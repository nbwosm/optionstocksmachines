# Portfolio simulation functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Simulation function
# Portfolio simulation functions

## Simulation function
class Port_sim:
    
    def calc_sim(df, sims, cols):
        wts = np.zeros((sims, cols))

        for i in range(sims):
            a = np.random.uniform(0,1,cols)
            b = a/np.sum(a)
            wts[i,] = b

        mean_ret = df.mean()
        port_cov = df.cov()
    
        port = np.zeros((sims, 2))
        for i in range(sims):
            port[i,0] =  np.sum(wts[i,:]*mean_ret)
            port[i,1] = np.sqrt(np.dot(np.dot(wts[i,:].T,port_cov), wts[i,:]))

        sharpe = port[:,0]/port[:,1]*np.sqrt(12)
    
        return port, wts, sharpe
    
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
    
    def graph_sim(port, sharpe):
        plt.figure(figsize=(14,6))
        plt.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
        plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
        plt.title('Simulated portfolios', fontsize=20)
        plt.xlabel('Risk (%)')
        plt.ylabel('Return (%)')
        plt.show()
        
# Constraint function
def port_select_func(port, wts, return_min, risk_max):
    port_select = pd.DataFrame(np.concatenate((port, wts), axis=1))
    port_select.columns = ['returns', 'risk', 1, 2, 3, 4]
    
    port_wts = port_select[(port_select['returns']*12 >= return_min) & (port_select['risk']*np.sqrt(12) <= risk_max)]
    port_wts = port_wts.iloc[:,2:6]
    port_wts = port_wts.mean(axis=0)
    
    def graph():
        plt.figure(figsize=(12,6))
        key_names = {1:"Stocks", 2:"Bonds", 3:"Gold", 4:"Real estate"}
        lab_names = []
        graf_wts = port_wts.sort_values()*100
        
        for i in range(len(graf_wts)):
            name = key_names[graf_wts.index[i]]
            lab_names.append(name)

        plt.bar(lab_names, graf_wts)
        plt.ylabel("Weight (%)")
        plt.title("Average weights for risk-return constraint", fontsize=15)
        
        for i in range(len(graf_wts)):
            plt.annotate(str(round(graf_wts.values[i])), xy=(lab_names[i], graf_wts.values[i]+0.5))
    
        plt.show()
    
    return port_wts, graph()

# Return function with no rebalancing
def rebal_func(act_ret, weights):
    ret_vec = np.zeros(len(act_ret))
    wt_mat = np.zeros((len(act_ret), len(act_ret.columns)))
    for i in range(len(act_ret)):
        wt_ret = act_ret.iloc[i,:].values*weights
        ret = np.sum(wt_ret)
        ret_vec[i] = ret
        weights = (weights + wt_ret)/(np.sum(weights) + ret)
        wt_mat[i,] = weights
    
    return ret_vec, wt_mat