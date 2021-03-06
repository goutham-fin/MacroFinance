
import sys
sys.path.insert(0,'../Models/')
from Extended.model_class import model_nnpde
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fsolve
import scipy.interpolate
import scipy as sp
from pylab import plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 22})
import numpy as np
import pandas as pd
import seaborn as sns
from Benchmark.interpolate_var import interpolate_grid as interp_grid
from Benchmark.interpolate_var import interpolate_simple as interp_simple
from Benchmark.interpolate_var import interpolate_loop
import time
import os
import dill
import statsmodels.tsa.api as tsa
import statsmodels.regression.linear_model as sm
import statsmodels.api as sm_filters
from statsmodels.api import add_constant
from statsmodels.iolib.summary2 import summary_col
import statistics
from itertools import groupby
import warnings
warnings.filterwarnings("ignore")

class simulationExtended():

    def __init__(self,params):
        self.params = params
        self.T = 5000
        self.dt = 1/12
        self.t = np.arange(0,self.T,self.dt)
        self.burn_period = 1000/self.dt
        
        if self.params['load_pickle'] == True:
            def read_pickle(filename):
                with open(str(filename) + '.pkl', 'rb') as f:
                    return dill.load(f)

            self.ex = read_pickle('../Models/Extended/model2D')
        else:
            self.ex =  model_nnpde(params)
            self.ex.solve()
        self.z, self.f, self.crisis_z, self.mu_z, self.sig_zk, self.sig_zf, self.ssq, self.ssf, self.iota, self.theta, self.thetah, self.rp, self.rp_, self.r, self.Q, self.rho, self.Phi, self.params['delta'], self.params['sigma'], self.beta_f = self.ex.z, self.ex.f, self.ex.crisis, self.ex.mu_z, self.ex.sig_zk, self.ex.sig_zf, self.ex.ssq, self.ex.ssf, self.ex.iota, \
                                                            self.ex.theta, self.ex.thetah, self.ex.rp, self.ex.rp_, self.ex.r, self.ex.q, self.ex.params['rho'], self.ex.Phi, self.ex.params['delta'], self.ex.params['sigma'], self.ex.params['beta_f']
        self.rp = self.ex.rp
        self.crisis_flag = self.ex.crisis_flag
        self.A, self.AminusIota, self.pd = self.ex.A, self.ex.AminusIota, self.ex.pd
        self.mrpk_e, self.mrpf_e = self.ex.priceOfRiskE_k, self.ex.priceOfRiskE_f
        self.psi = self.ex.psi
        self.mu_re = self.ex.mu_rE
        try:
            if not os.path.exists('../output'):
                os.mkdir('../output')
        except:
            print('Warning: Cannot create directory')
        self.interp_method = 'bivariate' #interp2d sorts by x and y axis automatically
        
        #modify crisis region
        self.crisis_flag[:,self.crisis_flag.shape[1]//2:]=0
        
    def interpolate_values(self):
        if self.interp_method == 'bivariate':
            self.mu_z_fn = RectBivariateSpline(self.z,self.f,self.mu_z,kx=1,ky=1)
            self.sig_zk_fn = RectBivariateSpline(self.z,self.f,self.sig_zk,kx=1,ky=1)
            self.sig_zf_fn = RectBivariateSpline(self.z,self.f,self.sig_zf,kx=1,ky=1)
            self.iota_fn = RectBivariateSpline(self.z,self.f,self.iota,kx=1,ky=1)
            self.r_fn = RectBivariateSpline(self.z,self.f,self.r,kx=1,ky=1)
            self.thetah_fn = RectBivariateSpline(self.z,self.f,self.thetah,kx=1,ky=1)
            self.theta_fn = RectBivariateSpline(self.z,self.f,self.theta,kx=1,ky=1)
            self.rp_fn = RectBivariateSpline(self.z,self.f,self.rp,kx=1,ky=1)
            self.rph_fn = RectBivariateSpline(self.z,self.f,self.rp_,kx=1,ky=1)
            self.ssq_fn = RectBivariateSpline(self.z,self.f,self.ssq,kx=1,ky=1)
            self.ssf_fn = RectBivariateSpline(self.z,self.f,self.ssf,kx=1,ky=1)
            self.Q_fn =  RectBivariateSpline(self.z,self.f,self.Q,kx=1,ky=1)
            self.Phi_fn = RectBivariateSpline(self.z,self.f,self.Phi,kx=1,ky=1)
            self.mrpke_fn = RectBivariateSpline(self.z,self.f,self.mrpk_e,kx=1,ky=1)
            self.mrpfe_fn = RectBivariateSpline(self.z,self.f,self.mrpf_e,kx=1,ky=1)
            self.AminusIota_fn = RectBivariateSpline(self.z,self.f,self.AminusIota,kx=1,ky=1)
            self.A_fn = RectBivariateSpline(self.z,self.f,self.A,kx=1,ky=1)
            self.pd_fn = RectBivariateSpline(self.z,self.f,self.pd,kx=1,ky=1)
            self.mure_fn = RectBivariateSpline(self.z,self.f,self.mu_re,kx=1,ky=1)
            self.psi_fn = RectBivariateSpline(self.z,self.f,self.psi,kx=1,ky=1)
            
      
    def interpolate_var(self,var):
        return interp2d(self.z,self.f_extended,var, kind='cubic', fill_value = 'extrapolate')
            
    def simulate(self):
        self.interpolate_values()
        self.mu0 = 0.1
        self.s0 = 0.15
        self.k0 = 2
        self.z_sim = np.array(np.tile(self.mu0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.f_sim = np.array(np.tile(self.s0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.k_sim = np.array(np.tile(self.k0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.shock_series = np.array(np.tile(0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.shock2_series = np.array(np.tile(0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.count = 0
        
        start = time.time()
        for n in range(self.params['nsim']):  
            for i in range(1, self.t.shape[0]):
                shock = np.random.normal(0,1)
                shock2 = np.random.normal(0,1)*np.sqrt(1-self.params['corr']**2)
                self.shock_series[i,n] = shock
                self.shock2_series[i,n] = shock2
                self.points = np.array((self.z_sim[i-1],self.f_sim[i-1]))
                self.z_sim[i,n] = self.z_sim[i-1,n] + self.mu_z_fn(self.z_sim[i-1,n],self.f_sim[i-1,n])*self.dt + (self.sig_zk_fn(self.z_sim[i-1,n],self.f_sim[i-1,n]) + self.sig_zf_fn(self.z_sim[i-1,n],self.f_sim[i-1,n])) *shock *np.sqrt(self.dt)
                #below is OU process with stochastic volatility
                self.f_sim[i,n] = self.f_sim[i-1,n] + self.ex.params['pi']* (self.ex.params['f_avg']-   self.f_sim[i-1,n] ) *self.dt + self.beta_f * (self.ex.params['f_u'] - self.f_sim[i-1,n]) * (self.f_sim[i-1,n] - self.ex.params['f_l'])  * shock * np.sqrt(self.dt)
                self.k_sim[i,n] = self.k_sim[i-1,n] + self.k_sim[i-1,n]*(self.Phi_fn(self.z_sim[i,n], self.f_sim[i,n]) - self.params['delta']) * self.dt + self.k_sim[i-1,n] * self.params['sigma'] * shock * np.sqrt(self.dt)
                   
                if self.z_sim[i,n] < 0.001: #handle reflecting boundaries
                    self.z_sim[i,n] = 0.001 + np.abs(self.z_sim[i,n])
                if self.z_sim[i,n] > 1:
                    self.z_sim[i,n] = 2 - self.z_sim[i,n]
                
                if self.f_sim[i,n] < (self.ex.params['f_l']):
                    self.f_sim[i,n] = 2 * (self.ex.params['f_l']) - self.f_sim[i,n]
                if self.f_sim[i,n] >= self.ex.params['f_u']:
                    self.f_sim[i,n] = 2 * self.ex.params['f_u'] - self.f_sim[i,n]
                
            if n%100==0:
                print(n)
        
        end = time.time() - start
        print('total time taken is:{}'.format(end) )
        
        self.z_trim = self.z_sim[int(self.burn_period):,:]
        self.f_trim = self.f_sim[int(self.burn_period):,:]
        self.k_trim = self.k_sim[int(self.burn_period):,:]
        self.shock1_trim = self.shock_series[int(self.burn_period):,:]
        self.shock2_trim = self.shock2_series[int(self.burn_period):,:]
        self.z_trim_ann = np.full([ int(self.z_trim.shape[0]*self.dt),int(self.z_trim.shape[1])],np.nan)
        self.f_trim_ann = np.full([ int(self.f_trim.shape[0]*self.dt),int(self.f_trim.shape[1])],np.nan)
        self.k_trim_ann = np.full([ int(self.k_trim.shape[0]*self.dt),int(self.k_trim.shape[1])],np.nan)
        
        self.shock1_trim_ann = np.full([ int(self.z_trim.shape[0]*self.dt),int(self.z_trim.shape[1])],np.nan)
        self.shock2_trim_ann = np.full([ int(self.z_trim.shape[0]*self.dt),int(self.z_trim.shape[1])],np.nan)
        for j in range(self.z_trim_ann.shape[1]):
            self.z_trim_ann[:,j] = self.z_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
            self.f_trim_ann[:,j] = self.f_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
            self.k_trim_ann[:,j] = self.k_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
            self.shock1_trim_ann[:,j] = self.shock1_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
            self.shock2_trim_ann[:,j] = self.shock2_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
    
        self.z_sim_avg = self.z_trim_ann.reshape(-1).mean(axis=0)
        self.f_sim_avg = self.f_trim_ann.reshape(-1).mean(axis=0)
        
        
        plt.hist(self.z_trim_ann.reshape(-1), bins=100);
        plt.figure()
        plt.hist(self.f_trim_ann.reshape(-1), bins=100);
        
        
        plt.figure()
        sns.distplot(self.z_trim_ann.reshape(-1), hist=False, kde=True);
        plt.axvline(self.z_trim_ann.mean(), color='k', linestyle='dashed', linewidth=1)
        self.crisis_count = [0]
        for j in range(self.params['nsim']):
            temp_count = 0
            for i in range(self.z_trim_ann.shape[0]):
                temp_count = temp_count + self.crisis_flag[np.argwhere(self.z_trim_ann[i,j] < self.z)[0][0],np.argwhere(self.f_trim_ann[i,j] < self.f)[0][0]]   
            self.crisis_count.append(temp_count)
        self.crisis_count = list(map(int, self.crisis_count))
        self.crisis_indicator =  np.full([ int(self.z_trim_ann.shape[0]),int(self.z_trim_ann.shape[1])],np.nan)
        
        #statistics of variables
        for i in range(0,self.params['nsim']):
            for j in range(0, self.crisis_indicator.shape[0]):
                try:
                    temp_crisis = self.crisis_flag[np.argwhere(self.z_trim_ann[j,i] < self.z)[0][0]-1,np.argwhere(self.f_trim_ann[j,i] < self.f)[0][0]-1]   
                except:
                    temp_crisis = self.crisis_flag[self.z.shape[0]-1,np.argwhere(self.f_trim_ann[j,i] < self.f)[0][0]]
                if temp_crisis == 1.0: 
                    self.crisis_indicator[j,i] = 1
        
                
        self.prob = np.nansum(self.crisis_indicator)/self.z_trim_ann.reshape(-1).shape[0]
        print('Probability of staying in crisis region is: {}'.format(self.prob))
        self.crisis_z = self.z_trim_ann[(self.crisis_indicator==1)]
        self.crisis_f = self.f_trim_ann[(self.crisis_indicator==1)]
        
        self.z_sim_avg = self.z_trim_ann.reshape(-1).mean()
        
        self.crisis_indicator_freq =  np.full([ int(self.z_trim.shape[0]),int(self.z_trim.shape[1])],np.nan)
        for i in range(0,self.params['nsim']):
            for j in range(0, self.crisis_indicator_freq.shape[0]):
                try:
                    temp_crisis_freq = self.crisis_flag[np.argwhere(self.z_trim[j,i] < self.z)[0][0]-1, np.argwhere(self.f_trim[j,i] < self.f)[0][0]-1]
                except:
                    temp_crisis_freq = self.crisis_flag[self.z.shape[0] -1, np.argwhere(self.f_trim[j,i] < self.f)[0][0]-1]
                if temp_crisis_freq == 1.0:
                    self.crisis_indicator_freq[j,i]=1
        
        self.crisis_length =  np.array([(k, sum(1 for i in g)) for k,g in groupby(self.crisis_indicator.transpose().reshape(-1))])
        self.crisis_length_mean = np.mean(self.crisis_length[~np.isnan(self.crisis_length[:,0])][:,1])
        
        self.crisis_length_freq =  np.array([(k, sum(1 for i in g)) for k,g in groupby(self.crisis_indicator_freq.transpose().reshape(-1))])
        self.crisis_length_freq_mean = np.mean(self.crisis_length_freq[~np.isnan(self.crisis_length_freq[:,0])][:,1])
        self.crisis_length_freq_data = pd.DataFrame(self.crisis_length_freq[~np.isnan(self.crisis_length_freq[:,0])][:,1])
        
    #statistics of variables
    #compute relevant statistics 
    def compute_statistics_ann_fn(self,fn):
        temp_all = fn(self.z_trim_ann,self.f_trim_ann,grid=False)
        temp_all = np.array(temp_all)
        temp_crisis = temp_all * self.crisis_indicator
        temp_good = temp_all * np.array(pd.DataFrame(1-self.crisis_indicator).fillna(1).replace(0,np.nan))
        return temp_all, temp_crisis, temp_good
    
    def compute_statistics_ann_var(self,temp_all):
            temp_crisis = temp_all * self.crisis_indicator
            temp_good = temp_all * np.array(pd.DataFrame(1-self.crisis_indicator).fillna(1).replace(0,np.nan))
            return temp_crisis, temp_good
    
    def compute_statistics_fn(self,fn):
        temp_all = fn(self.z_trim.transpose().reshape(-1),self.f_trim.transpose().reshape(-1),grid=False).reshape(self.params['nsim'],-1).transpose()
        temp_crisis= (temp_all*self.crisis_indicator_freq)
        temp_good = (temp_all * np.array(pd.DataFrame(1-self.crisis_indicator_freq).fillna(1).replace(0,np.nan)))
        return temp_all, temp_crisis, temp_good    

    def compute_statistics(self):
        
        self.theta_fn_all, self.theta_fn_crisis, self.theta_fn_good = self.compute_statistics_ann_fn(self.theta_fn)
        self.thetah_fn_all, self.thetah_fn_crisis, self.thetah_fn_good = self.compute_statistics_ann_fn(self.thetah_fn)
        self.iota_fn_all, self.iota_fn_crisis, self.iota_fn_good = self.compute_statistics_ann_fn(self.iota_fn)
        self.r_fn_all, self.r_fn_crisis, self.r_fn_good = self.compute_statistics_ann_fn(self.r_fn)
        self.rp_fn_all, self.rp_fn_crisis, self.rp_fn_good = self.compute_statistics_ann_fn(self.rp_fn)
        self.rph_fn_all, self.rph_fn_crisis, self.rph_fn_good = self.compute_statistics_ann_fn(self.rph_fn)
        self.Q_fn_all, self.Q_fn_crisis, self.Q_fn_good = self.compute_statistics_ann_fn(self.Q_fn)
        self.sigka_fn_all, self.sigka_fn_crisis, self.sigka_fn_good = self.compute_statistics_ann_fn(self.ssq_fn)
        self.sigfa_fn_all, self.sigfa_fn_crisis, self.sigfa_fn_good = self.compute_statistics_ann_fn(self.ssf_fn)
        self.mrpke_fn_all, self.mrpke_fn_crisis, self.mrpke_fn_good = self.compute_statistics_ann_fn(self.mrpke_fn)
        self.mrpfe_fn_all, self.mrpfe_fn_crisis, self.mrpfe_fn_good = self.compute_statistics_ann_fn(self.mrpfe_fn)
        self.Phi_fn_all, self.Phi_fn_crisis, self.Phi_fn_good = self.compute_statistics_ann_fn(self.Phi_fn)
        self.AminusIota_fn_all, self.AminusIota_fn_crisis, self.AminusIota_fn_good = self.compute_statistics_ann_fn(self.AminusIota_fn)
        self.A_fn_all, self.A_fn_crisis, self.A_fn_good = self.compute_statistics_ann_fn(self.A_fn)
        self.pd_fn_all, self.pd_fn_crisis, self.pd_fn_good = self.compute_statistics_ann_fn(self.pd_fn)
        self.mure_fn_all, self.mure_fn_crisis, self.mure_fn_good = self.compute_statistics_ann_fn(self.mure_fn)
        self.muz_fn_all, self.muz_fn_crisis, self.muz_fn_good = self.compute_statistics_ann_fn(self.mu_z_fn)
        
        
        self.shock1_fn_all = self.shock1_trim_ann
        self.shock1_fn_crisis, self.shock1_fn_good = self.compute_statistics_ann_var(self.shock1_fn_all)
        self.shock2_fn_all = self.shock2_trim_ann
        self.shock2_fn_crisis, self.shock2_fn_good = self.compute_statistics_ann_var(self.shock2_fn_all)
        self.retQ_fn_all = np.diff(np.log(self.Q_fn_all),axis=0)
        self.retQ_fn_all = np.vstack([self.retQ_fn_all[0,:],self.retQ_fn_all])
        self.retQ_fn_crisis, self.retQ_fn_good = self.compute_statistics_ann_var(self.retQ_fn_all)
        self.output_fn_all = self.A_fn_all * self.k_trim_ann
        self.outputG_fn_all = np.array(np.tile(0,(self.output_fn_all.shape[0],self.params['nsim'])), dtype=np.float64);
        for n in range(self.params['nsim']):
            self.outputG_fn_all[:,n] = np.hstack([np.nan,np.diff(np.log(self.output_fn_all[:,n]))])
        self.outputG_fn_crisis, self.outputG_fn_good = self.outputG_fn_all*self.crisis_indicator, self.outputG_fn_all * np.array(pd.DataFrame(1-self.crisis_indicator).fillna(1).replace(0,np.nan))
        
        vars_corr = ['theta_shock1','theta_Q','theta_mrpke','shock1_mrpke','Q_r', 'retQ_r', 'rp_sigka', 'shock1_sigka', 'mure_r'] #rp_outputG
        self.stats_corr = pd.DataFrame(np.full([len(vars_corr),3],np.nan))
        #correlation statistics
        for i in range(len(vars_corr)):
            temp_corr_all, temp_corr_crisis, temp_corr_good = [],[],[]
            for sim in range(self.params['nsim']):
                temp_corr_all.append(np.corrcoef(eval('self.'+ str(vars_corr[i].split('_')[0]) + '_fn_all').transpose()[sim],\
                                                 eval('self.'+ str(vars_corr[i].split('_')[1]) + '_fn_all').transpose()[sim])[1][0])
                try:
                    temp_corr_crisis.append(np.corrcoef(np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[0]) + '_fn_crisis')[:,sim]).dropna()).transpose(),\
                                                        np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[1]) + '_fn_crisis')[:,sim]).dropna()).transpose())[1][0])
                except:
                    pass
                temp_corr_good.append(np.corrcoef(np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[0]) + '_fn_good')[:,sim]).dropna()).transpose(),\
                                                        np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[1]) + '_fn_good')[:,sim]).dropna()).transpose())[1][0])
            
            self.stats_corr.iloc[i,0] = np.nanmean(temp_corr_all)
            self.stats_corr.iloc[i,1] = np.nanmean(temp_corr_crisis)
            self.stats_corr.iloc[i,2] = np.nanmean(temp_corr_good)
            
        index_temp = []
        for j in range(len(vars_corr)): index_temp.append('Corr(' + str(vars_corr[j].split('_')[0]) + ',' + str(vars_corr[j].split('_')[1]) + ')')
        self.stats_corr.index = index_temp  
        del index_temp
        
        #compute relevant statistics
        vars = ['theta','thetah', 'iota', 'r', 'rp', 'rph', 'Q', 'sigka','sigfa', 'mrpke', 'mrpfe', 'pd', 'mure','outputG', 'retQ']
        self.stats = pd.DataFrame(np.zeros([len(vars)*2,3]))
        for i in range(len(vars)):
            self.stats.iloc[i,0] = round(np.nanmean(eval('self.' + str(vars[i]) + '_fn_all')),3)
            self.stats.iloc[i,1] = round(np.nanmean(eval('self.' + str(vars[i]) + '_fn_crisis')),3)
            self.stats.iloc[i,2] = round(np.nanmean(eval('self.' + str(vars[i]) + '_fn_good')),3)
        for i in range(len(vars)):
            std_temp_all, std_temp_crisis, std_temp_good = [], [], []
            for sim in range(self.params['nsim']):
                std_temp_all.append(np.nanstd(eval('self.' + str(vars[i]) + '_fn_all').transpose()[sim]))
                std_temp_crisis.append(np.nanstd(eval('self.' + str(vars[i]) + '_fn_crisis').transpose()[sim]))
                std_temp_good.append(np.nanstd(eval('self.' + str(vars[i]) + '_fn_good').transpose()[sim]))
            
            self.stats.iloc[i+len(vars),0] = np.nanmean(std_temp_all)
            self.stats.iloc[i+len(vars),1] = np.nanmean(std_temp_crisis)
            self.stats.iloc[i+len(vars),2] = np.nanmean(std_temp_good)
        index_temp = []
        for j in range(len(vars)): index_temp.append('E[' + str(vars[j]) + ']')        
        for j in range(len(vars)): index_temp.append('Std[' + str(vars[j]) + ']')
        self.stats.index = index_temp
        del index_temp
        self.stats = pd.concat((self.stats,self.stats_corr))
        
        self.stats =  pd.concat((self.stats, pd.Series({'No of crisis periods': self.Q_fn_crisis.shape[0], 'freq': int(1/self.dt), 'prob (dt freq)': self.prob,  'gamma':self.ex.params['gamma'],'sigma':self.params['sigma'], 
                                                         'ah':self.params['aH'], 'depr':self.params['delta'], 'kappa':self.params['kappa']}).transpose()), axis=0, ignore_index = False)
        self.stats.columns = ['All','Crisis','Good']
        self.stats = round(self.stats,4)
        

if __name__ == '__main__':
    params={'rho': 0.05, 'aH': 0.02,
            'alpha':0.65, 'kappa':5, 'delta':0.05, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gamma':5,  'corr':0.9,
             'pi' : 0.01, 'f_u' : 0.2, 'f_l' : 0.1, 'f_avg': 0.15,
            'hazard_rate1' :0.065, 'hazard_rate2':0.45,'active':'on','epochE': 3000, 'epochH':2000,
            'Nz':1000,'Nf':30}
    params['beta_f'] = 0.25/params['sigma']
    params['load_pickle'] = True
    params['write_pickle'] = False
    params['scale'] = 2
    params['nsim'] = 30
    sim_ex = simulationExtended(params)
    sim_ex.simulate()
    sim_ex.compute_statistics()
    #sim_ex.write_files()
    print(sim_ex.stats)
    print(sim_ex.crisis_length_mean,sim_ex.crisis_length_freq_mean)
    
    if True:
        try:
            if not os.path.exists('../output/plots'):
                os.mkdir('../output/plots')
        except:
            print('Warning: Cannot create directory')
        
        plt.figure()
        plt.hist(sim_ex.z_trim_ann.reshape(-1),bins=100, density=True);
        plt.grid(False)
        plt.xlabel('Wealth share',fontsize=15)
        plt.savefig('../output/plots/sim_z.png')
        
        plt.figure()
        plt.hist(sim_ex.crisis_z,bins=100,density=True);
        plt.grid(False)
        #plt.title('Extended model')
        plt.xlabel('Wealth share')
        plt.savefig('../output/plots/crisis_z_extended.png')
        
        
        plt.figure()
        plt.hist2d(sim_ex.crisis_z,sim_ex.crisis_f);
        
        plt.figure()
        plt.plot(sim_ex.f_trim[:,0]) 
        plt.ylabel('Productivity of experts',fontsize=15)
        plt.xlabel('Months',fontsize=15)
        plt.savefig('../output/plots/experts_sim.png')
        
        
        sns.set_style({'axes.grid' : False})
        import matplotlib.gridspec as gridspec
        x = sim_ex.crisis_z
        y = sim_ex.crisis_f
        
        fig = plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        ax_main.grid(False)
        ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
        ax_xDist.grid(False)
        ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
        ax_yDist.grid(False)
        
        ax_main.scatter(x,y,marker='.')
        ax_main.set(xlabel="Wealth Share", ylabel="Expert Productivity")
        
        ax_xDist.hist(x,bins=100,align='mid')
        ax_xDist.set(ylabel='count')
        ax_xCumDist = ax_xDist.twinx()
        ax_xCumDist.set_ylabel('Distribution',color='r')
        
        
        ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid')
        ax_yDist.set(xlabel='count')
        ax_yCumDist = ax_yDist.twiny()
        ax_yCumDist.set_xlabel('Distribution',color='r')
        plt.savefig('../output/plots/crisis_z_2d.png')
        plt.show()
        
        print(sim_ex.crisis_length_freq_data.quantile(0.1))
        print(sim_ex.crisis_length_freq_data.quantile(0.5))
        print(sim_ex.crisis_length_freq_data.quantile(0.9))
        print(sim_ex.crisis_length_freq_data.mean())
        
        trim = 400000
        sns.set(font_scale=1.2)    
        df = pd.DataFrame({'Wealth share':sim_ex.z_trim_ann.reshape(-1)[0:trim],'Expert productivity':sim_ex.f_trim_ann.reshape(-1)[0:trim]})
        p = sns.jointplot(x='Wealth share',y='Expert productivity',data=df,kind='kde')
        p.savefig('../output/plots/z_a_jointplot.png')
        
        sns.set(font_scale=1.2)
        df_crisis = pd.DataFrame({'Wealth share':sim_ex.crisis_z,'Expert productivity':sim_ex.crisis_f})
        p_crisis = sns.jointplot(x='Wealth share', y='Expert productivity', data=df_crisis, kind='hex')
        p_crisis.savefig('../output/plots/z_a_jointplot_crisis.png')
        
        
        