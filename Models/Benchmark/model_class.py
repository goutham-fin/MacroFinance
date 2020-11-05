# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:29:38 2019

@author: goutham
"""
import sys
sys.path.insert(0, '../')
from scipy.optimize import fsolve
from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import numpy as np
from Benchmark.finite_difference import hjb_implicit, hjb_explicit, hjb_implicit_upwind, hjb_implicit_policy
import os
from scipy.interpolate import interp1d

'''
Variable names used in model input:
rhoE: discount rate of experts
rhoH: discount rate of households
aE: productivity of experts
aH: productivity of households
alpha: skin-in-the-game constraint
kappa: investment costs 
delta: depreciation rate
gammaE: risk aversion of experts
gammaH: risk aversion of households
IES: Inter-temporal elasticity of substitution
zbar: mean proportion of experts 
lambda_d : death rate
utility: type of utility function
nsim: number of simulations to run
'''

class  model():
    '''
    This class solves the baseline model with CRRA utility from the paper 'Asset Pricing with Realistic Crisis Dynamics (2020)'. 
    '''
    def __init__(self,params):
        self.params = params
        
        self.Nz   = 1000;  #Wealth-share grid
        self.Nf = 1 #used for convenience
        self.Nz_temp = 0; #helper variable 
        self.z = np.linspace(0.001,0.999,self.Nz) #linear grid for wealth share
        self.z_mat = np.tile(self.z,(self.Nf,1)).transpose() 
        self.dz  = self.z_mat[1:self.Nz] - self.z_mat[0:self.Nz-1]; #first difference of grid
        self.dz2 = self.dz[0:self.Nz-2]**2; 
        
        self.q   =  np.array(np.tile(1,(self.Nz,self.Nf)),dtype=np.float64); #capital price
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); #first difference of capital price
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); #second difference of capital price
        
        self.thetah=  np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); #portfolio of households 
        self.theta= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); #portfolio of experts
        self.r = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); #risk free rate
        self.ssq= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); #return volatility
        self.chi= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        self.dz_mat = np.tile(self.dz,(self.Nf,1))
        self.dz2_mat = np.tile(self.dz2,(self.Nf,1))
        self.psi = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64) #capital share of experts
        p = self.params['aE']**(-self.params['gammaH']) * (1-self.z) **(1-self.params['gammaH'])
        p1 = self.params['aE']**(-self.params['gammaE']) * self.z **(1-self.params['gammaE'])
        self.Jh=np.array(np.tile(p,(self.Nf,1)).transpose(), dtype=np.float64) #value function of households
        self.Je=np.array(np.tile(p1,(self.Nf,1)).transpose(), dtype=np.float64) #value function of experts
        self.Jhz= np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64) #first derivative of value function
        self.Jhzz= np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64) #second derivative of value function
        self.Jez = np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64)
        self.Jezz = np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64)

        self.Jhzl= np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64)
        self.Jhzzl= np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64)
        self.Jezl= np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64)
        self.Jezzl= np.array(np.tile(0,(self.Nz,self.Nf)),dtype=np.float64)
        self.crisis_z =0 #crisis boundary
        if not os.path.exists('../output'):
            os.mkdir('../output')

    def equations_region1(self, q_p, Psi_p, sig_ka_p, zi, fi):  
        '''
        Solves for the equilibrium policy in the crisis region 
        Input: old values of capital price(q_p), capital share(Psi_p), return volatility(sig_ka_p), grid point(zi)
        Output: new values from Newton-Rhaphson method
        '''
        i_p = (q_p -1)/self.params['kappa']
        eq1 = (self.params['aE']-self.params['aH'])/q_p  - \
                            self.params['alpha']*(self.Jhzl[zi,fi] - self.Jezl[zi,fi] + 1/(self.z_mat[zi,fi] * (1-self.z_mat[zi,fi])))  * (self.params['alpha']* Psi_p - self.z_mat[zi,fi]) * sig_ka_p**2
                    
        eq2 = (self.z_mat[zi,fi]/self.Je[zi,fi])**(1/self.params['gammaE']) * q_p**(1/self.params['gammaE'])  + \
                ((1-self.z_mat[zi,fi])/self.Jh[zi,fi]) ** (1/self.params['gammaH']) * q_p**(1/self.params['gammaH'])  - \
                                                    Psi_p * (self.params['aE'] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1,fi])/(self.dz[zi-1,fi]*q_p) * self.z_mat[zi-1,fi] *(self.params['alpha']* Psi_p/self.z_mat[zi,fi]-1))) - self.params['sigma'] 
                     
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        
        QN[0,:] = np.array([-(self.Jhzl[zi,fi] - self.Jezl[zi,fi] + 1/(self.z_mat[zi,fi] * (1-self.z_mat[zi,fi]))) * self.params['alpha']**2 * sig_ka_p**2, \
                            -2* self.params['alpha']* (self.params['alpha']* Psi_p - self.z_mat[zi,fi]) * (self.Jhzl[zi,fi] - self.Jezl[zi,fi] + 1/(self.z_mat[zi,fi] * (1-self.z_mat[zi,fi]))) * sig_ka_p, \
                                  -(self.params['aE'] - self.params['aH'])/q_p**2])
        QN[1,:] = np.array([self.params['aH'] - self.params['aE'], 0, (self.z_mat[zi,fi]/self.Je[zi,fi])**(1/self.params['gammaE']) * q_p**((1-self.params['gammaE'])/self.params['gammaE']) * (1/self.params['gammaE']) + ((1-self.z_mat[zi,fi])/self.Jh[zi,fi])**(1/self.params['gammaH']) * q_p**((1-self.params['gammaH'])/self.params['gammaH'])  * (1/self.params['gammaH']) + 1/self.params['kappa']])
        
        QN[2,:] = np.array([-sig_ka_p *self.params['alpha'] * (1- self.q[zi-1,fi]/q_p) / (self.dz[zi-1,fi]) , \
                          1 - (1- (self.q[zi-1,fi]/q_p)) / self.dz[zi-1,fi] * (self.params['alpha']*Psi_p/self.z_mat[zi,fi] -1) * self.z_mat[zi,fi] , \
                            sig_ka_p * (-self.q[zi-1,fi]/(q_p**2 * self.dz[zi-1,fi]) * (self.params['alpha']* Psi_p/self.z_mat[zi,fi] -1) * self.z_mat[zi,fi])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN
    def equations_region_(self,q_p, Psi_p, sig_ka_p, zi, fi):
        '''
        Solves for the equilibrium policies in crisis regime
        '''
        eq1 = np.log(q_p)/self.params['gammaE'] + np.log(self.GS[zi,fi]) - np.log(self.params['aE']*Psi_p + self.params['aH']*(1-Psi_p) - (q_p-1)/self.params['kappa'])
        eq2 = sig_ka_p*(q_p - (q_p - self.q[zi-1,fi])*(self.params['alpha']*Psi_p - self.z_mat[zi,fi])/self.dz[zi-1,fi]) - self.params['sigma']*q_p
        eq3 = self.params['aE'] - self.params['aH'] - q_p*self.params['alpha']*(self.params['alpha']*Psi_p - self.z_mat[zi,fi])*sig_ka_p**2* self.JJlp[zi,fi]

        ER = np.array([eq1,eq2,eq3]);
        
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([1/(q_p*self.params['gammaE']) + 1/((self.params['aE'] - self.params['aH'])*Psi_p + self.params['aH'] - (q_p -1)/self.params['kappa'])/self.params['kappa'],\
                            -(self.params['aE'] - self.params['aH'])/((self.params['aE'] - self.params['aH'])*Psi_p + self.params['aH'] - (q_p -1)/self.params['kappa']),0])
        QN[1,:] = np.array([sig_ka_p*(1-(self.params['alpha'] * Psi_p - self.z_mat[zi,fi])/self.dz[zi-1,fi]) - self.params['sigma'],\
                            -sig_ka_p*(q_p - self.q[zi-1,fi])*self.params['alpha']/self.dz[zi-1,fi], \
                            q_p - (q_p - self.q[zi-1,fi])*(self.params['alpha']*Psi_p - self.z_mat[zi,fi])/self.dz[zi-1,fi]])

        QN[2:] = np.array([-self.params['alpha']*(self.params['alpha']*Psi_p-self.z_mat[zi,fi])*sig_ka_p**2 * self.JJlp[zi,fi],\
                            -q_p* self.params['alpha']**2*sig_ka_p**2 *self.JJlp[zi,fi], \
                            -2*q_p*self.params['alpha']*(self.params['alpha']*Psi_p - self.z_mat[zi,fi])*sig_ka_p*self.JJlp[zi,fi]])
                
        
                
        EN = np.array([q_p,Psi_p,sig_ka_p]) - np.linalg.solve(QN,ER)
        del ER
        del QN
        return [EN[1],EN[2],EN[0]]

    def equations_region2(self, q_p, sig_ka_p, zi, fi):
        '''
        Solves for the equilibrium policies in normal regime
        '''
        i_p = (q_p -1)/self.params['kappa']
        
        
        eq1 = sig_ka_p - self.params['sigma'] / (1-((q_p - self.q[zi-1,fi])/(self.dz[zi-1,fi]*q_p) * self.z_mat[zi-1,fi] *(self.chi[zi,fi]/self.z_mat[zi,fi]-1)))
        
        eq2 = (q_p*self.z_mat[zi,fi]/self.Je[zi,fi])**(1/self.params['gammaE']) + (q_p*(1-self.z_mat[zi,fi])/self.Jh[zi,fi]) ** (1/self.params['gammaH'])   -  (self.params['aE'] - i_p)
        ER = np.array([eq1, eq2])
        QN = np.zeros(shape=(2,2))
        QN[0,:] = np.array([1 - (1- (self.q[zi-1,fi]/q_p)) / self.dz[zi-1,fi] * (self.chi[zi,fi]/self.z_mat[zi,fi] -1) * self.z_mat[zi-1,fi] , \
                            sig_ka_p * (-self.q[zi-1,fi]/(q_p**2 * self.dz[zi-1,fi]) * (self.chi[zi,fi]/self.z_mat[zi,fi] -1) * self.z_mat[zi-1,fi])])
        QN[1,:] = np.array([0,(self.z_mat[zi,fi]/self.Je[zi,fi])**(1/self.params['gammaE']) * q_p**((1-self.params['gammaE'])/self.params['gammaE']) + ((1-self.z_mat[zi,fi])/self.Jh[zi,fi])**(1/self.params['gammaH']) * q_p**((1-self.params['gammaH'])/self.params['gammaH']) + 1/self.params['kappa']])
        EN = np.array([sig_ka_p,q_p]) - np.linalg.solve(QN,ER)
        return EN

    def initialize_(self):
        '''
        Initialize variables at z=0
        '''
        sig_ka_q = self.params['sigma']
        Psi_q =0
        GS_e = (self.z_mat/self.Je)**(1/self.params['gammaE'])
        GS_h = ((1-self.z_mat)/self.Jh)**(1/self.params['gammaH'])
        self.GS = GS_e + GS_h
        qL = 0
        qR = self.params['aH']*self.params['kappa'] +1
        
        for k in range(30):
            q = (qL + qR)/2
            iota = (q-1)/self.params['kappa']
            A0 = self.params['aH'] - iota
            
            if (np.log(q)/self.params['gammaE'] + np.log(self.GS[0])) > np.log(A0):
                qR = q
            else:
                qL = q
            
            self.q[0,:] = q
                    
        return (q, Psi_q, sig_ka_q)

    
    def solve(self):
        '''
        Model solution
        '''
        fi=0 #for convenience (to handle numpy matrices)
        for t in range(40):
            self.chi[:,0] = np.maximum(self.params['alpha'],self.z)
            self.first_time = 0
            self.Jhz[1:self.Nz,:]  = (self.Jh [1:self.Nz,:] - self.Jh [0:self.Nz-1,:])/self.dz_mat; self.Jhz[0,:]=0;
            self.Jhzz[0:self.Nz-2,:] = (self.Jh[2:self.Nz,:] + self.Jh[0:self.Nz-2,:] - 2.*self.Jh[1:self.Nz-1,:])/(self.dz2_mat); self.Jhzz[-2,:]=0; self.Jhzz[-1,:]=0 
            self.Jhzl  = self.Jhz/self.Jh ; 
            self.Jhzzl = self.Jhzz/ self.Jh;
            
            self.Jez[1:self.Nz,:]  = (self.Je [1:self.Nz,:] - self.Je [0:self.Nz-1,:])/self.dz_mat; self.Jez[0,:]=0;
            self.Jezz[0:self.Nz-2,:] = (self.Je[2:self.Nz,:] + self.Je[0:self.Nz-2,:] - 2.*self.Je[1:self.Nz-1,:])/(self.dz2_mat); self.Jezz[-2,:]=0; self.Jezz[-1,:]=0
            self.Jezl  = self.Jez/self.Je ; 
            self.Jezzl = self.Jezz/ self.Je;
            
            self.JJlp = self.Jhzl - self.Jezl + 1/(self.z_mat*(1-self.z_mat))
        

            
            for zi in range(1,self.Nz):
                if zi==1:
                    '''
                    For the first grid point
                    '''
                    q_init, Psi_init, sig_ka_init = self.initialize_()
                    self.q[zi-1,:], self.psi[zi-fi], self.ssq[zi-1,fi]  = q_init, Psi_init, sig_ka_init
                    result = self.equations_region1(q_init, Psi_init, sig_ka_init, zi, fi)
                    self.psi[zi,fi], self.ssq[zi,fi],  self.q[zi,fi] = result[0], result[1], result[2] 
                    del(result)
                             
                else:
                    result= self.equations_region1(self.q[zi-1,fi], self.psi[zi-1,fi], self.ssq[zi-1,fi], zi, fi)
                    if result[0]>=1:                     
                        if self.first_time == 0:
                            self.crisis_z = zi
                            self.first_time = 1
                        result = self.equations_region2(self.q[zi-1,fi],self.ssq[zi-1,fi], zi, fi)
        
                        self.ssq[zi,fi], self.q[zi,fi] =result[0], result[1]
                        self.psi[zi,fi]=1
    
                        del(result)
                                     
                    else:
                        self.psi[zi,fi], self.ssq[zi,fi], self.q[zi,fi] =result[0], result[1], result[2]
                        del(result)

            #value function iteration

            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2.reshape(-1,1)); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            
            self.qzl  = self.qz/self.q ; 
            self.qzzl = self.qzz/ self.q;

            self.crisis_flag = np.array(np.tile(0,(self.Nz,self.Nf)), dtype = np.float64)
            self.crisis_flag[0:self.crisis_z] = 1
            
            
            self.iota = (self.q - 1)/self.params['kappa'] 
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.theta = (self.chi*self.psi)/self.z_mat
            self.thetah = (1-(self.chi*self.psi))/(1-self.z_mat)
            self.theta[0,:] = self.theta[1,:]
                
            self.sig_za =  self.z_mat * (self.theta-1) * self.ssq
            

            self.sig_jha =  self.Jhzl*self.sig_za
            
            self.sig_jea =  self.Jezl*self.sig_za
            self.priceOfRiskE = -self.sig_jea + self.sig_za/self.z_mat + self.ssq + (self.params['gammaE']-1)*self.params['sigma']
            self.priceOfRiskH = -self.sig_jha + self.sig_za/(1-self.z_mat) + self.ssq + (self.params['gammaH']-1)*self.params['sigma']
            
            self.rp = self.priceOfRiskE*self.ssq
            self.rp_ = self.priceOfRiskE*self.ssq
            self.cw_h = (((1-self.z_mat)*self.q) ** ((1-self.params['gammaH']) / self.params['gammaH'])) / self.Jh**(1/self.params['gammaH'])
            self.cw_e = ((self.z_mat*self.q) ** ((1-self.params['gammaE']) / self.params['gammaE'])) / self.Je**(1/self.params['gammaE'])
            self.mu_z = ((self.params['aE'] - self.iota)/self.q - self.cw_e)*self.z_mat + \
                                self.sig_za*(self.priceOfRiskE - self.ssq) + self.z_mat*self.ssq*(self.priceOfRiskE - self.priceOfRiskH)*(1-self.params['alpha']) + (self.params['lambda_d'])*(self.params['zbar']-self.z_mat)
            
            
            self.mu_jh=  self.Jhzl*self.mu_z + 0.5 *self.Jhzzl * (self.sig_za**2) 
            self.mu_je= self.Jezl*self.mu_z+  0.5 *self.Jezzl * (self.sig_za**2) 
            self.mu_q =  self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.rp[0,:] = self.rp[1,:]
            self.rp_[0,:] = self.rp_[1,:]
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.params['aE'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.priceOfRiskE = self.rp/self.ssq
            self.priceOfRiskH = self.rp_/self.ssq
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            self.r[self.crisis_z:self.crisis_z+2] = 0.5*(self.r[self.crisis_z+2] + self.r[self.crisis_z-1]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
            
            self.Ah = self.params['rhoH'] - self.cw_h - (1-self.params['gammaH']) * (self.Phi - self.params['delta']) + 0.5 * self.params['gammaH'] *(1-self.params['gammaH']) * self.params['sigma']**2 - (1-self.params['gammaH']) * self.params['sigma'] * self.sig_jha
            self.Ae = self.params['rhoE'] - self.cw_e - (1-self.params['gammaE']) * (self.Phi - self.params['delta']) + 0.5 * self.params['gammaE'] *(1-self.params['gammaE']) * self.params['sigma']**2 - (1-self.params['gammaE']) * self.params['sigma'] * self.sig_jea
            self.B1 = self.mu_z
            self.C11 = 0.5*self.sig_za**2
            dt =0.8
            
            self.Je[:,0] = hjb_implicit_policy(self.z,self.Ae,self.mu_z,self.sig_za,0,self.Je,dt).reshape(-1)
            self.Jh[:,0] = hjb_implicit_policy(self.z,self.Ah,self.mu_z,self.sig_za,0,self.Jh,dt).reshape(-1)
        
        self.AminusIota = self.psi* (self.params['aE'] - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
        self.A = self.psi*(self.params['aE']) + (1-self.psi) * (self.params['aH'])
        self.pd = self.q / self.AminusIota
        self.kfe()
    
            
    def plots_(self):
        '''
        create plots
        '''
        if not os.path.exists('../output/plots'):
                os.mkdir('../output/plots') 
        plot_path = '../output/plots/'
        f=[0]
        index1 = 0
        index2=  0
        index3 = 0
                 
        vars = ['self.theta','self.thetah','self.psi','self.ssq','self.mu_z','self.sig_za','self.priceOfRiskE','self.priceOfRiskH', 'self.rp']
        labels = ['$\theta_{e}$','$\theta_{h}$','$\psi$','$\sigma_{ka}$','$\mu_z$','$\sigma_{z}$','$\kappa_{e}$', '$\kappa_{h}$', '$(\mu_rE-r)$']
        title = ['Portfolio Choice: Experts', 'Portfolio Choice: Households',\
                         'Capital Share: Experts', 'Price volatility','Drift of wealth share: Experts',\
                         'Volatility of wealth share: Experts', 'Market price of risk: Experts',\
                         'Market price of risk: Households', 'Risk premia: Experts']
        for i in range(len(vars)):
                plt.plot(self.z[1:],eval(vars[i])[1:,index1],label='S={i}'.format(i=str(f[0])))
                plt.plot(self.z[1:],eval(vars[i])[1:,int(index2)],label='S={i}'.format(i= str(round(f[int(index2)]))))
                plt.plot(self.z[1:],eval(vars[i])[1:,int(index3)],label='S={i}'.format(i= str(round(f[int(index3)],2))),color='b') 
                plt.grid(True)
                plt.legend(loc=0)
                plt.axis('tight')
                plt.xlabel('Wealth share (z)')
                plt.ylabel('r'+labels[i])
                plt.title(title[i])
                plt.savefig(plot_path + str(vars[i]).replace('self.','') + '_benchmark.png')
                plt.figure()
        plt.figure()
        plt.plot(self.z[10:-20], self.f[10:-20])
        plt.figure()
        plt.plot(self.z, self.pd)

    def kfe(self):
        '''
        compute stationary distribution of wealth share
        '''
        self.coeff = 2*self.mu_z[1:-1]/(self.sig_za[1:-1]**2)
        self.coeff_fn = interp1d(self.z[1:-1],self.coeff.reshape(-1),kind='nearest',fill_value='extrapolate')
        Nh = 10000
        self.tv, self.dist = self.forward_euler(self.coeff_fn,1,0,1,Nh);
        #convert from dist to feta
        self.dist_fn = interp1d(self.tv, self.dist, kind = 'linear', fill_value = 'extrapolate')
        self.f = np.full([self.Nz,1],np.nan)
        for i in range(1,self.Nz):
            self.f[i]= self.dist_fn(self.z[i])/self.sig_za[i]**2
        
        h=1/Nh
        area = np.abs(np.nansum(h*self.f[1:]))
        self.f_norm = self.f/area

    def forward_euler(self, fun, y0, t0, tf, Nh):
        u0 = y0; h=(tf-t0)/Nh;
        uv = [u0];
        tv = [t0]; 
        tn=[];
        
        for n in range(Nh-1):
            tn = t0 + n*h; 
            unew = uv[n] + h*(fun(tn)*uv[n]);
            uv.append(unew);
            tv.append(tn)
    
        return tv, uv
    
if __name__ == '__main__':
    params={'rhoE': 0.06, 'rhoH': 0.03, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.5, 'kappa':7, 'delta':0.025, 'zbar':0.1, 
            'lambda_d':0, 'sigma':0.06, 'gammaE':2, 'gammaH':2}
    m = model(params)
    m.solve()
    m.plots_()
    
    