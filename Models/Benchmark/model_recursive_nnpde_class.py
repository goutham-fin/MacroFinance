import sys
sys.path.insert(0, '../')
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d
from Benchmark.nnpde import nnpde_informed
import dill
import os
os.system('clear')

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
scale: 
'''

class model_recursive_nnpde():
    def __init__(self, params):
        self.params = params
        # algorithm parameters
        self.maxIterations = self.params['maxIterations']; 
        self.convergenceCriterion = 1e-2; 
        self.dt = 1; #time step width
        self.converged = 'False'
        self.Iter=0
        # grid parameters
        self.Nf = 1
        self.grid_method = 'uniform' #specify 'uniform' or 'non-uniform'
        self.Nz = 1000;
        zMin = 0.001; 
        zMax = 0.999;
       
        # grid parameters
        self.Nz = 1000;
        zMin = 0.001; 
        zMax = 0.999;
        
        if self.grid_method == 'non-uniform':
            auxGrid = np.linspace(0,1,self.Nz);
            auxGridNonuniform = 3*auxGrid ** 2  - 2*auxGrid **3; #nonuniform grid from 0 to 1
            self.z = zMin + [zMax - zMin]*auxGridNonuniform; #nonuniform grid from zMin to zMax
        if self.grid_method == 'uniform':
            self.z = np.linspace(zMin,zMax,self.Nz);
        
        self.z_mat = np.tile(self.z,(1,1)).transpose()
        self.dz  = self.z_mat[1:self.Nz] - self.z_mat[0:self.Nz-1]; 
        self.dz2 = self.dz[0:self.Nz-2]**2
        self.dz_mat = np.tile(self.dz,(self.Nf,1))

        ## initial guesses [terminal conditions]
        self.Je = np.ones([self.Nz]) * 1
        self.Jh = np.ones([self.Nz]) * 1
        self.q = np.ones([self.Nz,1]);
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        # allocate memory for other variables
        self.psi = np.full([self.Nz,1],np.NaN)
        self.chi = np.full([self.Nz,1],np.NaN)
        self.ssq = np.full([self.Nz,1],np.NaN)
        self.iota = np.full([self.Nz,1],np.NaN)
        self.dq = np.full([self.Nz,1],np.NaN)
        self.amax_vec = []
    
    
    def equations_region1(self, q_p, Psi_p, sig_ka_p, zi):  
        '''
        Solves for the equilibrium policy in the crisis region 
        Input: old values of capital price(q_p), capital share(Psi_p), return volatility(sig_ka_p), grid point(zi)
        Output: new values from Newton-Rhaphson method
        ''' 
        dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        i_p = (q_p -1)/self.params['kappa']
        eq1 = (self.params['aE']-self.params['aH'])/q_p  - \
                            self.params['alpha']* (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * (self.params['alpha']* Psi_p - self.z[zi]) * sig_ka_p**2 - (self.params['gammaE'] - self.params['gammaH']) * sig_ka_p * self.params['sigma']
                    
        eq2 = (self.params['rhoE']*self.z[zi] + self.params['rhoH']*(1-self.z[zi])) * q_p  - Psi_p * (self.params['aE'] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1][0])/(dz[zi-1]*q_p) * self.z[zi-1] *(self.params['alpha']* Psi_p/self.z[zi]-1)))  - self.params['sigma'] 
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([-self.params['alpha']**2 * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p**2, \
                            -2*self.params['alpha']*(self.params['alpha'] * Psi_p - self.z[zi]) * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p + (self.params['gammaH'] - self.params['gammaE']) * self.params['sigma'], \
                                  -(self.params['aE'] - self.params['aH'])/q_p**2])
        QN[1,:] = np.array([self.params['aH'] - self.params['aE'], 0, self.z[zi]* self.params['rhoE'] + (1-self.z[zi])* self.params['rhoH'] + 1/self.params['kappa']])
        
        QN[2,:] = np.array([-sig_ka_p * self.params['alpha'] * (1- self.q[zi-1][0]/q_p) / (dz[zi-1]) , \
                          1 - (1- (self.q[zi-1][0]/q_p)) / dz[zi-1] * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1] , \
                            sig_ka_p * (-self.q[zi-1][0]/(q_p**2 * dz[zi-1]) * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN
    def equations_region1_scaled(self, q_p, Psi_p, sig_ka_p, zi):  
        '''
        Solves for the equilibrium policy in the crisis region 
        Input: old values of capital price(q_p), capital share(Psi_p), return volatility(sig_ka_p), grid point(zi)
        Output: new values from Newton-Rhaphson method
        ''' 
        dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        i_p = (q_p -1)/self.params['kappa']
        eq1 = (self.params['aE']-self.params['aH'])/q_p  - \
                            self.params['alpha']* (self.dLogJh[zi]*(1-self.params['gammaH']) - self.dLogJe[zi]*(1-self.params['gammaE']) + 1/(self.z[zi] * (1-self.z[zi]))) * (self.params['alpha']* Psi_p - self.z[zi]) * sig_ka_p**2 - (self.params['gammaE'] - self.params['gammaH']) * sig_ka_p * self.params['sigma']
                    
        eq2 = (self.params['rhoE']*self.z[zi] + self.params['rhoH']*(1-self.z[zi])) * q_p  - Psi_p * (self.params['aE'] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1][0])/(dz[zi-1]*q_p) * self.z[zi-1] *(self.params['alpha']* Psi_p/self.z[zi]-1)))  - self.params['sigma'] 
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([-self.params['alpha']**2 * (self.dLogJh[zi]*(1-self.params['gammaH']) - self.dLogJe[zi]*(1-self.params['gammaE']) + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p**2, \
                            -2*self.params['alpha']*(self.params['alpha'] * Psi_p - self.z[zi]) * (self.dLogJh[zi]*(1-self.params['gammaH']) - self.dLogJe[zi]*(1-self.params['gammaE']) + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p + (self.params['gammaH'] - self.params['gammaE']) * self.params['sigma'], \
                                  -(self.params['aE'] - self.params['aH'])/q_p**2])
        QN[1,:] = np.array([self.params['aH'] - self.params['aE'], 0, self.z[zi]* self.params['rhoE'] + (1-self.z[zi])* self.params['rhoH'] + 1/self.params['kappa']])
        
        QN[2,:] = np.array([-sig_ka_p * self.params['alpha'] * (1- self.q[zi-1][0]/q_p) / (dz[zi-1]) , \
                          1 - (1- (self.q[zi-1][0]/q_p)) / dz[zi-1] * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1] , \
                            sig_ka_p * (-self.q[zi-1][0]/(q_p**2 * dz[zi-1]) * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN



    def solve(self):
        # initialize variables at z=0
        self.psi[0] = 0;
        self.q[0] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[0]*(self.params['aE']-self.params['aH'])))/(1 + self.params['kappa']*(self.params['rhoH'] + self.z[0] * (self.params['rhoE'] - self.params['rhoH'])));
        self.chi[0] = 0;
        self.ssq[0] = self.params['sigma'];
        self.q0 = (1 + self.params['kappa'] * self.params['aH'])/(1 + self.params['kappa'] * self.params['rhoH']); #heoretical limit at z=0 [just used in a special case below that is probably never entered]
        self.iota[0] = (self.q0-1)/self.params['kappa']
        
        for timeStep in range(self.maxIterations):
            self.Iter+=1
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe = np.hstack([(self.logValueE[1]-self.logValueE[0])/(self.z[1]-self.z[0]),(self.logValueE[2:]-self.logValueE[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueE[-1]-self.logValueE[-2])/(self.z[-1]-self.z[-2])]);
            self.dLogJh = np.hstack([(self.logValueH[1]-self.logValueH[0])/(self.z[1]-self.z[0]),(self.logValueH[2:]-self.logValueH[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueH[-1]-self.logValueH[-2])/(self.z[-1]-self.z[-2])]);
          	
            for i in range(1,self.Nz):
                  if self.psi[i-1] >= 1:
                	      break; 
                  if self.params['scale']>1: result= self.equations_region1_scaled(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                  else: result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                  self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                  self.chi[i] = self.params['alpha']*self.psi[i]
                  self.dq[i] = (1 - self.params['sigma']/self.ssq[i])/(self.chi[i] - self.z[i])*self.q[i]
                  self.iota[i] = (self.q[i]-1)/self.params['kappa']
            self.thresholdIndex = i-1;
            self.crisis_eta = self.z[self.thresholdIndex]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,1)), dtype = np.float64)
            self.crisis_flag[0:self.thresholdIndex] = 1
            self.psi[self.thresholdIndex:] = 1;
            self.q[self.thresholdIndex:] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[self.thresholdIndex:]*(self.params['aE']-self.params['aH']))).reshape(-1,1)/(1 + self.params['kappa']*(self.params['rhoH'] + self.z[self.thresholdIndex:]*(self.params['rhoE']-self.params['rhoH']))).reshape(-1,1);
            self.chi[self.thresholdIndex:] = np.maximum(self.z[self.thresholdIndex:],self.params['alpha']).reshape(-1,1); #NOTE: this seems incorrect for gammaE~=gammaH!
            #self.iota[self.thresholdIndex:] = 1 + self.params['kappa']* self.q[self.thresholdIndex:];
            self.iota[self.thresholdIndex:] = (self.q[self.thresholdIndex:]-1)/self.params['kappa']
            if self.thresholdIndex==0:
                self.dq[self.thresholdIndex:-1] = (self.q[1:] - np.vstack([self.q0,self.q[0:-2]])) / (self.z - np.vstack([0,self.z[:-2]])) #needs fixing
            else:
                self.dq[self.thresholdIndex:] = (self.q[self.thresholdIndex:]- self.q[self.thresholdIndex-1:-1]).reshape(-1,1)/(self.z[self.thresholdIndex:]-self.z[self.thresholdIndex-1:-1]).reshape(-1,1);
            self.ssq[self.thresholdIndex:] = self.params['sigma']/(1-self.dq[self.thresholdIndex:]/self.q[self.thresholdIndex:] * (self.chi[self.thresholdIndex:]-self.z[self.thresholdIndex:].reshape(-1,1)));
            self.theta = (self.chi)/self.z.reshape(-1,1)
            self.thetah = (1-self.chi)/(1-self.z.reshape(-1,1))
            self.theta[0] = self.theta[1]
            self.thetah[0] = self.thetah[1]
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2.reshape(-1,1)); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qzl  = self.qz/self.q ; 
            self.qzzl = self.qzz/ self.q;
            self.consWealthRatioE = self.params['rhoE'];
            self.consWealthRatioH = self.params['rhoH'];
            self.sig_za = (self.chi - self.z.reshape(-1,1))*self.ssq; #sig_za := \sigma^\z \z, similary mu_z
            if self.params['scale']>1:
                self.priceOfRiskE = (1/self.z.reshape(-1,1) - self.dLogJe.reshape(-1,1)*(1-self.params['gammaE'])) * self.sig_za + self.ssq + (self.params['gammaE']-1) * self.params['sigma'];
                self.priceOfRiskH = -(1/(1-self.z.reshape(-1,1)) + self.dLogJh.reshape(-1,1)*(1-self.params['gammaH']))*self.sig_za + self.ssq + (self.params['gammaH']-1) * self.params['sigma'];
            else:
                self.priceOfRiskE = (1/self.z.reshape(-1,1) - self.dLogJe.reshape(-1,1)) * self.sig_za + self.ssq + (self.params['gammaE']-1) * self.params['sigma'];
                self.priceOfRiskH = -(1/(1-self.z.reshape(-1,1)) + self.dLogJh.reshape(-1,1))*self.sig_za + self.ssq + (self.params['gammaH']-1) * self.params['sigma'];
            self.sig_je = self.dLogJe.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.sig_jh = self.dLogJh.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.rp = self.priceOfRiskE * self.ssq
            self.rp_ = self.priceOfRiskH * self.ssq
            self.mu_z = self.z_mat*((self.params['aE']-self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*self.ssq*(self.rp/self.ssq - self.ssq) + self.ssq*(1-self.params['alpha'])*(self.rp/self.ssq - self.rp_/self.ssq) + (self.params['lambda_d']/self.z_mat)*(self.params['zbar']-self.z_mat))
            self.growthRate = np.log(self.q)/self.params['kappa']-self.params['delta'];
            self.sig_za[0] = 0; 
            self.mu_z[0] = 0
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.mu_q = self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.params['aE'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            #fix numerical issues
            self.r[self.thresholdIndex:self.thresholdIndex+2] = 0.5*(self.r[self.thresholdIndex+2] + self.r[self.thresholdIndex-1]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
            if self.grid_method =='non-uniform':
                self.r[0:10] = self.r[10];     
                self.mu_q[0:10] = self.mu_q[10];  
                self.mu_rH[0:10] = self.mu_rH[10]; 
                self.mu_rE[0:10] = self.mu_rE[10];    
                self.priceOfRiskE[0:10] = self.priceOfRiskE[10]; 
                self.priceOfRiskH[0:10] = self.priceOfRiskH[10]; 
                self.ssq[0:10] = self.ssq[10]; 
                self.rp[0:10] = self.rp[10]; 
                self.rp_[0:10] = self.rp_[10]; 
                self.mu_z[0:10] = self.mu_z[10]
            if False:
                self.r[0:10] = self.r[10]; 
                self.mu_q[0:10] = self.mu_q[10];  
                self.mu_rH[0:10] = self.mu_rH[10]; 
                self.mu_rE[0:10] = self.mu_rE[10];    
                self.priceOfRiskE[0] = self.priceOfRiskE[1]; 
                self.priceOfRiskH[0] = self.priceOfRiskH[1]; 
                self.ssq[0] = self.ssq[1]; 
                self.rp[0] = self.rp[1]; 
                self.rp_[0] = self.rp_[1];
            
            #########################################################################################
            #########################################################################################
            #########################################################################################
            # PDE time steps
            # (a) common terms in both PDEs and neural network architecture
            tb = np.vstack((0,self.dt)).astype(np.float32)
            layers=[]
            layers.append(2)
            for i in range(4): layers.append(30)
            layers.append(1)
            learning_rate = 0.001
            
            X = np.vstack((self.z,np.full(self.z.shape[0],self.dt))).transpose().astype(np.float32)
            X_f = np.vstack((self.z,np.random.uniform(0,self.dt,self.z.shape[0]))).transpose().astype(np.float32)
            x_star = np.vstack((self.z,np.full(self.z.shape[0],0))).transpose()
            X_f_plot = X_f.copy()
            Jhat_e0 = self.Je.copy().reshape(-1,1)
            Jhat_h0 = self.Jh.copy().reshape(-1,1)
            self.diffusion = self.sig_za**2/2;
            if self.params['scale']>1:                
                self.linearTermE =  (1-self.params['gammaE'])*self.params['sigma']*self.sig_je + 0.5*self.params['gammaE']*(self.sig_je**2 + self.params['sigma']**2)- self.growthRate - self.params['rhoE']*(np.log(self.params['rhoE']) - np.log(self.Je.reshape(-1,1)) + np.log(self.z.reshape(-1,1)*self.q)) #PDE coefficient multiplying Je
                self.linearTermH = (1-self.params['gammaH'])*self.params['sigma']*self.sig_jh + 0.5*self.params['gammaH']*(self.sig_jh**2 + self.params['sigma']**2)- self.growthRate - self.params['rhoH']*(np.log(self.params['rhoH']) - np.log(self.Jh.reshape(-1,1)) + np.log((1-self.z.reshape(-1,1))*self.q)) #PDE coefficient multiplying Jh
                self.linearTermE = -self.linearTermE
                self.linearTermH = -self.linearTermH
            else:
                self.linearTermE = (1-self.params['gammaE'])*(self.growthRate.reshape(-1,1) + self.params['rhoE']*(np.log(self.params['rhoE'])+np.log(self.z.reshape(-1,1)*self.q)) - self.params['gammaE']/2*self.params['sigma']**2) - self.params['rhoE']* np.log(self.Je.reshape(-1,1)); #PDE coefficient multiplying Je
                self.linearTermH = (1-self.params['gammaH'])*(self.growthRate.reshape(-1,1) + self.params['rhoH']*(np.log(self.params['rhoH'])+np.log((1-self.z.reshape(-1,1))*self.q)) - self.params['gammaH']/2*self.params['sigma']**2) -  self.params['rhoH']* np.log(self.Jh.reshape(-1,1)); #PDE coefficient multiplying Jh
            self.advectionE = self.mu_z 
            self.advectionH = self.mu_z  
            
            #active learning
            X_,X_f_ = X.copy(),X_f.copy()
            Jhat_e0_,Jhat_h0_ = Jhat_e0.copy(), Jhat_h0.copy()
            def add_crisis_points(vector):
                    new_vector = vector.copy()
                    new_vector = np.vstack((new_vector,vector[max(0,self.thresholdIndex-50) : min(self.thresholdIndex+50,self.Nz)]))
                    return new_vector
            
            if self.params['active']=='on':
                X_,X_f_,Jhat_e0_,Jhat_h0_ = add_crisis_points(X),add_crisis_points(X_f),add_crisis_points(Jhat_e0),add_crisis_points(Jhat_h0)
                diffusion, advectionE, advectionH = add_crisis_points(self.diffusion.reshape(-1,1)),add_crisis_points(self.advectionE.reshape(-1,1)),add_crisis_points(self.advectionH.reshape(-1,1))
                linearTermE, linearTermH = add_crisis_points(self.linearTermE.reshape(-1,1)),add_crisis_points(self.linearTermH.reshape(-1,1))
                crisisPointsLength = X_.shape[0]-X.shape[0]
            else:
                diffusion, advectionE, advectionH = self.diffusion.reshape(-1,1), self.advectionE.reshape(-1,1), self.advectionH.reshape(-1,1)
                linearTermE,linearTermH = self.linearTermE.reshape(-1,1), self.linearTermH.reshape(-1,1)
            idx1 = np.random.choice(X_.shape[0], 200, replace=False)
            if self.params['active'] == 'on':
                idx2 = np.random.choice(np.arange(X_.shape[0]-crisisPointsLength,X_.shape[0]),100,replace=True)
                idx = np.hstack((idx1,idx2))
            else:
                idx2 = np.random.choice(X_.shape[0],100,replace=False)
                idx = np.hstack((idx1,idx2))
            #idx = np.arange(0,X.shape[0])
            X_, X_f_, Jhat_e0_, Jhat_h0_ = X_[idx], X_f_[idx], Jhat_e0_[idx], Jhat_h0_[idx]
            linearTermE_tile,linearTermH_tile = linearTermE[idx], linearTermH[idx]
            advectionE_tile, advectionH_tile, diffusion_tile = advectionE[idx], advectionH[idx], diffusion[idx]
            
            #solve the PDE
            model_E = nnpde_informed(-linearTermE_tile,advectionE_tile,diffusion_tile,Jhat_e0_.reshape(-1,1).astype(np.float32),X_,layers,X_f_,self.dt,tb,learning_rate,self.params['nEpochs'])
            model_E.train()
            newJe = model_E.predict(x_star)
            model_E.sess.close()
            del model_E

            model_H = nnpde_informed(-linearTermH_tile,advectionH_tile,diffusion_tile,Jhat_h0_.reshape(-1,1).astype(np.float32),X_,layers,X_f_,self.dt,tb,learning_rate,self.params['nEpochs'])
            model_H.train()
            newJh = model_H.predict(x_star)
            model_H.sess.close()
            del model_H
                  
            #check convergence
            cutoff=20
            
            self.relChangeJe = np.abs((newJe[cutoff:-cutoff,:].reshape(-1)-self.Je[cutoff:-cutoff].reshape(-1)) / self.Je[cutoff:-cutoff].reshape(-1));
            self.relChangeJh = np.abs((newJh[cutoff:-cutoff,:].reshape(-1)-self.Jh[cutoff:-cutoff].reshape(-1)) / self.Jh[cutoff:-cutoff].reshape(-1))
            self.ChangeJe = np.abs(newJe.reshape(-1)- self.Je.reshape(-1))
            self.ChangeJh = np.abs(newJh.reshape(-1)- self.Jh.reshape(-1))
            self.Je = newJe.reshape(-1);
            self.Jh = newJh.reshape(-1);
            self.A = self.psi*(self.params['aE']) + (1-self.psi) * (self.params['aH'])
            self.AminusIota = self.psi*(self.params['aE'] - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
            self.pd = np.log(self.q / self.AminusIota)
            print('Iteration number: ',self.Iter)
            
            if self.params['scale']>1:
                self.amax = np.maximum(np.amax(self.ChangeJe),np.amax(self.ChangeJh))
            else:
                self.amax = np.maximum(np.amax(self.relChangeJe),np.amax(self.relChangeJh))
            self.amax_vec.append(self.amax)
            
            def plot_grid(data,name):
                fix,ax = plt.subplots()
                mypoints=[]
                mypoints.append([data[:,0],data[:,1]])
                data=list(zip(*mypoints))
                plt.scatter(data[0],data[1])
                plt.xlabel('Wealth share (z)',fontsize=15)
                plt.ylabel('Time (t)',fontsize=15)
                plt.ylim(0,1)
                plt.xlim(0,1)
                plt.title(name,fontsize=20)
                plt.savefig(plot_path + str(name) + '.png')
            if self.Iter==1:
                plt.style.use('classic')
                if not os.path.exists('../output/plots'):
                    os.mkdir('../output/plots/')
                plot_path = '../output/plots/'
                plot_grid(X_f_plot,'Full grid')
                plot_grid(X_f,'Training sample')
                plt.style.use('seaborn')
            del X,X_f,x_star,tb
            if  self.amax < self.convergenceCriterion:
                self.converged = 'True'
                break
            print('Absolute error: ',self.amax )
        if self.converged == 'True':
            print('Algortihm converged after {} time steps.\n'.format(timeStep));
        else:
            print('Algorithm terminated without convergence after {} time steps.'.format(timeStep));

if __name__ == '__main__':
    params={'rhoE': 0.05, 'rhoH': 0.05, 'aE': 0.15, 'aH': 0.02,
            'alpha':0.5, 'kappa':5, 'delta':0.05, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':5, 'gammaH':5,'maxIterations':400,'nEpochs':2000}
    params['scale'] = 2
    params['active'] = 'off'
    model1 = model_recursive_nnpde(params)
    model1.maxIterations=40
    model1.solve()
    
    if False:
        def pickle_stuff(object_name,filename):
            with open(filename,'wb') as f:
                dill.dump(object_name,f)
        pickle_stuff(model1,'model_recursive_nnpde_passive'+'.pkl')
        
    
    