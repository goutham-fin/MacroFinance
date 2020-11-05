import sys
sys.path.insert(0, '../')
from scipy.optimize import fsolve
import numpy as np
from Benchmark.finite_difference import hjb_implicit_upwind
import matplotlib.pyplot as plt
from scipy import optimize
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

class model_recursive():
    '''
    This class solves the baseline model with recursive utility (IES=1) from the paper 
    'Confronting Macro-finance model with Data (2020). 
    Thanks to Sebastian Merkel for MATLAB version of parts of code. 
    '''
    def __init__(self, params):
        self.params = params
        
        # algorithm parameters
        self.maxIterations = self.params['maxIterations']; 
        self.convergenceCriterion = 1e-4; 
        self.dt = 0.9; #time step width
        self.converged = 'False'
        self.amax_vec=[]
        # grid parameters
        self.Nf = 1
        self.grid_method = 'uniform' #specify 'uniform' or 'non-uniform'
        self.Nz = 1000;
        zMin = 0.001; 
        zMax = 0.999;
        self.Iter=0
        
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
        self.psi = np.full([self.Nz,1],np.NaN) #capital share of experts
        self.chi = np.full([self.Nz,1],np.NaN) #skin-in-the game constraint
        self.ssq = np.full([self.Nz,1],np.NaN) #return volatility
        self.iota = np.full([self.Nz,1],np.NaN) #investment rate
        self.dq = np.full([self.Nz,1],np.NaN)
    
    
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
        
        ## main iteration
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
                	      break; #break out if normal regime is reached
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
            self.chi[self.thresholdIndex:] = np.maximum(self.z[self.thresholdIndex:],self.params['alpha']).reshape(-1,1); 
            
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
            self.sig_za = (self.chi - self.z.reshape(-1,1))*self.ssq; 
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
            self.mu_z[0] = np.maximum(self.mu_z[0],0); 
            self.mu_z[-1] = np.minimum(self.mu_z[-1],0); 
            
            
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.mu_q = self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.params['aE'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            self.r[self.thresholdIndex:self.thresholdIndex+2] = 0.5*(self.r[self.thresholdIndex+2] + self.r[self.thresholdIndex-1]) 
            self.A = self.psi*(self.params['aE']) + (1-self.psi) * (self.params['aH'])
            self.AminusIota = self.psi*(self.params['aE'] - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
            self.pd = self.q / self.AminusIota
        
            #fix numerical issues
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
            else:
                self.r[0:10] = self.r[10]; 
                self.mu_q[0:10] = self.mu_q[10];  
                self.mu_rH[0:10] = self.mu_rH[10]; 
                self.mu_rE[0:10] = self.mu_rE[10];    
                self.priceOfRiskE[0] = self.priceOfRiskE[1]; 
                self.priceOfRiskH[0] = self.priceOfRiskH[1]; 
                self.ssq[0] = self.ssq[1]; 
                self.rp[0] = self.rp[1]; 
                self.rp_[0] = self.rp_[1];
            
            # Solve the PDEs
            
            self.diffusion = self.sig_za**2/2;
            if self.params['scale']>1:                
                self.linearTermE = 0.5*self.params['gammaE']*(self.sig_je**2 + self.params['sigma']**2)- self.growthRate - self.params['rhoE']*(np.log(self.params['rhoE']) - np.log(self.Je.reshape(-1,1)) + np.log(self.z.reshape(-1,1)*self.q)) #PDE coefficient multiplying Je
                self.linearTermH = 0.5*self.params['gammaH']*(self.sig_jh**2 + self.params['sigma']**2)- self.growthRate - self.params['rhoH']*(np.log(self.params['rhoH']) - np.log(self.Jh.reshape(-1,1)) + np.log((1-self.z.reshape(-1,1))*self.q)) #PDE coefficient multiplying Jh
                self.linearTermE = -self.linearTermE
                self.linearTermH = -self.linearTermH
            else:
                self.linearTermE = (1-self.params['gammaE'])*(self.growthRate.reshape(-1,1) + self.params['rhoE']*(np.log(self.params['rhoE'])+np.log(self.z.reshape(-1,1)*self.q)) - self.params['gammaE']/2*self.params['sigma']**2) - self.params['rhoE']* np.log(self.Je.reshape(-1,1)); #PDE coefficient multiplying Je
                self.linearTermH = (1-self.params['gammaH'])*(self.growthRate.reshape(-1,1) + self.params['rhoH']*(np.log(self.params['rhoH'])+np.log((1-self.z.reshape(-1,1))*self.q)) - self.params['gammaH']/2*self.params['sigma']**2) -  self.params['rhoH']* np.log(self.Jh.reshape(-1,1)); #PDE coefficient multiplying Jh
            self.advectionE = self.mu_z +  (1-self.params['gammaE'])*self.params['sigma']*self.sig_za
            self.advectionH = self.mu_z + (1-self.params['gammaH'])*self.params['sigma']*self.sig_za;
            
            # solving PDE using implicit scheme
            newValueAux = hjb_implicit_upwind(np.vstack([0,self.z.reshape(-1,1),1]),np.vstack([np.NaN,self.Je.reshape(-1,1),np.NaN]),self.dt, \
                                                        np.vstack([0,self.diffusion,0]),np.vstack([0,self.advectionE,0]),np.vstack([0,self.linearTermE,0]),np.vstack([0,np.zeros([self.z.shape[0],1]),0]), 0, 0); #add fake grid extensions, because solver ignores boundary coefficients
            newJe = newValueAux[1:-1]; 
            newValueAux = hjb_implicit_upwind(np.vstack([0,self.z.reshape(-1,1),1]),np.vstack([np.NaN,self.Jh.reshape(-1,1),np.NaN]),self.dt, \
                                                         np.vstack([0,self.diffusion,0]),np.vstack([0,self.advectionH,0]),np.vstack([0,self.linearTermH,0]),np.vstack([0,np.zeros([self.z.shape[0],1]),0]),0,0); #add fake grid extensions, because solver ignores boundary coefficients
            newJh = newValueAux[1:-1]; 
                  
            # check convergence
            self.relChangeJe = np.abs((newJe-self.Je)/newJe)
            self.relChangeJh = np.abs((newJh-self.Jh)/newJh)
            self.Je = newJe;
            self.Jh = newJh;
            self.amax = np.maximum(np.amax(self.relChangeJe.reshape(-1)),np.amax(self.relChangeJh.reshape(-1)))
            self.amax_vec.append(self.amax)
            if self.amax < self.convergenceCriterion:
                self.converged = 'True'
                break
            print('Iteration number and absolute max of error:', self.Iter, self.amax)
        if self.converged == 'True':
            print('Algortihm converged after {} time steps.\n'.format(timeStep));
        else:
            print('Algorithm terminated without convergence after {} time steps.'.format(timeStep));
        
        self.kfe()
        
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
    params={'rhoE': 0.05, 'rhoH': 0.05, 'aE': 0.15, 'aH': 0.02,
            'alpha':0.5, 'kappa':5, 'delta':0.05, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':5, 'gammaH':5,'maxIterations':400}
    params['scale']=2
    model_recur = model_recursive(params)
    model_recur.maxIterations=150
    model_recur.solve()  
    