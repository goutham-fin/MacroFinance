import sys
sys.path.insert(0, '../')
from scipy.optimize import fsolve
import numpy as np
from Benchmark.finite_difference import hjb_implicit_upwind, hjb_implicit_policy
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")
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
class model_recursive_general():
    '''
    This class solves the baseline model with recursive utility and IES!=1 from the paper 
    'Asset Pricing with Realistic Crisis Dynamics (2020)'
    '''
    def __init__(self, params):
        self.params = params    
        # algorithm parameters
        self.continueOldIteration = 'False'; #if true, does not reset initial guesses
        self.maxIterations = 50; #always stop after this many iterations
        self.convergenceCriterion = 1e-6; #stop earlier, if relative change is smaller than this
        self.dt = 0.4; #time step width
        self.converged = 'False'
        self.Nf = 1
        self.grid_method = 'uniform' #specify 'uniform' or 'non-uniform'
        ## state grid
        
        # grid parameters
        self.Nz = 1000;
        zMin = 0.001; 
        zMax = 0.99;
        
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
        # terminal conditions for value functions and q [can choose anything "reasonable" here]
        self.Je = self.params['aE']**(-self.params['gammaE'])*self.z**(1-self.params['gammaE']); # v of experts
        self.Jh = self.params['aE']**(-self.params['gammaH'])*(1-self.z)**(1-self.params['gammaH']); # v of households
        self.q = np.ones([self.Nz,1]);
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        # allocate memory for other variables
        self.psi = np.full([self.Nz,1],np.NaN)
        self.chi = np.full([self.Nz,1],np.NaN)
        self.ssq = np.full([self.Nz,1],np.NaN)
        self.iota = np.full([self.Nz,1],np.NaN)
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
                    
        eq2 = (self.GS[zi]) * q_p**self.params['IES']  - Psi_p * (self.params['aE'] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1][0])/(dz[zi-1]*q_p) * self.z[zi-1] *(self.params['alpha']* Psi_p/self.z[zi]-1)))  - self.params['sigma'] 
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([-self.params['alpha']**2 * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p**2, \
                            -2*self.params['alpha']*(self.params['alpha'] * Psi_p - self.z[zi]) * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p + (self.params['gammaH'] - self.params['gammaE']) * self.params['sigma'], \
                                  -(self.params['aE'] - self.params['aH'])/q_p**2])
        QN[1,:] = np.array([self.params['aH'] - self.params['aE'], 0, self.GS[zi]*q_p**(self.params['IES']-1)*self.params['IES'] + 1/self.params['kappa']])
        
        QN[2,:] = np.array([-sig_ka_p * self.params['alpha'] * (1- self.q[zi-1][0]/q_p) / (dz[zi-1]) , \
                          1 - (1- (self.q[zi-1][0]/q_p)) / dz[zi-1] * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1] , \
                            sig_ka_p * (-self.q[zi-1][0]/(q_p**2 * dz[zi-1]) * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN
        
    def equations_region2(self, q_p, zi):
        '''
        Solves for the equilibrium policy in the normal region 
        Input: old values of capital price(q_p), return volatility(sig_ka_p), grid point(zi)
        Output: new values from Newton-Rhaphson method
        '''
        i_p = (q_p - 1)/self.params['kappa']
        eq1 = q_p ** self.params['IES'] * self.GS[zi] - self.params['aE'] + i_p 
        QN = self.params['IES'] * q_p ** (self.params['IES'] - 1) * self.GS[zi] + 1/self.params['kappa']
        EN = q_p - eq1/QN
        return EN

    def initialize_(self):
        '''
        Initialize the capital price, capital share, and return volatility
        '''
        sig_ka_q = self.params['sigma']
        Psi_q =0
        GS_e = self.Je**((1-self.params['IES'])/(1-self.params['gammaE'])) * ((self.z)**(self.params['IES']))
        GS_h = self.Jh**((1-self.params['IES'])/(1-self.params['gammaH'])) * ((1-self.z)**(self.params['IES']))
        self.GS = GS_e + GS_h
        qL = 0
        qR = self.params['aH']*self.params['kappa'] +1
        
        for k in range(30):
            q = (qL + qR)/2
            iota = (q-1)/self.params['kappa']
            A0 = self.params['aH'] - iota
            
            if (np.log(q) * (self.params['IES']) + np.log(self.GS[0])) > np.log(A0):
                qR = q
            else:
                qL = q
            
            self.q[0,:] = q
                    
        return (q, Psi_q, sig_ka_q)

    def solve(self):
        
        ## main iteration
        
        for timeStep in range(self.maxIterations):
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe = np.hstack([(self.logValueE[1]-self.logValueE[0])/(self.z[1]-self.z[0]),(self.logValueE[2:]-self.logValueE[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueE[-1]-self.logValueE[-2])/(self.z[-1]-self.z[-2])]);
            self.dLogJh = np.hstack([(self.logValueH[1]-self.logValueH[0])/(self.z[1]-self.z[0]),(self.logValueH[2:]-self.logValueH[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueH[-1]-self.logValueH[-2])/(self.z[-1]-self.z[-2])]);
            self.chi[:,0] = np.maximum(self.params['alpha'],self.z)
            
            for i in range(1,self.Nz):
                  
                  if self.psi[i-1] >= 1:
                	      break; 
                  if i==1:

                    q_init, Psi_init, sig_ka_init = self.initialize_()  
                    self.q[i-1,:], self.psi[i-1][0], self.ssq[i-1][0]  = q_init, Psi_init, sig_ka_init
                    result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                    self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                  else:
                    result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                    self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                    self.chi[i] = self.params['alpha']*self.psi[i]
                  
                  self.dq[i] = (1 - self.params['sigma']/self.ssq[i])/(self.chi[i] - self.z[i])*self.q[i]
                  self.iota[i] = (self.q[i]-1)/self.params['kappa']
            self.iota[0] = self.iota[1]      
            self.thresholdIndex = i-1;
            self.crisis_eta = self.z[self.thresholdIndex]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,1)), dtype = np.float64)
            self.crisis_flag[0:self.thresholdIndex] = 1
              
            self.psi[self.thresholdIndex:] = 1;

            for i in range(self.thresholdIndex,self.Nz):
                self.q[i] = self.equations_region2(self.q[i-1],i)

            
            self.chi[self.thresholdIndex:] = np.maximum(self.z[self.thresholdIndex:],self.params['alpha']).reshape(-1,1); 
            self.iota[self.thresholdIndex:] = (self.q[self.thresholdIndex:]-1)/self.params['kappa']
            if self.thresholdIndex==0:
                self.dq[self.thresholdIndex:-1] = (self.q[1:] - np.vstack([self.q0,self.q[0:-2]])) / (self.z - np.vstack([0,self.z[:-2]])) 
            else:
                self.dq[self.thresholdIndex:] = (self.q[self.thresholdIndex:]- self.q[self.thresholdIndex-1:-1]).reshape(-1,1)/(self.z[self.thresholdIndex:]-self.z[self.thresholdIndex-1:-1]).reshape(-1,1);
            self.ssq[self.thresholdIndex:] = self.params['sigma']/(1-self.dq[self.thresholdIndex:]/self.q[self.thresholdIndex:] * (self.chi[self.thresholdIndex:]-self.z[self.thresholdIndex:].reshape(-1,1)));
            
            self.theta = (self.chi)/self.z.reshape(-1,1)
            self.thetah = (1-self.chi)/(1-self.z.reshape(-1,1))
            self.Phi = (np.log(self.q))/self.params['kappa']
            

            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2.reshape(-1,1)); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qzl  = self.qz/self.q ; 
            self.qzzl = self.qzz/ self.q;

            self.sig_za = (self.chi - self.z.reshape(-1,1))*self.ssq; 
            self.priceOfRiskE = (1/self.z.reshape(-1,1) - self.dLogJe.reshape(-1,1)) * self.sig_za + self.ssq + (self.params['gammaE']-1) * self.params['sigma'];
            self.priceOfRiskH = -(1/(1-self.z.reshape(-1,1)) + self.dLogJh.reshape(-1,1))*self.sig_za + self.ssq + (self.params['gammaH']-1) * self.params['sigma'];
            self.sig_je = self.dLogJe.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.sig_jh = self.dLogJh.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            
            self.rp = self.priceOfRiskE * self.ssq
            self.rp_ = self.priceOfRiskH * self.ssq
            self.consWealthRatioE = self.Je.reshape(-1,1)**((1-self.params['IES'])/(1-self.params['gammaE'])) / ((self.z_mat*self.q)**(1-self.params['IES']))
            self.consWealthRatioH = self.Jh.reshape(-1,1)**((1-self.params['IES'])/(1-self.params['gammaH'])) / (((1-self.z_mat)*self.q)**(1-self.params['IES']))
            self.mu_z = self.z_mat*((self.params['aE']-self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*self.ssq*(self.rp/self.ssq - self.ssq) + self.ssq*(1-self.params['alpha'])*(self.rp/self.ssq - self.rp_/self.ssq) + (self.params['lambda_d']/self.z_mat)*(self.params['zbar']-self.z_mat))
            self.growthRate = np.log(self.q)/self.params['kappa']-self.params['delta'];
            self.sig_za[0:2] = 0; 
            self.mu_z[0] = 0; 
            
            
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.mu_q = self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.params['aE'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            try:
                self.r[self.thresholdIndex:self.thresholdIndex+2] = 0.5*(self.r[self.thresholdIndex+2] + self.r[self.thresholdIndex-1]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
            except:
                print('no crisis',self.thresholdIndex)
            
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
            self.A = self.psi*(self.params['aE']) + (1-self.psi) * (self.params['aH'])
            self.AminusIota = self.psi*(self.params['aE'] - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
            self.pd = np.log(self.q / self.AminusIota) #log pd ratio
            # PDE time steps
            self.linearTermE = (self.params['gammaE']-1)/(1-1/self.params['IES']) * ((self.q.reshape(-1,1)*self.z_mat)**(self.params['IES']-1) * self.Je.reshape(-1,1)**((1-self.params['IES'])/(1-self.params['gammaE'])) - self.params['rhoE']) - \
                        (1-self.params['gammaE']) * (self.growthRate - self.params['gammaE']*self.params['sigma']**2 *0.5 + self.params['sigma'] * self.sig_je); 
            dt = 0.8
            newJe = hjb_implicit_policy(self.z,self.linearTermE,self.mu_z,self.sig_za,0,self.Je,dt).reshape(-1)
           
           
            
            self.linearTermH = (self.params['gammaH']-1)/(1-1/self.params['IES']) * ((self.q.reshape(-1,1)*(1-self.z_mat))**(self.params['IES']-1) * self.Jh.reshape(-1,1)**((1-self.params['IES'])/(1-self.params['gammaH'])) - self.params['rhoH']) - \
                        (1-self.params['gammaH']) * (self.growthRate - self.params['gammaH']*self.params['sigma']**2/2 + self.params['sigma']*self.sig_jh)
            newJh = hjb_implicit_policy(self.z,self.linearTermH,self.mu_z,self.sig_za,0,self.Jh,dt).reshape(-1)
            newJe[0] = newJe[1]
            newJh[0] = newJh[1]
            
            #check convergence
            relChangeJe = np.abs(newJe-self.Je) / (np.abs(newJe)+np.abs(self.Je))*2/self.dt;
            relChangeJh = np.abs(newJh-self.Jh) / (np.abs(newJh)+np.abs(self.Jh))*2/self.dt;
            self.Je = newJe;
            self.Jh = newJh;
            if np.maximum(np.amax(relChangeJe),np.amax(relChangeJh)) < self.convergenceCriterion:
                self.converged = 'True'
                break
        self.kfe()
        
    def kfe(self):
        '''
        compute stationary distribution of wealth share
        '''
        self.coeff = 2*self.mu_z[1:-1]/(self.sig_za[1:-1]**2)
        self.coeff[0] = self.coeff[1]
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
    params={'rhoE': 0.05, 'rhoH': 0.05, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.75, 'kappa':7, 'delta':0.025, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':2, 'gammaH':2, 'IES':1.5}
    sim1 = model_recursive_general(params)
    sim1.solve()  
    
    
    alpha = 1.0
    sim2 = model_recursive_general(params)
    sim2.solve()  
    
    plt.figure()
    plt.plot(sim1.chi[2:]), plt.plot(sim2.chi[2:])
    plt.figure()
    plt.plot(sim1.q[2:]), plt.plot(sim2.q[2:])
    plt.figure()
    plt.plot(sim1.AminusIota[2:]), plt.plot(sim2.AminusIota[2:])
    plt.figure()
    
    #plt.rc('text', usetex=False)
    plt.plot(sim1.z[2:], sim1.pd[2:],label='$\chi$ = 0.5')
    plt.plot(sim1.z[2:], sim2.pd[2:],label='$\chi$ = 1.0')
    plt.ylabel('Price-Dividend ratio',fontsize=15)
    plt.xlabel('Wealth-share (z)',fontsize=15)
    plt.legend(loc=0, fontsize = 15)
    plt.savefig('../output/plots/pdratio.png')

    plt.figure()
    plt.plot(sim1.consWealthRatioE[2:]), plt.plot(sim2.consWealthRatioE[2:])    

    plt.figure()
    plt.plot(sim1.z[2:], sim1.consWealthRatioH[2:], label='$\chi$ = 0.5')
    plt.plot(sim2.z[2:], sim2.consWealthRatioH[2:], label='$\chi$ = 1.0')
    plt.ylabel('Consumption-Wealth ratio: Households',fontsize=15)
    plt.xlabel('Wealth-share (z)',fontsize=15)
    plt.legend(loc=0, fontsize = 15)
    plt.savefig('../output/plots/cwh.png')
    
    plt.figure()
    plt.plot(sim1.z[2:], sim1.consWealthRatioE[2:], label='$\chi$ = 0.5')
    plt.plot(sim2.z[2:], sim2.consWealthRatioE[2:], label='$\chi$ = 1.0')
    plt.ylabel('Consumption-Wealth ratio: Experts',fontsize=15)
    plt.xlabel('Wealth-share (z)',fontsize=15)
    plt.legend(loc=0, fontsize = 15)
    plt.savefig('../output/plots/cwe.png')
    
    
    
    
    
    
    
    
 
    
    
    
    