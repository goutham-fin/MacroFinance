from scipy.optimize import fsolve
import numpy as np
from finite_difference import hjb_implicit_bruSan_upwind, hjb_implicit_bruSan_policy
from staticStepAllocation import staticStepAllocationInnerPsi
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
    This class solves the baseline model with recursive utility (IES!=1) from the paper 
    'Confronting Macro-finance model with Data (2020).  
    '''
    def __init__(self, rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, IES, kappa, delta, lambda_d, zbar):
        
        self.rhoE = rhoE # discount rate of experts
        self.rhoH = rhoH; # discount rate of households
        self.aE = aE # productivity of experts 
        self.aH = aH # productivity of households
        self.sigma = sigma # aggregate risk
        self.alpha = alpha # skin in the game
        self.gammaE = gammaE # relative risk aversion of experts
        self.gammaH = gammaH # relative risk aversion of households [before choosing ~= gammaE, correct line 105 first!]
        self.IES = IES
        self.kappa = kappa # capital adjustment cost parameter
        self.delta = delta # depreciation rate
        self.etabar = zbar;
        self.lambda_d = lambda_d;
        
        # algorithm parameters
        self.maxIterations = 50; 
        self.convergenceCriterion = 1e-6; 
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
        
        self.Je = self.aE**(-self.gammaE)*self.z**(1-self.gammaE); # value function of experts
        self.Jh = self.aE**(-self.gammaH)*(1-self.z)**(1-self.gammaH); # value function of households
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
        i_p = (q_p -1)/self.kappa
        
        eq1 = (self.aE-self.aH)/q_p  - \
                            self.alpha* (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * (self.alpha* Psi_p - self.z[zi]) * sig_ka_p**2 - (self.gammaE - self.gammaH) * sig_ka_p * self.sigma
                    
        eq2 = (self.GS[zi]) * q_p**self.IES  - Psi_p * (self.aE - i_p) - (1- Psi_p) * (self.aH - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1][0])/(dz[zi-1]*q_p) * self.z[zi-1] *(self.alpha* Psi_p/self.z[zi]-1)))  - self.sigma 
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([-self.alpha**2 * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p**2, \
                            -2*self.alpha*(self.alpha * Psi_p - self.z[zi]) * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p + (self.gammaH - self.gammaE) * self.sigma, \
                                  -(self.aE - self.aH)/q_p**2])
        QN[1,:] = np.array([self.aH - self.aE, 0, self.GS[zi]*q_p**(self.IES-1)*self.IES + 1/self.kappa])
        
        QN[2,:] = np.array([-sig_ka_p * self.alpha * (1- self.q[zi-1][0]/q_p) / (dz[zi-1]) , \
                          1 - (1- (self.q[zi-1][0]/q_p)) / dz[zi-1] * (self.alpha * Psi_p/self.z[zi] -1) * self.z[zi-1] , \
                            sig_ka_p * (-self.q[zi-1][0]/(q_p**2 * dz[zi-1]) * (self.alpha * Psi_p/self.z[zi] -1) * self.z[zi-1])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN
        
    def equations_region2(self, q_p, zi):
        '''
        Solves for the equilibrium policies in normal regime
        '''
        i_p = (q_p - 1)/self.kappa #equilibrium investment rate
        eq1 = q_p ** self.IES * self.GS[zi] - self.aE + i_p 
        QN = self.IES * q_p ** (self.IES - 1) * self.GS[zi] + 1/self.kappa
        EN = q_p - eq1/QN
        return EN

    def initialize_(self):
        sig_ka_q = self.sigma
        Psi_q =0
        GS_e = self.Je**((1-self.IES)/(1-self.gammaE)) * ((self.z)**(self.IES))
        GS_h = self.Jh**((1-self.IES)/(1-self.gammaH)) * ((1-self.z)**(self.IES))
        self.GS = GS_e + GS_h
        qL = 0
        qR = self.aH*self.kappa +1
        
        for k in range(30):
            q = (qL + qR)/2
            iota = (q-1)/self.kappa
            A0 = self.aH - iota
            
            if (np.log(q) * (self.IES) + np.log(self.GS[0])) > np.log(A0):
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
            self.chi[:,0] = np.maximum(self.alpha,self.z)
            
            for i in range(1,self.Nz):
                  
                  if self.psi[i-1] >= 1:
                	      break; #break out if normal regime is reached
                  if i==1:

                    q_init, Psi_init, sig_ka_init = self.initialize_()  
                    self.q[i-1,:], self.psi[i-1][0], self.ssq[i-1][0]  = q_init, Psi_init, sig_ka_init
                    result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                    self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                  else:
                    result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                    self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                    self.chi[i] = self.alpha*self.psi[i]
                  
                  self.dq[i] = (1 - self.sigma/self.ssq[i])/(self.chi[i] - self.z[i])*self.q[i]
                  self.iota[i] = (self.q[i]-1)/self.kappa
            self.iota[0] = self.iota[1]      
            self.thresholdIndex = i-1;
            self.crisis_eta = self.z[self.thresholdIndex]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,1)), dtype = np.float64)
            self.crisis_flag[0:self.thresholdIndex] = 1
              
            self.psi[self.thresholdIndex:] = 1;

            for i in range(self.thresholdIndex,self.Nz):
                self.q[i] = self.equations_region2(self.q[i-1],i)

        
            self.chi[self.thresholdIndex:] = np.maximum(self.z[self.thresholdIndex:],self.alpha).reshape(-1,1); 
            self.iota[self.thresholdIndex:] = (self.q[self.thresholdIndex:]-1)/self.kappa
            if self.thresholdIndex==0:
                self.dq[self.thresholdIndex:-1] = (self.q[1:] - np.vstack([self.q0,self.q[0:-2]])) / (self.z - np.vstack([0,self.z[:-2]])) #needs fixing
            else:
                self.dq[self.thresholdIndex:] = (self.q[self.thresholdIndex:]- self.q[self.thresholdIndex-1:-1]).reshape(-1,1)/(self.z[self.thresholdIndex:]-self.z[self.thresholdIndex-1:-1]).reshape(-1,1);
            self.ssq[self.thresholdIndex:] = self.sigma/(1-self.dq[self.thresholdIndex:]/self.q[self.thresholdIndex:] * (self.chi[self.thresholdIndex:]-self.z[self.thresholdIndex:].reshape(-1,1)));
            
            self.theta = (self.chi)/self.z.reshape(-1,1)
            self.thetah = (1-self.chi)/(1-self.z.reshape(-1,1))
            self.Phi = (np.log(self.q))/self.kappa
            
            #Time step 

            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2.reshape(-1,1)); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qzl  = self.qz/self.q ; 
            self.qzzl = self.qzz/ self.q;

            self.sig_za = (self.chi - self.z.reshape(-1,1))*self.ssq; #sig_za := \sigma^\z \z, similary mu_z
            self.priceOfRiskE = (1/self.z.reshape(-1,1) - self.dLogJe.reshape(-1,1)) * self.sig_za + self.ssq + (self.gammaE-1) * self.sigma;
            self.priceOfRiskH = -(1/(1-self.z.reshape(-1,1)) + self.dLogJh.reshape(-1,1))*self.sig_za + self.ssq + (self.gammaH-1) * self.sigma;
            self.sig_je = self.dLogJe.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.sig_jh = self.dLogJh.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            
            self.rp = self.priceOfRiskE * self.ssq
            self.rp_ = self.priceOfRiskH * self.ssq
            self.consWealthRatioE = self.Je.reshape(-1,1)**((1-self.IES)/(1-self.gammaE)) / ((self.z_mat*self.q)**(1-self.IES))
            self.consWealthRatioH = self.Jh.reshape(-1,1)**((1-self.IES)/(1-self.gammaH)) / (((1-self.z_mat)*self.q)**(1-self.IES))
            self.mu_z = self.z_mat*((self.aE-self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*self.ssq*(self.rp/self.ssq - self.ssq) + self.ssq*(1-self.alpha)*(self.rp/self.ssq - self.rp_/self.ssq) + (self.lambda_d/self.z_mat)*(self.etabar-self.z_mat))
            self.growthRate = np.log(self.q)/self.kappa-self.delta;
            self.sig_za[0:2] = 0; 
            self.mu_z[0] = 0; 
            self.Phi = (np.log(self.q))/self.kappa
            self.mu_q = self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.mu_rH = (self.aH - self.iota)/self.q + self.Phi - self.delta + self.mu_q + self.sigma * (self.ssq - self.sigma)
            self.mu_rE = (self.aE - self.iota)/self.q + self.Phi - self.delta + self.mu_q + self.sigma * (self.ssq - self.sigma)
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
            try:
                self.r[self.thresholdIndex:self.thresholdIndex+2] = 0.5*(self.r[self.thresholdIndex+2] + self.r[self.thresholdIndex-1]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
            except:
                print('no crisis',self.thresholdIndex)
            self.A = self.psi*(self.aE) + (1-self.psi) * (self.aH)
            self.AminusIota = self.psi*(self.aE - self.iota) + (1-self.psi) * (self.aH - self.iota)
            self.pd = np.log(self.q / self.AminusIota) #log pd ratio
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
            # PDE time steps
            self.diffusion = self.sig_za**2/2;
            dt = 0.8
            #solve for experts
            newJe = hjb_implicit_policy(self.z,self.linearTermE,self.mu_z,self.sig_za,0,self.Je,dt).reshape(-1)
            #solve for households
            newJh = hjb_implicit_policy(self.z,linearTerm,self.mu_z,self.sig_za,0,self.Jh,dt).reshape(-1)
            self.Je[0] = self.Je[1]
            self.Jh[0] = self.Jh[1]
            # (4) check convergence
            relChangeJe = np.abs((newJe-self.Je) / self.Je)
            relChangeJh = np.abs((newJh-self.Jh) / self.Jh)
            self.Je = newJe;
            self.Jh = newJh;
            if np.maximum(np.amax(relChangeJe),np.amax(relChangeJh)) < self.convergenceCriterion:
                self.converged = 'True'
                break #break out if relative error is less then convergence criterion
        self.kfe()
        
    def kfe(self):
        '''
        Solves for the Kolmogorov-Forward-equation
        Input: grid, diffusion (sig_za), and drift (mu_z) of wealth share
        Output: stationary distribution of wealth share
        '''
        self.coeff = 2*self.mu_z[1:-1]/(self.sig_za[1:-1]**2)
        self.coeff[0] = self.coeff[1]
        self.coeff_fn = interp1d(self.z[1:-1],self.coeff.reshape(-1),kind='nearest',fill_value='extrapolate')
        Nh = 10000
        self.tv, self.dist = self.forward_euler(self.coeff_fn,1,0,1,Nh);
        #convert into distribution
        self.dist_fn = interp1d(self.tv, self.dist, kind = 'linear', fill_value = 'extrapolate')
        self.f = np.full([self.Nz,1],np.nan)
        for i in range(1,self.Nz):
            self.f[i]= self.dist_fn(self.z[i])/self.sig_za[i]**2
        h=1/Nh
        area = np.abs(np.nansum(h*self.f[1:]))
        self.f_norm = self.f/area
    def forward_euler(self, fun, y0, t0, tf, Nh):
        '''
        Forward euler method to solve the Kolmogorov-Forward-equation
        Input: fun: old values of distribution
                (y0, t0): initial values of wealth share and time
                Nh: number of steps in the grid
        Output: new values of stationary distribution
        '''
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
    rhoE = 0.05; rhoH = 0.05; lambda_d = 0.03; sigma = 0.06; kappa = 7; delta = 0.025; zbar = 0.1; aE = 0.11; aH = 0.03; alpha=1.0;
    gammaE = 2; gammaH = 2; IES = 1.5; utility = 'recursive_general'; nsim = 3
    sim1 = benchmark_bruSan_recursive_general(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, IES, kappa, delta, lambda_d, zbar)
    sim1.solve()  
    
    alpha = 1.0
    sim2 = benchmark_bruSan_recursive_general(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, IES, kappa, delta, lambda_d, zbar)
    sim2.solve()  
    
    plt.figure()
    plt.plot(sim1.chi[2:]), plt.plot(sim2.chi[2:])
    plt.figure()
    plt.plot(sim1.q[2:]), plt.plot(sim2.q[2:])
    plt.figure()
    plt.plot(sim1.AminusIota[2:]), plt.plot(sim2.AminusIota[2:])
    plt.figure()
    
    #inspect price dividend ratio
    plt.plot(sim1.z[2:], sim1.pd[2:],label='$\chi$ = 0.5')
    plt.plot(sim1.z[2:], sim2.pd[2:],label='$\chi$ = 1.0')
    plt.ylabel('Price-Dividend ratio',fontsize=15)
    plt.xlabel('Wealth-share (z)',fontsize=15)
    plt.legend(loc=0, fontsize = 15)
    plt.savefig('../output/plots/pdratio.png')

    plt.figure()
    plt.plot(sim1.consWealthRatioE[2:]), plt.plot(sim2.consWealthRatioE[2:])    

    #inspect consumption-wealth ratio
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
    
    
    
    
    
    
    
    
 
    
    
    
    