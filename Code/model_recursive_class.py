from scipy.optimize import fsolve

#import matplotlibpyplot as plt
#USES LINEAR z Grid and allows for different gamma values
#switched r to r1
import numpy as np

from finite_difference import hjb_implicit_upwind
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d

class model_recursive():
    def __init__(self, rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar):
        
        self.rhoE = rhoE # discount rate of experts
        self.rhoH = rhoH; # discount rate of households
        self.aE = aE # productivity of experts 
        self.aH = aH # productivity of households
        self.sigma = sigma # exogenous volatility
        self.alpha = alpha # skin in the game constraint
        self.gammaE = gammaE #  risk aversion of experts
        self.gammaH = gammaH #  risk aversion of households 
        self.kappa = kappa # capital adjustment cost parameter
        self.delta = delta # depreciation rate
        self.etabar = zbar; # population share of experts
        self.lambda_d = lambda_d; # death rate
        
        # algorithm parameters
        self.continueOldIteration = 'False'; 
        self.maxIterations = 400; 
        self.convergenceCriterion = 1e-6; 
        self.dt = 0.4; #time step width
        self.converged = 'False'

        self.Nf = 1
        self.grid_method = 'uniform' #specify 'uniform' or 'non-uniform'
        ## state grid
        
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
        
        self.Je = self.aE**(-self.gammaE)*self.z**(1-self.gammaE); # v of experts
        self.Jh = self.aE**(-self.gammaH)*(1-self.z)**(1-self.gammaH); # v of households
        self.q = np.ones([self.Nz,1]);
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        # allocate memory for other variables
        self.psi = np.full([self.Nz,1],np.NaN)
        self.chi = np.full([self.Nz,1],np.NaN)
        self.ssq = np.full([self.Nz,1],np.NaN)
        self.iota = np.full([self.Nz,1],np.NaN)
        self.dq = np.full([self.Nz,1],np.NaN)
    
    
    def equations_region1(self, q_p, Psi_p, sig_ka_p, zi):    #crisis region
        dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        i_p = (q_p -1)/self.kappa
        eq1 = (self.aE-self.aH)/q_p  - \
                            self.alpha* (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * (self.alpha* Psi_p - self.z[zi]) * sig_ka_p**2 - (self.gammaE - self.gammaH) * sig_ka_p * self.sigma
                    
        eq2 = (self.rhoE*self.z[zi] + self.rhoH*(1-self.z[zi])) * q_p  - Psi_p * (self.aE - i_p) - (1- Psi_p) * (self.aH - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1][0])/(dz[zi-1]*q_p) * self.z[zi-1] *(self.alpha* Psi_p/self.z[zi]-1)))  - self.sigma 
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([-self.alpha**2 * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p**2, \
                            -2*self.alpha*(self.alpha * Psi_p - self.z[zi]) * (self.dLogJh[zi] - self.dLogJe[zi] + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p + (self.gammaH - self.gammaE) * self.sigma, \
                                  -(self.aE - self.aH)/q_p**2])
        QN[1,:] = np.array([self.aH - self.aE, 0, self.z[zi]* self.rhoE + (1-self.z[zi])* self.rhoH + 1/self.kappa])
        
        QN[2,:] = np.array([-sig_ka_p * self.alpha * (1- self.q[zi-1][0]/q_p) / (dz[zi-1]) , \
                          1 - (1- (self.q[zi-1][0]/q_p)) / dz[zi-1] * (self.alpha * Psi_p/self.z[zi] -1) * self.z[zi-1] , \
                            sig_ka_p * (-self.q[zi-1][0]/(q_p**2 * dz[zi-1]) * (self.alpha * Psi_p/self.z[zi] -1) * self.z[zi-1])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN



    def solve(self):
        
        ## main iteration
        
        # initialize variables at eta=0
        self.psi[0] = 0;
        self.q[0] = (1 + self.kappa*(self.aH + self.psi[0]*(self.aE-self.aH)))/(1 + self.kappa*(self.rhoH + self.z[0] * (self.rhoE - self.rhoH)));
        self.chi[0] = 0;
        self.ssq[0] = self.sigma;
        self.q0 = (1 + self.kappa * self.aH)/(1 + self.kappa * self.rhoH); #heoretical limit at z=0 [just used in a special case below that is probably never entered]
        self.iota[0] = (self.q0-1)/self.kappa
        
        for timeStep in range(self.maxIterations):
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe = np.hstack([(self.logValueE[1]-self.logValueE[0])/(self.z[1]-self.z[0]),(self.logValueE[2:]-self.logValueE[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueE[-1]-self.logValueE[-2])/(self.z[-1]-self.z[-2])]);
            self.dLogJh = np.hstack([(self.logValueH[1]-self.logValueH[0])/(self.z[1]-self.z[0]),(self.logValueH[2:]-self.logValueH[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueH[-1]-self.logValueH[-2])/(self.z[-1]-self.z[-2])]);
          	
            for i in range(1,self.Nz):
                  
                  if self.psi[i-1] >= 1:
                	      break; #break out if normal regime is reached
            	    
                  result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                  self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                  self.chi[i] = self.alpha*self.psi[i]
                  self.dq[i] = (1 - self.sigma/self.ssq[i])/(self.chi[i] - self.z[i])*self.q[i]
                  self.iota[i] = (self.q[i]-1)/self.kappa
            self.thresholdIndex = i-1;
            self.crisis_eta = self.z[self.thresholdIndex]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,1)), dtype = np.float64)
            self.crisis_flag[0:self.thresholdIndex] = 1
              
            self.psi[self.thresholdIndex:] = 1;
            self.q[self.thresholdIndex:] = (1 + self.kappa*(self.aH + self.psi[self.thresholdIndex:]*(self.aE-self.aH))).reshape(-1,1)/(1 + self.kappa*(self.rhoH + self.z[self.thresholdIndex:]*(self.rhoE-self.rhoH))).reshape(-1,1);
            self.chi[self.thresholdIndex:] = np.maximum(self.z[self.thresholdIndex:],self.alpha).reshape(-1,1); 
            
            self.iota[self.thresholdIndex:] = (self.q[self.thresholdIndex:]-1)/self.kappa
            if self.thresholdIndex==0:
                self.dq[self.thresholdIndex:-1] = (self.q[1:] - np.vstack([self.q0,self.q[0:-2]])) / (self.z - np.vstack([0,self.z[:-2]])) #needs fixing
            else:
                self.dq[self.thresholdIndex:] = (self.q[self.thresholdIndex:]- self.q[self.thresholdIndex-1:-1]).reshape(-1,1)/(self.z[self.thresholdIndex:]-self.z[self.thresholdIndex-1:-1]).reshape(-1,1);
            self.ssq[self.thresholdIndex:] = self.sigma/(1-self.dq[self.thresholdIndex:]/self.q[self.thresholdIndex:] * (self.chi[self.thresholdIndex:]-self.z[self.thresholdIndex:].reshape(-1,1)));
            self.theta = (self.chi)/self.z.reshape(-1,1)
            self.thetah = (1-self.chi)/(1-self.z.reshape(-1,1))
            self.theta[0] = self.theta[1]
            self.thetah[0] = self.thetah[1]
            self.Phi = (np.log(self.q))/self.kappa
            # (3) time step preparations

            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2.reshape(-1,1)); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qzl  = self.qz/self.q ; 
            self.qzzl = self.qzz/ self.q;

            self.consWealthRatioE = self.rhoE;
            self.consWealthRatioH = self.rhoH;
            self.sig_za = (self.chi - self.z.reshape(-1,1))*self.ssq; #sig_za := \sigma^\z \z, similary mu_z
            self.priceOfRiskE = (1/self.z.reshape(-1,1) - self.dLogJe.reshape(-1,1)) * self.sig_za + self.ssq + (self.gammaE-1) * self.sigma;
            self.priceOfRiskH = -(1/(1-self.z.reshape(-1,1)) + self.dLogJh.reshape(-1,1))*self.sig_za + self.ssq + (self.gammaH-1) * self.sigma;
            self.sig_je = self.dLogJe.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.sig_jh = self.dLogJh.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.rp = self.priceOfRiskE * self.ssq
            self.rp_ = self.priceOfRiskH * self.ssq
            
            self.mu_z = self.z_mat*((self.aE-self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*self.ssq*(self.rp/self.ssq - self.ssq) + self.ssq*(1-self.alpha)*(self.rp/self.ssq - self.rp_/self.ssq) + (self.lambda_d/self.z_mat)*(self.etabar-self.z_mat))
            
            self.growthRate = np.log(self.q)/self.kappa-self.delta;
            self.sig_za[0] = 0; 
            self.mu_z[0] = np.maximum(self.mu_z[0],0); 
            self.mu_z[-1] = np.minimum(self.mu_z[-1],0); 
            
            
            self.Phi = (np.log(self.q))/self.kappa
            self.mu_q = self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.mu_rH = (self.aH - self.iota)/self.q + self.Phi - self.delta + self.mu_q + self.sigma * (self.ssq - self.sigma)
            self.mu_rE = (self.aE - self.iota)/self.q + self.Phi - self.delta + self.mu_q + self.sigma * (self.ssq - self.sigma)
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            self.r[self.thresholdIndex:self.thresholdIndex+2] = 0.5*(self.r[self.thresholdIndex+2] + self.r[self.thresholdIndex-1]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
            
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
            
            diffusion = self.sig_za**2/2;
            
            advection = self.mu_z + (1-self.gammaE)*self.sigma*self.sig_za;
            linearTerm = (1-self.gammaE)*(self.growthRate.reshape(-1,1) + self.rhoE*(np.log(self.rhoE)+np.log(self.z.reshape(-1,1)*self.q)) - self.gammaE/2*self.sigma**2) - np.log(self.Je.reshape(-1,1)); #PDE coefficient multiplying v(\z)
            # solving PDE using implicit scheme
            newValueAux = hjb_implicit_upwind(np.vstack([0,self.z.reshape(-1,1),1]),np.vstack([np.NaN,self.Je.reshape(-1,1),np.NaN]),self.dt, \
                                                        np.vstack([0,diffusion,0]),np.vstack([0,advection,0]),np.vstack([0,linearTerm,0]),np.vstack([0,np.zeros([self.z.shape[0],1]),0]), 0, 0); #add fake grid extensions, because solver ignores boundary coefficients
            newJe = newValueAux[1:-1]; 
            # (c) households
            advection = self.mu_z + (1-self.gammaH)*self.sigma*self.sig_za;
            linearTerm = (1-self.gammaH)*(self.growthRate.reshape(-1,1) + self.rhoH*(np.log(self.rhoH)+np.log((1-self.z.reshape(-1,1))*self.q)) - self.gammaH/2*self.sigma**2) - np.log(self.Jh.reshape(-1,1)); #PDE coefficient multiplying v(\z)
            # solving PDE using implicit scheme
            newValueAux = hjb_implicit_upwind(np.vstack([0,self.z.reshape(-1,1),1]),np.vstack([np.NaN,self.Jh.reshape(-1,1),np.NaN]),self.dt, \
                                                         np.vstack([0,diffusion,0]),np.vstack([0,advection,0]),np.vstack([0,linearTerm,0]),np.vstack([0,np.zeros([self.z.shape[0],1]),0]),0,0); #add fake grid extensions, because solver ignores boundary coefficients
            newJh = newValueAux[1:-1]; 
                  
            # checkign convergence
            relChangeJe = np.abs(newJe-self.Je) / (np.abs(newJe)+np.abs(self.Je))*2/self.dt;
            relChangeJh = np.abs(newJh-self.Jh) / (np.abs(newJh)+np.abs(self.Jh))*2/self.dt;
            self.Je = newJe;
            self.Jh = newJh;
            if np.maximum(np.amax(relChangeJe),np.amax(relChangeJh)) < self.convergenceCriterion:
                self.converged = 'True'
                break
            self.A = self.psi*(self.aE) + (1-self.psi) * (self.aH)
            self.AminusIota = self.psi*(self.aE - self.iota) + (1-self.psi) * (self.aH - self.iota)
            self.pd = self.q / self.AminusIota
        
        if self.converged == 'True':
            print('Algortihm converged after {} time steps.\n'.format(timeStep));
        else:
            print('Algorithm terminated without convergence after {} time steps.'.format(timeStep));
        
        self.kfe()
        
    def kfe(self):
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
    #rhoE = 0.06; rhoH = 0.03; aE = 0.11; aH = 0.03;  alpha = 1.0;  kappa = 5; delta = 0.035; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
    rhoE = 0.06; rhoH = 0.04; lambda_d = 0.01; sigma = 0.06; kappa = 10; delta = 0.03; zbar = 0.1; aE = 0.11; aH = 0.03; alpha=0.5;
    gammaE = 1; gammaH = 1; 
    sim1 = model_recursive(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar)
    sim1.solve()  
    
    plt.plot(sim1.z[10:],sim1.mu_z[10:],label = '$\lambda_d = 0.01$')
    plt.legend(loc=0,fontsize = 15)
    plt.plot(sim2.z[10:],sim2.mu_z[10:],label = '$\lambda_d = 0.05$')
    plt.legend(loc=0,fontsize = 15)
    plt.axis('tight')
    plt.ylabel('$\mu_z$')
    plt.xlabel('$z$')
    plt.title('Drift of wealth share',fontsize = 20)
    plt.rc('legend', fontsize=12) 
    plt.rc('axes',labelsize = 15)
    plt.savefig('../output/plots/drift_lambdad.png')
    
    plt.figure()
    plt.plot(sim1.z[10:],sim1.mu_z[10:],label = '$\overline{z} = 0.1$')
    plt.legend(loc=0,fontsize = 15)
    plt.plot(sim3.z[10:],sim3.mu_z[10:],label = '$\overline{z} = 0.5$')
    plt.legend(loc=0,fontsize = 15)
    plt.axis('tight')
    plt.ylabel('$\mu_z$')
    plt.xlabel('$z$')
    plt.title('Drift of wealth share',fontsize = 20)
    plt.rc('legend', fontsize=12) 
    plt.rc('axes',labelsize = 15)            
    plt.savefig('../output/plots/drift_z.png')                    	