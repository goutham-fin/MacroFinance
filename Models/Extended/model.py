from scipy.optimize import fsolve
from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 15})
import numpy as np
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import pyplot
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.linewidth'] = 0
import dill
import tensorflow as tf
import time

class nnpde_informed():
    def __init__(self,linearTerm,advection_z,advection_f,diffusion_z,diffusion_f,cross_term,J0,X,layers,X_f,dt,tb,learning_rate,adam_iter):
        
        self.linearTerm = linearTerm
        self.advection_z = advection_z
        self.advection_y = advection_f
        self.diffusion_z = diffusion_z
        self.diffusion_y = diffusion_f
        self.cross_term = cross_term
        self.u = J0
        self.X = X
        self.layers = layers
        self.t_b = tb
        self.X_f = X_f
        self.dt = dt
        self.learning_rate = learning_rate
        self.adam_iter = adam_iter
        
        self.z_u = self.X[:,0:1]
        self.t_u = self.X[:,2:3]
        self.z_f = self.X_f[:,0:1]
        self.t_f = self.X_f[:,2:3]
        self.y_u = self.X[:,1:2]
        self.y_f = self.X_f[:,1:2]
        
        self.lb = np.array([0,self.y_u[0][0], self.dt])
        self.ub = np.array([1,self.y_u[-1][0], 0])
        #self.lb = 0
        #self.ub = 1
        
        self.X_b = np.array([[self.z_u[0][0],self.y_u[0][0], 0],[self.z_u[0][0],self.y_u[0][0], self.dt],[self.z_u[-1][0],self.y_u[-1][0],0.],[self.z_u[-1][0],self.y_u[-1][0],self.dt]])
        self.z_b = np.array(self.X_b[:,0]).reshape(-1,1)
        self.y_b = np.array(self.X_b[:,1]).reshape(-1,1)
        self.t_b = np.array(self.X_b[:,2]).reshape(-1,1)
        #Initialize NNs
        self.weights, self.biases = self.initialize_nn(layers)
        
        #tf placeholders and computational graph
        self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True))
        self.z_u_tf = tf.placeholder(tf.float32,shape=[None,self.z_u.shape[1]])
        self.y_u_tf = tf.placeholder(tf.float32,shape=[None,self.y_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32,shape=[None,self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None,self.u.shape[1]])
        
        
        self.z_b_tf =  tf.placeholder(tf.float32, shape=[None,self.z_b.shape[1]])
        self.y_b_tf =  tf.placeholder(tf.float32, shape=[None,self.y_b.shape[1]])
        self.t_b_tf =  tf.placeholder(tf.float32, shape=[None,self.t_b.shape[1]])
        
        self.z_f_tf = tf.placeholder(tf.float32, shape=[None,self.z_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None,self.y_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None,self.t_f.shape[1]])
        
        
        self.u_pred,_,_ = self.net_u(self.z_u_tf,self.y_u_tf,self.t_u_tf)
        self.f_pred = self.net_f(self.z_f_tf, self.y_f_tf,self.t_f_tf)
        _, self.ub_z_pred, self.ub_y_pred = self.net_u(self.z_b_tf,self.y_b_tf,self.t_b_tf)
        
        
        
        self.loss = tf.reduce_mean(tf.square(self.u_tf-self.u_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred))  #+\
                        #tf.reduce_mean(tf.square(self.ub_y_pred)) #+\
                        #tf.reduce_mean(tf.square(self.ub_z_pred)) #works even without this line for ies1 case
                        
                        
                        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,method='L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol': 1e-08})
                                                                           
        
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    def initialize_nn(self,layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers-1):
            W = self.xavier_init(size = [layers[l],layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype = tf.float32)
            weights.append(W)
            biases.append(b)
        
        return weights,biases

    
    def xavier_init(self,size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        try:
            val = tf.Variable(tf.random.truncated_normal([in_dim,out_dim], stddev = xavier_stddev), dtype = tf.float32)
        except:
            val = tf.Variable(tf.truncated_normal([in_dim,out_dim], stddev = xavier_stddev), dtype = tf.float32)
        return val
    
    def neural_net(self,X,weights,biases):
        num_layers = len(weights) +1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) -1
        #H=X
        for l in range(num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H,W),b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H,W),b)

        return Y

    def net_u(self,z,y,t):
        X = tf.concat([z,y,t],1)
        u = self.neural_net(X,self.weights,self.biases)
        u_z = tf.gradients(u,z)[0]
        u_y = tf.gradients(u,y)[0]
        return u,u_z,u_y
    
    def net_f(self,z,y,t):
        u,u_z,u_y = self.net_u(z,y,t)
        u_t = tf.gradients(u,t)[0]
        u_zz = tf.gradients(u_z,z)[0]
        u_yy = tf.gradients(u_y,y)[0]
        u_zy = tf.gradients(u_z,y)[0]
        f =  u_t + self.diffusion_z * u_zz +  self.diffusion_y * u_yy + self.advection_y * u_y + self.advection_z * u_z + self.cross_term * u_zy -  self.linearTerm *u
        #f = u_t + self.diffusion_z * u_zz + self.advection_z * u_z - self.linearTerm*u
        return f
    
    def callback(self,loss):
        print('Loss: ',loss)
    
    def train(self):
        #K.clear_session()
        tf_dict = {self.z_u_tf: self.z_u, self.y_u_tf: self.y_u, self.t_u_tf: self.t_u, self.u_tf:self.u,
                    self.z_f_tf: self.z_f,self.y_f_tf: self.y_f, self.t_f_tf: self.t_f,
                    self.z_b_tf: self.z_b,
                    self.y_b_tf: self.y_b,
                    self.t_b_tf: self.t_b}
                 
        start_time = time.time()
        
        if True: #set this to true if you want adam to run 
            for it in range(self.adam_iter):
                self.sess.run(self.train_op_Adam, tf_dict)
                # Print
                if it % 1000 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()
        
            start_time = time.time()
        self.optimizer.minimize(self.sess,feed_dict = tf_dict)
        elapsed = time.time() - start_time
        print('Time: %.2f' % elapsed)
        #self.sess.close()


    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.z_u_tf: X_star[:,0:1],self.y_u_tf: X_star[:,1:2], self.t_u_tf: X_star[:,2:3]})
        #f_star = self.sess.run(self.f_pred, {self.z_f_tf: X_star[:,0:1],self.y_f_tf: X_star[:,1:2], self.t_f_tf: X_star[:,2:3]})
        tf.reset_default_graph()
        return u_star




class model_nnpde():
    def __init__(self,params):
        self.params = params
        self.Nz = 1000
        self.Nf = 30
        self.z = np.linspace(0.001,0.999, self.Nz)
        #self.z = 3*zz**2  - 2*zz**3; 
        self.dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        self.dz2 = self.dz[0:self.Nz-2]**2;
        self.z_mat = np.tile(self.z,(self.Nf,1)).transpose()
        self.dz_mat = np.tile(self.dz,(self.Nf,1)).transpose()
        self.dz2_mat = np.tile(self.dz2,(self.Nf,1)).transpose()
        self.f = np.linspace(self.params['f_l'], self.params['f_u'], self.Nf)
        self.df = self.f[1:self.Nf] - self.f[0:self.Nf-1]
        self.df2 = self.df[0:self.Nf-2]**2
        self.f_mat = np.tile(self.f,(self.Nz,1))
        self.df_mat = np.tile(self.df,(self.Nz,1))
        self.df2_mat = np.tile(self.df2,(self.Nz,1))
        
        self.q   =  np.array(np.tile(1,(self.Nz,self.Nf)),dtype=np.float64); 
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qf = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.qff = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.Qfz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        
        self.thetah=  np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.theta= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.r = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.ssq= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.ssf= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.chi= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.iota= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.Je = np.ones([self.Nz,self.Nf]) * 1
        self.Jh = np.ones([self.Nz,self.Nf]) * 1
        self.crisis = np.zeros(self.f.shape)
        self.Jtilde_z= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.Jtilde_f= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);   

        self.first_time = np.linspace(0,0,self.Nf)
        self.psi = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64) 
        
        self.maxIterations=150
        if self.params['scale']>1: self.convergenceCriterion = 1e-2;
        else: self.convergenceCriterion = 1e-2;
        self.converged = False
        self.Iter=0
        if not os.path.exists('../output'):
            os.mkdir('../output')
        self.amax = np.float('Inf')
        self.amax_vec=[]
        
    def equations_region1(self,q_p, Psi_p, sig_qk_p, sig_qf_p, zi, fi):
        i_p = (q_p - 1)/self.params['kappa']
        eq1 = (self.f[fi]-self.params['aH'])/q_p -\
                self.params['alpha'] * self.Jtilde_z[zi,fi]*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi])*(sig_qk_p**2 + sig_qf_p**2 + 2*self.params['corr']*sig_qk_p*sig_qf_p) - self.params['alpha']* self.Jtilde_f[zi,fi]*self.sig_f[zi,fi]*sig_qf_p -\
                self.params['sigma']*(sig_qk_p + self.params['corr']* sig_qf_p)*(self.params['gammaE'] - self.params['gammaH'])
        eq2 = (self.params['rhoE']*self.z_mat[zi,fi] + self.params['rhoH']*(1-self.z_mat[zi,fi])) * q_p  - Psi_p * (self.f[fi] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
              
        eq3 = sig_qk_p - sig_qk_p*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])/self.dz[zi-1] + (sig_qk_p)*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi]) - self.params['sigma']

        if fi==0:
            eq4 = sig_qf_p * self.q[zi-1,fi]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1]  + sig_qf_p - sig_qf_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])
        else:
            eq4 = sig_qf_p * self.q[zi-1,fi]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1] + self.sig_f[zi,fi] * self.q[zi,fi-1]/(q_p * self.df[fi-1]) + sig_qf_p - sig_qf_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])

        ER = np.array([eq1,eq2,eq3,eq4])
        QN = np.zeros(shape=(4,4))

        QN[0,:] = np.array([-self.params['alpha']**2 * self.Jtilde_z[zi,fi]*(sig_qk_p**2 + sig_qf_p**2 + sig_qk_p*sig_qf_p*self.params['corr']*2), -2*self.params['alpha']*self.Jtilde_z[zi,fi]*(self.params['alpha']* Psi_p-self.z_mat[zi,fi])*sig_qk_p - 2*self.params['alpha']*self.Jtilde_z[zi,fi]*self.params['corr']*(self.params['alpha']*Psi_p - self.z_mat[zi,fi])*sig_qf_p -self.params['sigma']*(self.params['gammaE']-self.params['gammaH']), \
                            -2* self.params['alpha'] * self.Jtilde_z[zi,fi]*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])*sig_qf_p - self.Jtilde_f[zi,fi]*self.sig_f[zi,fi] - 2*self.params['corr']*self.params['alpha']*sig_qk_p - self.params['alpha']*self.params['sigma']*self.params['corr']*(self.params['gammaE'] - self.params['gammaH']), -(self.f[fi]-self.params['aH'])/(q_p**2)])
        QN[1,:] = np.array([self.params['aH'] - self.f[fi], 0, 0,  self.params['rhoE'] * self.z_mat[zi,fi] + (1-self.z_mat[zi,fi])*self.params['rhoH'] + 1/self.params['kappa']])
        QN[2,:] = np.array([-sig_qk_p * self.params['alpha']/self.dz[zi-1]*(1-self.q[zi-1,fi]/q_p), 1-((self.params['alpha'] * Psi_p-self.z_mat[zi,fi])/self.dz[zi-1])*(q_p - self.q[zi-1,fi])/q_p, \
                                 0, -sig_qk_p*(self.q[zi-1,fi]/q_p**2)*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])/self.dz[zi-1]])
        if fi==0:
            QN[3,:] = np.array([-sig_qf_p/self.dz[zi-1] + sig_qf_p/self.dz[zi-1] * self.q[zi-1,fi]/q_p, 0, 1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) ,-sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,fi])])
        else:
            QN[3,:] = np.array([-sig_qf_p/self.dz[zi-1] + sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/q_p, 0, \
                                1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]), -sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) - self.sig_f[zi,fi]*self.q[zi,fi-1]/(q_p**2 * self.df[fi-1]) ])
        
        EN = np.array([Psi_p, sig_qk_p, sig_qf_p, q_p]) - np.linalg.solve(QN,ER)
        del ER, QN
        return EN
    def equations_region2(self,q_p,sig_qk_p,sig_qf_p,zi,fi):
        i_p = (q_p-1)/self.params['kappa']
        eq1 = (self.params['rhoE']*self.z_mat[zi,fi] + self.params['rhoH']*(1-self.z_mat[zi,fi])) * q_p  - (self.f[fi] - i_p)
        eq2 = sig_qk_p - sig_qk_p*(self.chi[zi,fi]-self.z_mat[zi,fi])/self.dz[zi-1] + (sig_qk_p)*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.chi[zi,fi] - self.z_mat[zi,fi]) - self.params['sigma']

        if fi==0:
            eq3 = sig_qf_p*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.chi[zi,fi]-self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1]  + sig_qf_p - sig_qf_p/self.dz[zi-1]*(self.chi[zi,fi]-self.z_mat[zi,fi])
        else:
            eq3 = sig_qf_p*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.chi[zi,fi]-self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1] + self.sig_f[zi,fi]*self.q[zi,fi-1]/(q_p*self.df[fi-1]) + sig_qf_p - sig_qf_p/self.dz[zi-1]*(self.chi[zi,fi]-self.z_mat[zi,fi])
        ER = np.array([eq1,eq2,eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([0,0,self.params['rhoE']*self.z_mat[zi,fi] + (self.chi[zi,fi]-self.z_mat[zi,fi])*self.params['rhoH'] + 1/self.params['kappa']])
        QN[1,:] = np.array([1-(self.chi[zi,fi]-self.z_mat[zi,fi])/self.dz[zi-1] + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.chi[zi,fi]-self.z_mat[zi,fi]), 0, -sig_qk_p*(self.q[zi-1,fi]/q_p**2)*(self.chi[zi,fi]-self.z_mat[zi,fi])/self.dz[zi-1]])
        if fi==0:
            QN[2,:] = np.array([0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.chi[zi,fi]-self.z_mat[zi,fi]), -sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(self.chi[zi,fi]-self.z_mat[zi,fi])])
        else:
            QN[2,:] = np.array([0,1-1/self.dz[zi-1]*(self.chi[zi,fi]-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,fi]), -sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(self.chi[zi,fi]-self.z_mat[zi,fi]) - self.sig_f[zi,fi]*self.q[zi,fi-1]/(q_p**2 * self.df[fi-1])])
      
        EN = np.array([sig_qk_p,sig_qf_p,q_p]) - np.linalg.solve(QN,ER)
        del ER,QN
        return EN
    def pickle_stuff(self,object_name,filename):
                    with open(filename,'wb') as f:
                        dill.dump(object_name,f)
    
    def solve(self,pde='True'):
        self.psi[0,:]=0
        self.q[0,:] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[0,:]*(self.f-self.params['aH'])))/(1 + self.params['kappa']*(self.params['rhoH'] + self.z[0] * (self.params['rhoE'] - self.params['rhoH'])));
        self.chi[0,:] = 0;
        self.ssq[0,:] = self.params['sigma'];
        self.ssf[0,:] = 0
        self.q0 = (1 + self.params['kappa'] * self.params['aH'])/(1 + self.params['kappa'] * self.params['rhoH']); 
        self.iota[0,:] = (self.q0-1)/self.params['kappa']
        self.sig_f = self.params['beta_f'] * (self.params['f_u'] - self.f_mat)*(self.f_mat-self.params['f_l'])

        for timeStep in range(self.maxIterations):
            self.Iter+=1            
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe_z = np.vstack([((self.logValueE[1,:]-self.logValueE[0,:])/(self.z_mat[1,:]-self.z_mat[0,:])).reshape(-1,self.Nf),(self.logValueE[2:,:]-self.logValueE[0:-2,:])/(self.z_mat[2:,:]-self.z_mat[0:-2,:]),((self.logValueE[-1,:]-self.logValueE[-2,:])/(self.z_mat[-1,:]-self.z_mat[-2,:])).reshape(-1,self.Nf)]);
            self.dLogJh_z = np.vstack([((self.logValueH[1,:]-self.logValueH[0,:])/(self.z_mat[1,:]-self.z_mat[0,:])).reshape(-1,self.Nf),(self.logValueH[2:,:]-self.logValueH[0:-2,:])/(self.z_mat[2:,:]-self.z_mat[0:-2,:]),((self.logValueH[-1,:]-self.logValueH[-2,:])/(self.z_mat[-1,:]-self.z_mat[-2,:])).reshape(-1,self.Nf)]);
            self.dLogJe_f = np.hstack([((self.logValueE[:,1]-self.logValueE[:,0])/(self.f_mat[:,1]-self.f_mat[:,0])).reshape(self.Nz,-1),(self.logValueE[:,2:]-self.logValueE[:,0:-2])/(self.f_mat[:,2:]-self.f_mat[:,0:-2]),((self.logValueE[:,-1]-self.logValueE[:,-2])/(self.f_mat[:,-1]-self.f_mat[:,-2])).reshape(self.Nz,1)]);
            self.dLogJh_f = np.hstack([((self.logValueH[:,1]-self.logValueH[:,0])/(self.f_mat[:,1]-self.f_mat[:,0])).reshape(self.Nz,1),(self.logValueH[:,2:]-self.logValueH[:,0:-2])/(self.f_mat[:,2:]-self.f_mat[:,0:-2]),((self.logValueH[:,-1]-self.logValueH[:,-2])/(self.f_mat[:,-1]-self.f_mat[:,-2])).reshape(self.Nz,1)]);
            if self.params['scale']>1:
                self.Jtilde_z = (1-self.params['gammaH'])*self.dLogJh_z - (1-self.params['gammaE'])*self.dLogJe_z + 1/(self.z_mat*(1-self.z_mat))
                self.Jtilde_f = (1-self.params['gammaH'])*self.dLogJh_f - (1-self.params['gammaE'])*self.dLogJe_f
            else:
                self.Jtilde_z = self.dLogJh_z - self.dLogJe_z + 1/(self.z_mat*(1-self.z_mat))
                self.Jtilde_f = self.dLogJh_f - self.dLogJe_f

            for fi in range(self.Nf):
                for zi in range(1,self.Nz):
                    
                    if self.psi[zi-1,fi]<1:
                        result= self.equations_region1(self.q[zi-1,fi], self.psi[zi-1,fi], self.ssq[zi-1,fi], self.ssf[zi-1,fi], zi, fi)
                        if result[0]>=1:
                            self.crisis[fi]=zi
                            self.psi[zi,fi]=1
                            self.chi[zi,fi] = np.maximum(self.z[zi],self.params['alpha'])
                            result = self.equations_region2(self.q[zi-1,fi],self.ssq[zi-1,fi],self.ssf[zi-1,fi],zi,fi)
                            self.ssq[zi,fi], self.ssf[zi,fi], self.q[zi,fi] = result[0], result[1], result[2]
                            del result
                        else:
                            self.psi[zi,fi], self.ssq[zi,fi], self.ssf[zi,fi], self.q[zi,fi] =result[0], result[1], result[2], result[3]
                            self.chi[zi,fi] = self.params['alpha'] * self.psi[zi,fi]
                            del(result)
                    else:
                        self.psi[zi,fi]=1
                        result = self.equations_region2(self.q[zi-1,fi],self.ssq[zi-1,fi],self.ssf[zi-1,fi],zi,fi)
                        self.ssq[zi,fi], self.ssf[zi,fi], self.q[zi,fi] = result[0], result[1], result[2]
                        self.chi[zi,fi] = np.maximum(self.z[zi],self.params['alpha'])
                        del result
            #fix numerical error
            self.ssf[1,:] = self.ssf[2,:]
            self.ssq[1:3,:] = self.ssq[0,:]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,self.Nf)), dtype = np.float64)
            self.crisis_flag_bound = np.array(np.tile(0,(self.Nz,self.Nf)), dtype = np.float64)
            for j in range(self.Nf): 
                self.crisis_flag[0:int(self.crisis[j]),j] = 1
            
            def last_nonzero(arr, axis, invalid_val=-1):
                '''
                not used for still useful for other purposes
                '''
                mask = arr!=0
                val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
                return np.where(mask.any(axis=axis), val, invalid_val)
            
            
            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qf[:,1:self.Nf]  = (self.q [:,1:self.Nf] - self.q [:,0:self.Nf-1])/self.df_mat; self.qf[:,0]=self.qf[:,1];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2_mat); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qff[:,2:self.Nf] = (self.q[:,2:self.Nf] + self.q[:,0:self.Nf-2] - 2.*self.q[:,1:self.Nf-1])/(self.df2_mat); self.qff[:,0]=self.qff[:,2]; self.qff[:,1]=self.qff[:,2]
            q_temp = np.row_stack((self.q[0,:],self.q,self.q[self.q.shape[0]-1,:]))
            q_temp = np.column_stack((q_temp[:,0],q_temp,q_temp[:,q_temp.shape[1]-1]))
            for fi in range(1,self.Nf):
                for zi in range(1,self.Nz):
                            self.Qfz[zi,fi]= (q_temp[zi+1,fi+1] - q_temp[zi+1,fi-1] - q_temp[zi-1,fi+1] + q_temp[zi-1,fi-1])/(4*self.df_mat[zi-1,fi-1]*self.dz_mat[zi-1,fi-1]);
            del(q_temp)
            self.qzl  = self.qz/self.q; 
            self.qfl  = self.qf /self.q; 
            self.qzzl = self.qzz/ self.q;
            self.qffl = self.qff/self.q;
            self.qfzl = self.Qfz/self.q;
            
            self.iota = (self.q-1)/self.params['kappa']
            self.theta = self.chi/self.z_mat
            self.thetah = (1-self.chi)/(1-self.z_mat)
            self.theta[0] = self.theta[1]
            self.thetah[0] = self.thetah[1]
            
            
            self.consWealthRatioE = self.params['rhoE']
            self.consWealthRatioH = self.params['rhoH']
            self.sig_zk = self.z_mat*(self.theta-1)*self.ssq
            self.sig_zf = self.z_mat*(self.theta-1)*self.ssf
            self.sig_jk_e = self.dLogJe_z*self.sig_zk
            self.sig_jf_e = self.dLogJe_f*self.sig_f + self.dLogJe_z*self.sig_zf
            self.sig_jk_h = self.dLogJh_z*self.sig_zk
            self.sig_jf_h = self.dLogJh_f*self.sig_f + self.dLogJh_z*self.sig_zf
            self.priceOfRiskE_k = -(1-self.params['gammaE'])*self.sig_jk_e + self.sig_zk/self.z_mat + self.ssq + (self.params['gammaE']-1)*self.params['sigma']
            self.priceOfRiskE_f = -(1-self.params['gammaE'])*self.sig_jf_e + self.sig_zf/self.z_mat + self.ssf
            self.priceOfRiskH_k = -(1-self.params['gammaH'])*self.sig_jk_h - 1/(1-self.z_mat)*self.sig_zk + self.ssq + self.params['gammaH']*self.params['sigma']
            self.priceOfRiskH_f = -(1-self.params['gammaH'])*self.sig_jf_h - 1/(1-self.z_mat)*self.sig_zf + self.ssf
            self.priceOfRiskE_hat1 = self.priceOfRiskE_k + self.params['corr']*self.priceOfRiskE_f
            self.priceOfRiskE_hat2 = self.params['corr']* self.priceOfRiskE_k + self.priceOfRiskE_f
            self.priceOfRiskH_hat1 = self.priceOfRiskH_k + self.params['corr']*self.priceOfRiskH_f
            self.priceOfRiskH_hat2 = self.params['corr']* self.priceOfRiskH_k + self.priceOfRiskH_f
            
            self.rp = self.ssq*self.priceOfRiskE_hat1 + self.ssf*self.priceOfRiskE_hat2
            self.rp_ = self.ssq*self.priceOfRiskH_hat1 + self.ssf*self.priceOfRiskH_hat2
            self.rp_1 = self.params['alpha']*self.rp + (1-self.params['alpha'])*self.rp_
            
            self.mu_z = self.z_mat*( (self.f_mat - self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*(self.ssq*(self.priceOfRiskE_hat1 - self.ssq) + self.ssf*(self.priceOfRiskE_hat2 - self.ssf) - 2* self.params['corr'] * self.ssq*self.ssf ) + (1-self.params['alpha'])*(self.ssq*(self.priceOfRiskE_hat1 - self.priceOfRiskH_hat1) + self.ssf*(self.priceOfRiskE_hat2 - self.priceOfRiskH_hat2))) + self.params['lambda_d']*(self.params['zbar']-self.z_mat) - self.params['hazard_rate1']*self.z_mat - self.crisis_flag*self.params['hazard_rate2'] * self.z_mat
            for fi in range(self.Nf):
                crisis_temp = np.where(self.crisis_flag[:,fi]==1.0)[0][-1]+1
                try:
                    self.mu_z[crisis_temp,fi] = self.mu_z[crisis_temp-1,fi]
                except:
                    print('no crisis')
            
            self.mu_f = self.params['pi']*(self.params['f_avg'] - self.f_mat)
            self.growthRate = np.log(self.q)/self.params['kappa'] -self.params['delta']
            self.sig_zk[0]=0 #latest change
            self.ssTotal = self.ssf + self.ssq
            self.sig_zTotal = self.sig_zk + self.sig_zf
            self.priceOfRiskETotal = self.priceOfRiskE_k + self.priceOfRiskE_f
            self.priceOfRiskHTotal = self.priceOfRiskH_k + self.priceOfRiskH_f
            self.Phi = np.log(self.q)/self.params['kappa']
            self.mu_q = self.qzl*self.mu_z + self.qfl*self.mu_f + 0.5*self.qzzl*(self.sig_zk**2 + self.sig_zf**2 + 2*self.params['corr']*self.sig_zk*self.sig_zf) +\
                    0.5*self.qffl*self.sig_f**2 + self.qfzl*(self.sig_zk*self.sig_f*self.params['corr'] + self.sig_zf * self.sig_f)
            self.r = self.crisis_flag*(-self.rp_ + (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma']*(self.ssq-self.params['sigma']) + self.params['corr'] * self.params['sigma'] * self.ssf) +\
                    (1-self.crisis_flag)*(-self.rp + (self.f_mat - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq-self.params['sigma']) + self.params['corr'] * self.params['sigma'] * self.ssf)
            
            for fi in range(self.Nf):
                crisis_temp = np.where(self.crisis_flag[:,fi]==1.0)[0][-1]+1
                try:
                    self.r[crisis_temp-1:crisis_temp+2,fi] = 0.5*(self.r[crisis_temp+3,fi] + self.r[crisis_temp-2,fi]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
                except:
                    print('no crisis')
            self.A = self.psi*(self.f_mat) + (1-self.psi) * (self.params['aH'])
            self.AminusIota = self.psi*(self.f_mat - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
            self.rp_2 = self.AminusIota/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma']*(self.ssq-self.params['sigma'])+self.params['corr']*self.params['sigma']*self.ssf - self.r
            self.pd = np.log(self.q / self.AminusIota)
            self.vol = np.sqrt(self.ssq**2 + self.ssf**2)
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.f_mat - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.Jhat_e = self.Je.copy().reshape(self.Nz,self.Nf)
            self.Jhat_h = self.Jh.copy().reshape(self.Nz,self.Nf)
            self.diffusion_z = 0.5*(self.sig_zk**2 + self.sig_zf**2 + 2*self.params['corr']*self.sig_zk*self.sig_zf)
            self.diffusion_f = 0.5*(self.sig_f)**2
            if self.params['scale']>1:
                self.advection_z_e = self.mu_z 
                self.advection_f_e = self.mu_f 
                self.advection_z_h = self.mu_z
                self.advection_f_h = self.mu_f 
                self.linearTermE = -(0.5*self.params['gammaE']*(self.sig_jk_e**2 + self.sig_jf_e**2 + 2*self.params['corr']*self.sig_jk_e*self.sig_jf_e + self.params['sigma']**2) -\
                                    self.growthRate + (self.params['gammaE']-1)*(self.sig_jk_e*self.params['sigma'] + self.params['corr']*self.params['sigma']*self.sig_jf_e) -\
                                    self.params['rhoE']*(np.log(self.params['rhoE']) - np.log(self.Jhat_e) + np.log(self.z_mat*self.q)))
                self.linearTermH = -(0.5*self.params['gammaH']*(self.sig_jk_h**2 + self.sig_jf_h**2 + 2*self.params['corr']*self.sig_jk_h*self.sig_jf_h + self.params['sigma']**2) -\
                                    self.growthRate + (self.params['gammaH']-1)*(self.sig_jk_h*self.params['sigma'] + self.params['corr']*self.params['sigma']*self.sig_jf_h) -\
                                    self.params['rhoH']*(np.log(self.params['rhoH']) - np.log(self.Jhat_h) + np.log((1-self.z_mat)*self.q)))
            else:
                self.advection_z_e = self.mu_z + (1-self.params['gammaE'])*(self.params['sigma']*self.sig_zk + self.params['sigma']*self.sig_zf)
                self.advection_f_e = self.mu_f + (1-self.params['gammaE'])*self.params['corr']*self.params['sigma']*self.sig_f
                self.advection_z_h = self.mu_z + (1-self.params['gammaH'])*(self.params['sigma']*self.sig_zk + self.params['sigma']*self.sig_zf)
                self.advection_f_h = self.mu_f + (1-self.params['gammaH'])*self.params['corr']*self.params['sigma']*self.sig_f            
                self.linearTermE = (1-self.params['gammaE']) * (self.growthRate - 0.5*self.params['gammaE']*self.params['sigma']**2 +\
                                  self.params['rhoE']*(np.log(self.params['rhoE']) + np.log(self.q*self.z_mat))) -  self.params['rhoE']* np.log(self.Je)
                self.linearTermH = (1-self.params['gammaH']) * (self.growthRate - 0.5*self.params['gammaH']*self.params['sigma']**2  +\
                                  self.params['rhoH']*(np.log(self.params['rhoH']) + np.log(self.q*(1-self.z_mat)))) -   self.params['rhoH'] * np.log(self.Jh)

            self.cross_term = self.sig_zk*self.sig_f*self.params['corr'] + self.sig_zf*self.sig_f
            #Time step
            #data prep
            if pde=='True':
                if self.amax < 0.1: 
                    learning_rate = 0.01
                    layers = [3, 30, 30, 1]
                    self.dt = 1.0
                    adam_iter = 5000
                else:
                    learning_rate = 0.01
                    layers = [3, 30,30, 1]
                    self.dt = 2
                    adam_iter=5000
                tb = np.vstack((0,self.dt)).astype(np.float32)
                z_tile = np.tile(self.z,self.Nf)
                f_tile = np.repeat(self.f,self.Nz)  
                X = np.vstack((z_tile,f_tile,np.full(z_tile.shape[0],self.dt))).transpose().astype(np.float32)
                X_f = np.vstack((z_tile,f_tile,np.random.uniform(0,self.dt,z_tile.shape[0]))).transpose().astype(np.float32)
                x_star = np.vstack((z_tile,f_tile,np.full(z_tile.shape[0],0))).transpose()
                Jhat_e0 = self.Jhat_e.transpose().flatten().reshape(-1,1)
                Jhat_h0 = self.Jhat_h.transpose().flatten().reshape(-1,1)
                
                #sample more points around crisis region
                crisis_min,crisis_max = int(self.crisis.min()), int(self.crisis.max())
                X_,X_f_ = X.copy(),X_f.copy()
                Jhat_e0_,Jhat_h0_ = Jhat_e0.copy(), Jhat_h0.copy()
                
                
                def add_crisis_points(vector):
                    new_vector = vector.copy()
                    for i in range(self.Nf):
                        new_vector = np.vstack((new_vector,vector[crisis_min-1+ i*self.Nz : crisis_max+i*self.Nz,:]))
                    return new_vector
                
                def sample_boundary_points1():
                    boundary_points = []
                    for i in range(self.Nf):
                        boundary_points.append(np.arange(i*self.Nz,(i*self.Nz + 50)))
                    return boundary_points
                def sample_boundary_points2():
                    boundary_points = []
                    for i in range(1,self.Nf):
                        boundary_points.append(np.arange(i*self.Nz -50,(i*self.Nz + 1)))
                    return boundary_points
                boundary_points1= np.array(sample_boundary_points1()).flatten()
                boundary_points2= np.array(sample_boundary_points2()).flatten()
                
                
                X_,X_f_,Jhat_e0_,Jhat_h0_ = add_crisis_points(X),add_crisis_points(X_f),add_crisis_points(Jhat_e0),add_crisis_points(Jhat_h0)
                diffusion_z, diffusion_f, advection_z_e, advection_f_e = add_crisis_points(self.diffusion_z.transpose().reshape(-1,1)),add_crisis_points(self.diffusion_f.transpose().reshape(-1,1)),add_crisis_points(self.advection_z_e.transpose().reshape(-1,1)),add_crisis_points(self.advection_f_e.transpose().reshape(-1,1))    
                advection_z_h, advection_f_h = add_crisis_points(self.advection_z_h.transpose().reshape(-1,1)),add_crisis_points(self.advection_f_h.transpose().reshape(-1,1))
                cross_term, linearTermE, linearTermH = add_crisis_points(self.cross_term.transpose().reshape(-1,1)),add_crisis_points(self.linearTermE.transpose().reshape(-1,1)),add_crisis_points(self.linearTermH.transpose().reshape(-1,1))
                crisisPointsLength = X_.shape[0]-X.shape[0]
                
                np.random.seed(0)
                idx1 = np.random.choice(X_.shape[0],2500,replace=False)
                idx2 = np.random.choice(np.arange(X_.shape[0]-crisisPointsLength,X_.shape[0]),500,replace=True)
                idx3 = np.random.choice(boundary_points1, 200,replace=True)
                idx4 = np.random.choice(boundary_points2, 200, replace=True)
                idx = np.hstack((idx1,idx2,idx3,idx4))
                #idx = np.arange(0,X.shape[0])
                X_, X_f_, Jhat_e0_, Jhat_h0_ = X_[idx], X_f_[idx], Jhat_e0_[idx], Jhat_h0_[idx]
                diffusion_z_tile = diffusion_z.reshape(-1)[idx]
                diffusion_f_tile = diffusion_f.reshape(-1)[idx]
                advection_z_e_tile = advection_z_e.reshape(-1)[idx]
                advection_f_e_tile = advection_f_e.reshape(-1)[idx]
                advection_z_h_tile = advection_z_h.reshape(-1)[idx]
                advection_f_h_tile = advection_f_h.reshape(-1)[idx]
                cross_term_tile = cross_term.reshape(-1)[idx]
                linearTermE_tile = linearTermE.reshape(-1)[idx]
                linearTermH_tile = linearTermH.reshape(-1)[idx]
                
                #sovle the PDE
                model_E = nnpde_informed(-linearTermE_tile.reshape(-1,1), advection_z_e_tile.reshape(-1,1),advection_f_e_tile.reshape(-1,1), diffusion_z_tile.reshape(-1,1),diffusion_f_tile.reshape(-1,1),cross_term_tile.reshape(-1,1), Jhat_e0_.reshape(-1,1).astype(np.float32),X_,layers,X_f_,self.dt,tb,learning_rate,adam_iter)
                model_E.train()
                newJeraw = model_E.predict(x_star)
                model_E.sess.close()
                newJe = newJeraw.transpose().reshape(self.Nf,self.Nz).transpose()
                del model_E
                
                
                model_H = nnpde_informed(-linearTermH_tile.reshape(-1,1), advection_z_h_tile.reshape(-1,1),advection_f_h_tile.reshape(-1,1), diffusion_z_tile.reshape(-1,1),diffusion_f_tile.reshape(-1,1),cross_term_tile.reshape(-1,1), Jhat_h0_.reshape(-1,1).astype(np.float32),X_,layers,X_f_,self.dt,tb,learning_rate,adam_iter)
                model_H.train()
                newJhraw = model_H.predict(x_star)
                model_H.sess.close()
                newJh = newJhraw.transpose().reshape(self.Nf,self.Nz).transpose()
                
                self.ChangeJe = np.abs(newJe - self.Je)
                self.ChangeJh = np.abs(newJh - self.Jh)
                if self.params['scale']>1: cutoff = 1
                else: cutoff=10
                self.relChangeJe = np.abs((newJe[cutoff:-cutoff,:] - self.Je[cutoff:-cutoff,:]) / self.Je[cutoff:-cutoff,:])
                self.relChangeJh = np.abs((newJh[cutoff:-cutoff,:] - self.Jh[cutoff:-cutoff,:]) / self.Jh[cutoff:-cutoff,:])
                #break if nan values
                if np.sum(np.isnan(newJe))>0 or np.sum(np.isnan(newJh))>0:
                    print('NaN values found in Value function')
                    break
                self.Jh = newJh
                self.Je = newJe
                if self.params['scale']>1:
                    self.amax = np.maximum(np.amax(self.ChangeJe),np.amax(self.ChangeJh))
                else:
                    self.amax = np.maximum(np.amax(self.relChangeJe),np.amax(self.relChangeJh))
                del model_H
                
                if self.amax < self.convergenceCriterion:
                    self.converged = 'True'
                    break
                elif len(self.amax_vec)>1 and np.abs(self.amax - self.amax_vec[-1])>0.5:
                    print('check inner loop. amax error is very large: ',self.amax)
                    break
                print('Iteration number and Absolute max of relative error: ',self.Iter,',',self.amax)
                self.amax_vec.append(self.amax)
                self.pickle_stuff(self,'model2D' + '.pkl') 
            
if __name__ =="__main__":
    params={'rhoE': 0.05, 'rhoH': 0.05, 'aH': 0.02,
            'alpha':0.65, 'kappa':5, 'delta':0.05, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':4, 'gammaH':4, 'corr':1,
             'pi' : 0.01, 'f_u' : 0.2, 'f_l' : 0.1, 'f_avg': 0.15,
            'hazard_rate1' :0.06, 'hazard_rate2':0.35,'scale':2}
    params['beta_f'] = 0.25/params['sigma']

    ext = model_nnpde(params)
    ext.solve(pde='True')  
    ext.plots_()
    if False:
        def pickle_stuff(object_name,filename):
            with open(filename,'wb') as f:
                dill.dump(object_name,f)
        pickle_stuff(ext,  'ext_mac' + '.pkl')
    
    
    