#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import logging, os 
os.system('clear')
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np 
import pandas as pd 
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#from keras import backend as K


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



if __name__ =='__main__':
    J0 = Je0.astype(np.float32).reshape(-1,1)
    linearTerm = linearTermE_tile
    advection_z,advection_f,diffusion_z,diffusion_f,cross_term = advection_z_tile,advection_f_tile,diffusion_z_tile,diffusion_f_tile,cross_term_tile
    model_ = nnpde_informed(linearTerm,advection_z,advection_f,diffusion_z,diffusion_f,cross_term,J0,X,layers,X_f,dt,tb)
    model_.train()
    
    Jnew, Jnew_error = model_.predict(x_star)
    #plt.plot(z, Jnew)
    #plt.plot(z, Je)
    
    
    
    
