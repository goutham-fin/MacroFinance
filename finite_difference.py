import numpy as np
from scipy.sparse import spdiags
from scipy import sparse

def hjb_implicit_bruSan(A,B1,C11,J,dt,dzt,Nz,Nf):
    dzt_mat = np.tile(dzt,(Nf,1)).transpose()
    LD =  -C11[1:-1]*(dt/dzt_mat**2);
    UD = -dt* (B1[1:-1]/dzt_mat + C11[1:-1]/dzt_mat**2 )
    D = 1 - dt * (A[1:-1] - B1[1:-1]/dzt_mat - C11[1:-1]/dzt_mat**2)
    #RHS =  np.array(np.row_stack((J[1:,],J[-1,:])),dtype=np.float64)   - A*dt
    RHS =  np.array(J[1:-1,:],dtype=np.float64) 
    RHS[0,:] = RHS[0,:] + J[0,:] * LD[0] 
    RHS[-1,:] = RHS[-1,:] + J[-1,:] * UD[-1]
    J_new = np.zeros([J.shape[0]])

    Mat = spdiags(np.hstack([np.nan, UD[0:-1,0]]),1,Nz,Nz) + spdiags(D[:,0],0,Nz,Nz) + spdiags(np.hstack([LD[1:,0],np.nan]),-1,Nz,Nz)
    b=np.array(RHS)
    J_new[1:-1] = np.linalg.solve(Mat.toarray(), b).reshape(-1)
    
    del(RHS)
    return J_new[1:-1]
    

def hjb_explicit_bruSan(A,B1,C11,J,dt,dzt,Nz,Nf):
    J_new = np.zeros([J.shape[0],J.shape[1]])
    J_temp = J
    
    #J_temp = np.row_stack((J[0,:],J,J[J.shape[0]-1,:]))
    #J_temp = np.column_stack((J_temp[:,0],J_temp,J_temp[:,J_temp.shape[1]-1]))
    for i in range(0,Nz):
        for j in range(0,Nf):
            J_new[i,j] = -dt * (A[i,j] + J_temp[np.min((i+1,J.shape[0]-1)),j] * (B1[i,j]/dzt[i] + C11[i,j] * 0.5/dzt[i]**2) - \
                         J_temp[i,j] * (1 + B1[i,j] /dzt[i] + C11[i,j]/dzt[i]**2) + J_temp[np.max((i-1,0)),j] * (C11[i,j] * 0.5 / dzt[i]**2)) + J_temp[i,j]                 
                        
            
    #J_new[-1,:] = J_new[-2,:]
    del(J_temp)
    return J_new


def hjb_implicit_bruSan_upwind(z, J, dt, C11, B1, A, D, boundaryLeft, boundaryRight):
    diff1L_l = np.vstack([np.nan,-1/(z[1:-1]-z[0:-2]),np.nan]) 
    diff1M_l = np.vstack([np.nan, 1/(z[1:-1] - z[0:-2]), np.nan])
    diff1R_l = np.hstack([np.nan, np.zeros(z.shape[0]-2),np.nan]).reshape(-1,1)
    diff1L_r = np.hstack([np.nan, np.zeros(z.shape[0]-2), np.nan]).reshape(-1,1)
    diff1M_r = np.vstack([np.nan, -1/(z[2:]- z[1:-1]), np.nan])
    diff1R_r = np.vstack([np.nan, 1/(z[2:]- z[1:-1]), np.nan])
    #impose upwind condition
    posCoeff = B1>=0
    negCoeff = B1<0
    diff1L = posCoeff.reshape(-1,1) * diff1L_r.reshape(-1,1) + negCoeff.reshape(-1,1) * diff1L_l.reshape(-1,1)
    diff1M = posCoeff.reshape(-1,1) * diff1M_r.reshape(-1,1) + negCoeff.reshape(-1,1) * diff1M_l.reshape(-1,1)
    diff1R = posCoeff.reshape(-1,1) * diff1R_r.reshape(-1,1) + negCoeff.reshape(-1,1) * diff1R_l.reshape(-1,1)
    
    #second order differentiation
    diff2L = np.vstack([np.nan, 2/(z[2:] - z[0:-2])/(z[1:-1] - z[0:-2]), np.nan])
    diff2M = np.vstack([np.nan, -2/(z[2:] - z[0:-2])*(1/(z[2:] - z[1:-1]) + 1/(z[1:-1] - z[0:-2])), np.nan])
    diff2R = np.vstack([np.nan, 2/(z[2:] - z[0:-2])/(z[2:] - z[1:-1]), np.nan])
    
    J_new = np.zeros([J.shape[0]])
    J_new[0] = boundaryLeft
    J_new[-1] = boundaryRight
    
    
    D = A[1:-1].reshape(-1,1) + B1[1:-1].reshape(-1,1) * diff1M[1:-1].reshape(-1,1) + C11[1:-1].reshape(-1,1) *diff2M[1:-1].reshape(-1,1)
    LD = B1[1:-1].reshape(-1,1) * diff1L[1:-1].reshape(-1,1) + C11[1:-1].reshape(-1,1) * diff2L[1:-1].reshape(-1,1)
    UD = B1[1:-1].reshape(-1,1) * diff1R[1:-1].reshape(-1,1) + C11[1:-1].reshape(-1,1) * diff2R[1:-1].reshape(-1,1)
    
    #RHS =  np.array(np.row_stack((J[1:,],J[-1,:])),dtype=np.float64)   - A*dt
    RHS =  np.array(J[1:-1,:],dtype=np.float64) 
    RHS[0,:] = RHS[0,:] + dt * LD[0] * J_new[0]
    RHS[-1,:] = RHS[-1,:] + dt * UD[-1] * J_new[-1]
    
    
    Mat = spdiags(np.vstack([np.nan, UD[0:-1]]).transpose(),1,z.shape[0]-2,z.shape[0]-2) + spdiags(D.transpose(),0,z.shape[0]-2,z.shape[0]-2) + spdiags(np.vstack([LD[1:],np.nan]).transpose(),-1,z.shape[0]-2,z.shape[0]-2)
    b=np.array(RHS)
    J_new[1:-1] = np.linalg.solve((sparse.eye(z.shape[0]-2) - dt* Mat.toarray()), b).reshape(-1)
    
    del(RHS)
    return J_new

def hjb_implicit_bruSan_policy(X, R, MU, S, G, V, dt):
    N = X.shape[0];
    dX = X[1:N] - X[0:N-1]

    S0 = np.zeros([N,1])
    S0[1:N-1] = S[1:N-1].reshape(N-2,1)**2/(dX[0:N-2] + dX[1:N-1]).reshape(N-2,1)
    DU = np.zeros([N,1])
    DU[1:N] = -(np.maximum(MU[0:N-1],np.zeros([N-1,1])) + S0[0:N-1].reshape(N-1,1))/dX.reshape(N-1,1)* dt

    DD = np.zeros([N-1,1])
    DD = -(np.maximum(-MU[1:N],np.zeros([N-1,1])) + S0[1:N].reshape(N-1,1))/dX.reshape(N-1,1)*dt

    D0 = (1-dt)*np.ones([N,1]) + dt*R
    D0[0:N-1] = D0[0:N-1] - DU[1:N]
    D0[1:N] = D0[1:N] - DD

    A = sparse.spdiags(D0.transpose(),0,N,N) + sparse.spdiags(DU.transpose(),1,N,N) + sparse.spdiags(DD.transpose(),-1,N,N)
    F = np.linalg.solve(A.toarray(), V*(1-dt))
    return F


