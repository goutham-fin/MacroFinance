import scipy.spatial.qhull as qhull
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp2d, interp1d

'''
This file uses different customized interpolation functions used in model simulations. 
'''
def interpolate_variable(var,points, z, f):
    def interp_weights(xy, uv,d=2):
        tri = qhull.Delaunay(xy)
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uv - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    
    def interpolate(values, vtx, wts):
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    
    
    [Y,X] = np.meshgrid(f,z)
    [Yi, Xi] = np.meshgrid(points[0],points[1])
    xy=np.zeros([X.shape[0]*X.shape[1],2])
    xy[:,0]=Y.flatten()
    xy[:,1]=X.flatten()
    uv=np.zeros([Xi.shape[0]*Xi.shape[1],2])
    uv[:,0]=Yi.flatten()
    uv[:,1]=Xi.flatten()
    
    values= var.T
    
    #Computed once and for all !
    vtx, wts = interp_weights(xy, uv)
    valuesi=interpolate(values.flatten(), vtx, wts)
    valuesi=valuesi.reshape(Xi.shape[0],Xi.shape[1])
    return valuesi

def interpolate_simple(var, points, z, f):
    X, Y = np.meshgrid(z,f)
    temp = interpolate.bisplrep(X,Y,var) 
    inter_value = interpolate.bisplev(points[0], points[1], temp)
    del temp
    return inter_value

from scipy.interpolate import griddata

def interpolate_grid(var, z, f):
    f_extended = np.linspace(f[0],f[-1],z.shape[0])
    X, Y = np.meshgrid(z,f_extended)
    points1,points2 = np.meshgrid(z,f)
    value = griddata((points1.transpose().ravel(),points2.ravel()), var.ravel(), (X,Y), method='cubic')
    value[-1,:] = value[-2,:]
    return value

def interpolate_loop(var, z, f):
    #interpolates by considering each data point. Very inefficient.
    f_extended = np.linspace(f[0],f[-1],z.shape[0])
    var_fn = np.array(np.tile(np.NaN, (var.shape[0], f_extended.shape[0])))
    var_fn = interp1d(f, var, fill_value = 'extrapolate')(f_extended)
    return var_fn



from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

def interpolate_pd(var, points, z, f):
    mesh1 = z.shape[0]*2
    tri = Delaunay(mesh1)  # Compute the triangulation
    # Perform the interpolation with the given values:
    interpolator = LinearNDInterpolator(zip(z,f), var)
    values_mesh2 = interpolator(points[0], points[1])
    return values_mesh2

def interpolate_2d(var, z, f):
    return interp2d(z,f,var, kind='linear', fill_value = 'extrapolate')
    
