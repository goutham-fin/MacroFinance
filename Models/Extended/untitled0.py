#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:59:06 2020

@author: GG
"""
import dill
import matplotlib.pyplot as plt
def read_pickle(filename):
            with open(str(filename) + '.pkl', 'rb') as f:
                return dill.load(f)
ext = read_pickle('model2D')

plt.plot(ext.Je);
plt.figure()
plt.plot(ext.Jh);
print(ext.amax_vec);
plt.figure()
plt.plot(ext.ChangeJe);
plt.figure()
plt.plot(ext.ChangeJe);
plt.figure()
plt.plot(ext.relChangeJe);
plt.figure()
plt.plot(ext.relChangeJe);