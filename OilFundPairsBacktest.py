#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:53:48 2019

@author: himanshu
"""

import pandas as pd
import numpy as np
#import datetime as dt
#import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

def get_OU_params(port):
    X_x = np.sum(port.values[:-1])
    X_y = np.sum(port.values[1:])
    X_xx = np.sum((port.values[:-1])**2)
    X_xy = np.sum((port.values[:-1])*(port.values[1:]))
    X_yy = np.sum(port.values[1:]**2)
    
    n = len(port)
    dt = 1/252
    
    num = (X_y*X_xx) - (X_x*X_xy)
    den = n*(X_xx-X_xy) - ((X_x**2)-(X_x*X_y))
    theta = (num)/(den)

    num = (X_xy - (theta*X_x) - (theta*X_y) + (n*(theta**2)))
    den = X_xx - (2*theta*X_x) + (n*(theta**2))
    mu = -(1/dt)*(np.log(num/den))

    alpha = np.exp(-mu*dt)
    t1 = (2*mu)/(n*(1-(alpha**2)))
    t2 = X_yy - (2*alpha*X_xy) + ((alpha**2)*X_xx)
    t3 = 2*theta*(1-alpha)*(X_y - alpha*X_x) 
    t4 = n*(theta**2)*(1-alpha)**2
    sigma = np.sqrt(t1*(t2-t3+t4))

    return theta, mu, sigma

def avg_likelihood(params, port):
    n = len(port)
    dt = 1/252
    theta, mu, sigma = params
    
    sigma_tilde = (sigma**2)*(1-np.exp(-2*mu*dt))/(2*mu)
    sq_term = np.sum((port.values[1:] - (port.values[:-1]*(np.exp(-mu*dt))) - theta*(1-(np.exp(-mu*dt))))**2)
    l = -(0.5*np.log(2*np.pi)) - np.log(np.sqrt(sigma_tilde)) - (1/(2*n*sigma_tilde))*sq_term
    
    return l

def find_best_port(data):
    A = 1
    B = np.arange(0.001,1.001,0.001)
    l_hat = [0]*len(B)
    th = [0]*len(B)
    max_index = None
    ou_params = [None]*len(B)
    th = [None]*len(B)
    for i in range(len(B)):
        alpha = A/data.etf1.iloc[0]
        beta = B[i]/data.etf2.iloc[0]
        port = alpha*data.etf1-beta*data.etf2
        ou_params[i] = get_OU_params(port)
        th[i] = ou_params[i][0]
        l_hat[i] = avg_likelihood(ou_params[i], port) 
        if max_index is None or l_hat[i] > l_hat[max_index]:
            max_index = i
    return B[max_index], ou_params[max_index]

path = '../OilFunds'
etf1 = pd.read_csv(path+os.sep+'IXC.csv')
etf1['Date'] = pd.to_datetime(etf1['Date'], format='%d/%m/%Y')
etf1 = etf1.sort_values(['Date'])
etf1 = etf1.set_index(['Date'])

etf2 = pd.read_csv(path+os.sep+'USO.csv')
etf2['Date'] = pd.to_datetime(etf2['Date'], format='%d/%m/%Y')
etf2 = etf2.sort_values(['Date'])
etf2 = etf2.set_index(['Date'])

data = pd.concat([etf1['Close'], etf2['Close']], axis=1).dropna()
data.columns = ['etf1', 'etf2']
#data['etf1_rets'] = np.log(data['etf1']/data['etf1'].shift(1))
#data['etf2_rets'] = np.log(data['etf2']/data['etf2'].shift(1))

S0_1 = None
S0_2 = None

for i in range(252, len(data)):
    filt_data = data.iloc[i-252:i]
    B, params = find_best_port(filt_data)
    if i==252:
        S0_1 = filt_data.etf1.iloc[-1]
        S0_2 = filt_data.etf2.iloc[-1]
    
    alpha =  1/S0_1
    beta = B/S0_2
    spread = alpha*S0_1 - beta*S0_2
    
    if spread >= params[0]+(0.3*params[2]):
        pass
    elif spread <= params[0]-(0.3*params[2]):
        pass
    

