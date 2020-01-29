import csv
import os
import pprint
import math
import numpy as np
import qutip as qt
import scipy
from scipy import constants
from scipy import signal, interpolate
from scipy import integrate
import sympy as sym

import matplotlib.pyplot as plt
import sys
import quantum_okiba as qo
from tqdm import tqdm

iDir = os.path.abspath(os.path.dirname(__file__))

pi = np.pi
e = constants.e # [C]
h = constants.h # [m^2 kg/s]
hbar = constants.hbar

condition_lists=[]

with open(iDir+'/further_conditions.csv') as f:
    reader = csv.reader(f)
    condition_lists=[row for row in reader]

opts = qt.solver.Options(nsteps=10000)

def MW_shaped(t,args):
    
    amp = args['mwamp']
    shape = args['shape'] 
    if int(t*100)>=len(shape):
        n=len(shape)-1
    else:
        n=int(t*100)
    return amp * shape[n]

#for loop
count=len(condition_lists)

for condition in tqdm(condition_lists):
    
    #appropriate
    Cc,the_f,coeff2,tg = float(condition[0]),float(condition[1]),float(condition[2]),float(condition[3])

    coeff1=the_f*(pi/2)

    J0=4*(Cc*1e-15)*(e**2)/(hbar*((93.126*1e-15)*(67.966*1e-15)-(Cc*1e-15)**2))*1e-9
    #qo.intterm=(qt.tensor(c1, c2.dag()) + qt.tensor(c1.dag(), c2))
    Hint = J0*(2*pi)*qo.intterm

    lambdas=[0.1,(coeff1-0.1)/2,coeff2] 
    
    def co_optim(taug):

        tau_list=np.linspace(0,taug,int(tg*100))

        def theta_tau(t): #θを制御するスレピアン関数
            theta=lambdas[0]
            for i in range(1,len(lambdas)):
                theta=theta+lambdas[i]*(1-np.cos(2*pi*t*i/taug))
            return theta

        def sintheta_tau(t):
            theta=theta_tau(t)
            sintheta=np.sin(theta)
            return sintheta
            
        t_tau=[]
        for i in range(len(tau_list)):
            t_tau.append(integrate.quad(sintheta_tau,0,tau_list[i])[0])
        
        return(t_tau,theta_tau(tau_list),taug/t_tau[-1])

    theta_list=co_optim(tg)[1] #静止座標系のθ(0)~θ(tg)
    t_list=np.linspace(0,tg,len(theta_list)) #theta_listの長さに合うように等分割した時間のリスト
    #t_list=co_optim(tg*ratio)[0] #静止座標系の0~tg

    f1=interpolate.interp1d(t_list,theta_list,fill_value="extrapolate") #θ(t) calculated
    y1=f1(t_list)
    slepian_like = 2*np.sqrt(2)*J0*(2*pi)/np.tan(y1)+qo.qFreq20

    args = {'mwamp':1.0,'shape':slepian_like}
    H_rot = [qo.Hq1_lab + qo.Hq2_t_ind + Hint, [qo.Hq2_t_dep, MW_shaped]]
    res = qt.sesolve(H_rot, qo.ini_state, t_list, e_ops=[], args=args, options=opts, progress_bar=None)
    q_state_list=res.states

    pD,p11,p02=qo.PhaseChange(q_state_list)
    fig=plt.figure(figsize=(6.0,6.0))
    plt.plot(t_list,pD,label='|11> Phase')
    plt.plot(t_list,p11,label='|11> Population')
    plt.plot(t_list,p02,label='|02> Population')
    plt.xlabel('t[ns]')
    plt.legend()
    plt.grid(True)
    plt.savefig(iDir+"/Pics_furtherlook/Cc="+str(Cc)+"/Cc="+str(Cc)+",θf="+str(the_f)+"×π_2,J="+str(round(J0,3))+"[GHz_2π],tg="+str(tg)+"[ns].jpg")