# coding: utf-8
import qutip as qt
import numpy as np
from scipy import constants

import quantum_okiba as qo

pi = np.pi
e = constants.e
h = constants.h
hbar = constants.hbar
Nq=3

cz_gate_ref=qt.qdiags([1,1,1,1,-1,1,1,1,1,1],0)
ini=qo.tensor_to_flat(qo.ini_state)
rho_ini=ini*ini.dag()
rho=cz_gate_ref.dag()*rho_ini*cz_gate_ref

def fidelity(qobj):
    phi=qobj
    #reshape to get compatible density matrix
    fphi=qo.ket(9,0)-qo.ket(9,0) #Ensure it's qobj,which all components are zero.
    
    for i in range(9):
        fphi=fphi+qo.ket(9,i)*phi[i]
    
    sigma=fphi*fphi.dag()
    mix=rho*sigma
    return(mix.tr())
