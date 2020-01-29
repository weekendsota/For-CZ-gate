import numpy as np
import qutip as qt
import scipy
from scipy import constants

pi = np.pi
e = constants.e # [C]
h = constants.h # [m^2 kg/s]
hbar = constants.hbar

Nq=3
qFreq1 = 6.968*(2*pi)
qFreq2max = 7.877 *(2*pi)
qAnhar1 =  0.224 *(2*pi)
qAnhar2 = 0.312 *(2*pi)
qFreq20 = qFreq1+qAnhar2

def ket(Nq, i):
    return qt.basis(Nq, i)

def Hq(Nq, qFreq, qAnhar):
    Hqs = 0
    eigenFreq_list = [0,qFreq,2*qFreq-qAnhar]
    for i in range(Nq):
        Hqs = Hqs + eigenFreq_list[i] * ( ket(Nq, i) * ket(Nq, i).dag() )
    return Hqs

def c(Nq):
    cc = 0
    for i in range(Nq-1):
        cc = cc + np.sqrt(i+1) * ( ket(Nq, i) * ket(Nq, i+1).dag() )
    return cc

c1=c(Nq)
c2=c(Nq)

Iq1 = qt.qeye(Nq)
Iq2 = qt.qeye(Nq)

ini_coeff = [0,1e-9,0,1e-9,1,0,0,0,0] # 11

ini_state = ini_coeff[0]*qt.tensor(ket(Nq,0), ket(Nq,0)) \
            + ini_coeff[1]*qt.tensor(ket(Nq,0), ket(Nq,1)) \
            + ini_coeff[2]*qt.tensor(ket(Nq,0), ket(Nq,2)) \
            + ini_coeff[3]*qt.tensor(ket(Nq,1), ket(Nq,0)) \
            + ini_coeff[4]*qt.tensor(ket(Nq,1), ket(Nq,1)) \
            + ini_coeff[5]*qt.tensor(ket(Nq,1), ket(Nq,2)) \
            + ini_coeff[6]*qt.tensor(ket(Nq,2), ket(Nq,0)) \
            + ini_coeff[7]*qt.tensor(ket(Nq,2), ket(Nq,1)) \
            + ini_coeff[8]*qt.tensor(ket(Nq,2), ket(Nq,2))

q1_lab = Hq(Nq, qFreq1, qAnhar1)
Hq1_lab = qt.tensor(q1_lab, Iq2)
    
rot2 = Hq(Nq, 0, qAnhar2)
q2Freqs = qt.qdiags(np.arange(0,Nq,1),0)
Hq2_t_ind = qt.tensor(Iq1, rot2) #Hq2_rot(constant term)
Hq2_t_dep = qt.tensor(Iq1, q2Freqs) #Hq2_rot(modulation term)

taus_list=np.linspace(10,210,201)
intterm= ( qt.tensor(c1, c2.dag()) + qt.tensor(c1.dag(), c2) )

norm = np.dot(ini_coeff, ini_coeff) 

def PhaseChange(state_list):

    final00 = [0] * len(state_list)
    final01 = [0] * len(state_list)
    final02 = [0] * len(state_list)
    final10 = [0] * len(state_list)
    final11 = [0] * len(state_list)
    final12 = [0] * len(state_list)
    final20 = [0] * len(state_list)
    final21 = [0] * len(state_list)
    final22 = [0] * len(state_list)
    finalAdia = [0] * len(state_list)

    pop01 = [0] * len(state_list) # population of the state |01>
    pop10 = [0] * len(state_list) # population of the state |10>
    pop11 = [0] * len(state_list)# population of the state |11>
    pop02 = [0] * len(state_list)# population of the state |02>

    phase00 = [0] * len(state_list)
    phase10 = [0] * len(state_list)
    phase01 = [0] * len(state_list)
    phase11 = [0] * len(state_list)
    phaseAdia = [0] * len(state_list)
    phaseDiff = [0] * len(state_list)
        
    for i in range(len(state_list)):
            
        final00[i] = state_list[i][:][0]
        final01[i] = state_list[i][:][1]
        final02[i] = state_list[i][:][2]
        final10[i] = state_list[i][:][3]
        final11[i] = state_list[i][:][4]
        final12[i] = state_list[i][:][5]
        final20[i] = state_list[i][:][6]
        final21[i] = state_list[i][:][7]
        final22[i] = state_list[i][:][8]
        finalAdia[i] = state_list[i][:][2] + state_list[i][:][4] # eigenstate along the adiabatic line
        
        pop01[i] = norm * np.absolute(final01[i])**2 # the population is square of the magnitude.
        pop10[i] = norm * np.absolute(final10[i])**2
        pop11[i] = norm * np.absolute(final11[i])**2
        pop02[i] = norm * np.absolute(final02[i])**2
        
        phase00[i] = np.angle(final00[i]) / pi
        phase01[i] = np.angle(final01[i]) / pi
        phase10[i] = np.angle(final10[i]) / pi
        phase11[i] = np.angle(final11[i]) / pi
        phaseAdia[i] = np.angle(finalAdia[i]) / pi
        phaseDiff[i] = phaseAdia[i] - phase10[i] - phase01[i]
        
        #phaseDiff[i] = phase11[i] - phase10[i] - phase01[i]
        
        # phase ordering
        if i > 0 and phase10[i] - phase10[i-1] < -1:
            phase10[i] = phase10[i] + 2
        if i > 0 and phase10[i] - phase10[i-1] > 1:
            phase10[i] = phase10[i] - 2
            
        if i > 0 and phaseDiff[i] - phaseDiff[i-2] < -1:
            phaseDiff[i] = phaseDiff[i] + 2
        if i > 0 and phaseDiff[i] - phaseDiff[i-2] > 1:
            phaseDiff[i] = phaseDiff[i] - 2
    
    return(phaseDiff,pop11,pop02)
