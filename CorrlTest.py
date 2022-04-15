from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinSite
from tenpy.models.spins import SpinModel,SpinChain
from tenpy.algorithms import dmrg
import numpy as np
from scipy.linalg import expm
from tenpy.linalg.np_conserved import outer,tensordot
from tenpy.networks.mpo import MPO
import matplotlib.pyplot as plt
from  tenpy.algorithms import tebd

from scipy.optimize import curve_fit
#iDMRG
N=2
num_site=64


h = 0
D = 1
hz = np.zeros(num_site)
hz[0] = h
hz[-1] = -h
# iDMRG

iM = SpinChain({"S": 1, "L": N, "bc_MPS": "infinite",
               "Jx": 1, "Jy": 1, "Jz": 1, "D": D, "hz": 0})
Ipsi = MPS.from_product_state(iM.lat.mps_sites(), [1] * N, "infinite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}}
Iinfo = dmrg.run(Ipsi, iM, dmrg_params)
icorrL=Ipsi.correlation_length()
print(f'correlation length in iDMRG is{icorrL}')
#计算关联长度
def computeCorrL(L,psi:MPS,start):
    r=np.arange(0,L-start-1,1)
    corrSz=[]
    corrFun=psi.correlation_function(ops1='Sz',ops2='Sz',sites1=[start])
    for site in r:
        corrSz.append((np.abs(corrFun[0,site+start])))
    #fit
    def func(x, a, b, c):
        return a * np.exp(-b* x) +c

    para,cov=curve_fit(func,r,corrSz)
    corrL=1/para[1]
    yfit=[func(a,*para) for a in r]
    fig,ax=plt.subplots()

    s1=ax.plot(r,corrSz,'b',label='result')
    s2=ax.plot(r,yfit,'g',label='curve fit')
    ax.set_xlabel('L')
    ax.set_ylabel('Correlation Function')
    plt.legend()

    plt.show()
    return corrL

l=num_site
M = SpinChain({"S": 1, "L": l, "bc_MPS": "finite",
                   "Jx": 1, "Jy": 1, "Jz": 1, "D": D, "hz": hz})
psi = MPS.from_product_state(M.lat.mps_sites(), [1] * l, "finite")
dmrg_params = {"trunc_params": {"chi_max": 40, "svd_min": 1.e-10}}
info = dmrg.run(psi, M, dmrg_params)

corrl=computeCorrL(l,psi=psi,start=2)
print(f'correlation length used DMRG is {corrl}')
#D-corrL
'''
Dlist=np.arange(0,2.1,0.1)
IcorrL=[]
corrL=[]
for d in Dlist:
    # iDrmg
    iM = SpinChain({"S": 1, "L": N, "bc_MPS": "infinite",
                    "Jx": 1, "Jy": 1, "Jz": 1, "D": d, "hz": 0})
    Ipsi = MPS.from_product_state(iM.lat.mps_sites(), [1] * N, "infinite")
    dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}}
    Iinfo = dmrg.run(Ipsi, iM, dmrg_params)
    icorrL = Ipsi.correlation_length()
    IcorrL.append(icorrL)
    #Dmrg

    M = SpinChain({"S": 1, "L": num_site, "bc_MPS": "finite",
                   "Jx": 1, "Jy": 1, "Jz": 1, "D": d, "hz": hz})
    psi = MPS.from_product_state(M.lat.mps_sites(), [1] * num_site, "finite")
    dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}}
    info = dmrg.run(psi, M, dmrg_params)
    corrl = computeCorrL(l, psi=psi)
    corrL.append(corrl)


    print(f'dmrg done for D={d}')

#画图
fig,ax=plt.subplots(2)
ax[0].plot(Dlist,IcorrL)
ax[1].plot(Dlist,corrL)
plt.show()
'''






