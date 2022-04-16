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
num_site=16


h = 0
D = 0
hz = np.zeros(N)
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
def computeCorrL(L,psi:MPS):
    sites=np.arange(0,L,1)
    corrSz=[]
    corrFun=psi.correlation_function(ops1='Sz',ops2='Sz')
    for site in sites:
        corrSz.append(np.abs(corrFun[0,site]))
    #fit
    def func(x, a, b, c):
        return a * np.exp(-b* x) + c
    para,cov=curve_fit(func,sites,corrSz)
    corrL=1/para[1]
    return corrL


#计算关联长度随L的变化
L_list=list(np.arange(4,100,4))
corrL=[]
for l in L_list:
#DMRG
    M = SpinChain({"S": 1, "L": l, "bc_MPS": "finite",
                   "Jx": 1, "Jy": 1, "Jz": 1, "D": D, "hz": hz})
    psi = MPS.from_product_state(M.lat.mps_sites(), [1] * l, "finite")
    dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}}
    info = dmrg.run(psi, M, dmrg_params)

    corrl=computeCorrL(l,psi=psi)
    corrL.append(corrl)
    print(f'correlation length compute done for length={l}')
#画图
plt.scatter(L_list,corrL)
plt.hlines(icorrL,colors='red',xmin=0,xmax=L_list[-1])
plt.show()