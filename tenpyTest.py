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

# StringOrder
def computeStrOrder(start: int, end: int, M):
    sites = M.lat.mps_sites()

    N = len(sites)

    StrOrder = []
    mpo=[]
    button = False
    add_button = True
    if start == 0:
        StrOrder += [sites[0].Sz]

        #so = sites[0].Sz.replace_labels(['p', 'p*'], ['p0', 'p0*'])
        mpo.append([[sites[0].Sz]])

        button = True
    else:
        StrOrder += [sites[0].Id]
        #so = sites[0].Id.replace_labels(['p', 'p*'], ['p0', 'p0*'])
        mpo.append([[sites[0].Id]])

    for i in range(1, N):

        if i == start:
            button = True
            StrOrder += [sites[i].Sz]
            #so = outer(so, sites[i].Sz.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))
            mpo.append([[sites[i].Sz]])

            continue
        if i == end:
            button = False
            StrOrder += [sites[i].Sz]
            #so = outer(so, sites[i].Sz.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))
            mpo.append([[sites[i].Sz]])
            continue

        if button:

            StrOrder += [sites[i].expsz]
            #so = outer(so, sites[i].expsz.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))
            mpo.append([[sites[i].expsz]])
        else:
            StrOrder += [sites[i].Id]
            #so = outer(so, sites[i].Id.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))
            mpo.append([[sites[i].Id]])
    sompo = MPO.from_grids(sites, mpo, bc='finite', IdL=0, IdR=-1)

    return StrOrder, sompo


num_site =16


h = 0
D = 0
hz = np.zeros(num_site)
hz[0] = h
hz[-1] = -h
# Model

M = SpinChain({"S": 1, "L": num_site, "bc_MPS": "finite",
               "Jx": 1, "Jy": 1, "Jz": 1, "D": D, "hz": hz})
sites = M.lat.mps_sites()
#add op
sz = sites[0].Sz.to_ndarray()
expsz = expm(np.pi * 1.0j * sz)
sites[0].add_op(name='expsz', op=expsz)
#DMRG
psi = MPS.from_product_state(M.lat.mps_sites(), [1] * num_site, "finite")

dmrg_params = {"trunc_params": {"chi_max": 40, "svd_min": 1.e-10}}
info = dmrg.run(psi, M, dmrg_params)
#truncation error]
chi_list=list(np.arange(2,12,2))+list(np.arange(10,190,10))
trunError=[]
for chi in chi_list:
    psi = MPS.from_product_state(M.lat.mps_sites(), [1] * num_site, "finite")

    dmrg_params = {"trunc_params": {"chi_max": chi, "svd_min": 1.e-10}}
    info = dmrg.run(psi, M, dmrg_params)
    error=0
    for err in info['bond_statistics']['err']:

        error+=err.eps
    trunError.append(error)
    print(f'dmrg done for chi{chi}')

poly=np.polyfit(chi_list[0:15],trunError[0:15],deg=15)
y=np.zeros(len(chi_list))
for j in range(len(chi_list)):

    for i in range(len(poly)):
        y[j]+=poly[i]*pow(chi_list[j],i)

plt.plot(chi_list,trunError,chi_list,y)
plt.show()





#S = psi.entanglement_entropy()

#spcta = psi.entanglement_spectrum()
StrOrder, sompo = computeStrOrder(0, num_site - 1, M)
a=sompo.expectation_value(psi=psi)




'''
# Spring Order
L = np.arange(1, num_site)
Gl = []
for i in L:
    StrOrder, sompo = computeStrOrder(0, i, M)

    gl =sompo.expectation_value(psi)
    Gl.append(-gl)
plt.scatter(L, Gl)
plt.show()
'''

'''
Dlist = np.arange(0, 2.2, 0.1)
Slist = []  # Entanglement Entropy
for D in Dlist:
    M = SpinModel({"S": 1, "L": num_site, "bc_MPS": "finite",
                   "Jx": 1, "Jy": 1, "Jz": 1, "D": D,"hz":hz})
    psi = MPS.from_product_state(M.lat.mps_sites(), [1] * num_site, "finite")
    dmrg_params = {"trunc_params": {"chi_max": 30, "svd_min": 1.e-10}}
    info = dmrg.run(psi, M, dmrg_params)
    S=psi.entanglement_entropy()
    L=psi.correlation_length()

    Slist.append(S[int(num_site/2-1)])
    corrL.append(L)
plt.scatter(Dlist,Slist,c='blue',s=150)
plt.show()
'''












