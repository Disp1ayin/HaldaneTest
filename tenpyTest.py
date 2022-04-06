from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinSite
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
import numpy as np
from scipy.linalg import expm
from tenpy.linalg.np_conserved import outer
from tenpy.networks.mpo import MPO
import matplotlib.pyplot as plt


# StringOrder
def computeStrOrder(start: int, end: int, M):
    sites = M.lat.mps_sites()
    print(sites)
    N = len(sites)

    StrOrder = []
    button = False
    add_button = True
    so = sites[0].Id.replace_labels(['p', 'p*'], ['p0', 'p0*'])

    for i in range(1, N):

        if i == start - 1:
            button = True
            StrOrder += [sites[i].Sz]
            so = outer(so, sites[i].Sz.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))

            continue
        if i == end - 1:
            button = False
            StrOrder += [sites[i].Sz]
            so = outer(so, sites[i].Sz.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))
            continue

        if button:
            if add_button:
                sz = sites[i].Sz.to_ndarray()
                expsz = expm(np.pi * 1j * sz)
                sites[i].add_op(name='expsz', op=expsz)
                add_button = False

            StrOrder += [sites[i].expsz]
            so = outer(so, sites[i].expsz.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))
        else:
            StrOrder += [sites[i].Id]
            so = outer(so, sites[i].Id.replace_labels(['p', 'p*'], [f'p{i}', f'p{i}*']))

    return StrOrder, so


num_site = 8

Dlist = np.arange(0, 2.2, 0.1)
Slist = []  # Entanglement Entropy
corrL = []
h = 0.5
D = 1.0
hz = np.zeros(num_site)
hz[0] = h
hz[-1] = -h
# DMRG

M = SpinModel({"S": 1, "L": num_site, "bc_MPS": "finite",
               "Jx": 1, "Jy": 1, "Jz": 1, "D": D, "hz": hz})

StrOrder, so = computeStrOrder(num_site / 4, 3 * num_site / 4, M)

psi = MPS.from_product_state(M.lat.mps_sites(), [1] * num_site, "finite")
dmrg_params = {"trunc_params": {"chi_max": 30, "svd_min": 1.e-10}}
info = dmrg.run(psi, M, dmrg_params)
StringOrder=psi.expectation_value(StrOrder)
StringOrder2 = psi.expectation_value(so)
S = psi.entanglement_entropy()
print(psi.expectation_value('Sz'))

spcta = psi.entanglement_spectrum()

'''
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
