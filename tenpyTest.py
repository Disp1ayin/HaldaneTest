from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
import numpy as np
import matplotlib.pyplot as plt
num_site=32

Dlist = np.arange(0, 2.2, 0.1)
Slist=[]
for D in Dlist:
    M = SpinModel({"S": 1, "L": num_site, "bc_MPS": "finite",
                   "Jx": 1, "Jy": 1, "Jz": 1, "D": D})
    psi = MPS.from_product_state(M.lat.mps_sites(), [1] * num_site, "finite")
    dmrg_params = {"trunc_params": {"chi_max": 30, "svd_min": 1.e-10}}
    info = dmrg.run(psi, M, dmrg_params)
    S=psi.entanglement_entropy()

    Slist.append(S[int(num_site/2-1)])
plt.scatter(Dlist,Slist,c='blue',s=150)
plt.show()


