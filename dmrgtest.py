"""
Example of backend independent DMRG calculation for the
Haldane Model.
H=J*S(i)*S(i+1)+D*Sz^2
"""

from typing import Type, Text
import tensornetwork as tn

from tensornetwork.ncon_interface import ncon
from sys import stdout
import numpy as np
import tensornetwork as tn
from typing import List, Union, Text, Optional, Any, Type, Tuple
from tensornetwork.backends import backend_factory
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends.abstract_backend import AbstractBackend
import matplotlib.pyplot as plt


def initialize_spin1_mps(N: int, D: int, dtype: Type[np.number], backend: Text):
    """
  Helper function to initialize an MPS for a given backend.

  Args:
    N: Number of spins.
    D: The bond dimension.
    dtype: The data type of the MPS.

  Returns:
    `tn.FiniteMPS`: A spin 1 mps for the corresponding backend.
  """

    return tn.FiniteMPS.random([3] * (N ) , [D] * (N - 1), dtype=dtype, backend=backend)


class FiniteHaldanespin1(tn.FiniteMPO):
    """

  H=J*S(i)*S(i+1)+D*Sz^2
  """

    def __init__(self,
                 Jz: np.ndarray,
                 Jxy: np.ndarray,
                 D: np.ndarray,
                 dtype: Type[np.number],
                 backend: Optional[Union[AbstractBackend, Text]] = None,
                 name: Text = 'XXZ_MPO') -> None:
        """
    Returns the MPO of the finite Haldane model.
    Args:
      Jz:  The Sz*Sz coupling strength between nearest neighbor lattice sites.
      Jxy: The (Sx*Sx + Sy*Sy) coupling strength between nearest neighbor.
        lattice sites

      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FiniteHaldane: The mpo of the finite Haldane model.
    """
        self.Jz = Jz
        self.Jxy = Jxy
        self.D = D
        N = len(D)
        mpo = []
        temp = np.zeros((1, 5, 3, 3), dtype=dtype)

        # small field at the first site For numerical stability


        # D
        temp[0, 0, 0, 0] = D[0]+0.5
        temp[0, 0, 2, 2] = D[0]-0.5

        # Sm*J/2
        temp[0, 1, 1, 0] = 2 ** 0.5 * Jxy[0] / 2.0
        temp[0, 1, 2, 1] = 2 ** 0.5 * Jxy[0] / 2.0

        # Sp*J/2
        temp[0, 2, 0, 1] = 2 ** 0.5 * Jxy[0] / 2.0
        temp[0, 2, 1, 2] = 2 ** 0.5 * Jxy[0] / 2.0

        # J*Sz
        temp[0, 3, 0, 0] = Jz[0]
        temp[0, 3, 2, 2] = -Jz[0]

        # 11
        temp[0, 4, 0, 0] = 1.0
        temp[0, 4, 1, 1] = 1.0
        temp[0, 4, 2, 2] = 1

        mpo.append(temp)
        for n in range(1, N - 1):
            temp = np.zeros((5, 5, 3, 3), dtype=dtype)
            # 11
            temp[0, 0, 0, 0] = 1.0
            temp[0, 0, 1, 1] = 1.0
            temp[0, 0, 2, 2] = 1.0
            # Sp
            temp[1, 0, 0, 1] = 2 ** 0.5
            temp[1, 0, 1, 2] = 2 ** 0.5
            # Sm
            temp[2, 0, 1, 0] = 2 ** 0.5
            temp[2, 0, 2, 1] = 2 ** 0.5
            # Sz
            temp[3, 0, 0, 0] = 1
            temp[3, 0, 2, 2] = -1
            # D
            temp[4, 0, 0, 0] = D[n]
            temp[4, 0, 2, 2] = D[n]

            # Sm*J/2
            temp[4, 1, 1, 0] = 2 ** 0.5 * Jxy[n] / 2.0
            temp[4, 1, 2, 1] = 2 ** 0.5 * Jxy[n] / 2.0
            # Sp*J/2
            temp[4, 2, 0, 1] = 2 ** 0.5 * Jxy[n] / 2.0
            temp[4, 2, 1, 2] = 2 ** 0.5 * Jxy[n] / 2.0
            # Sz*J
            temp[4, 3, 0, 0] = Jz[n]
            temp[4, 3, 2, 2] = -Jz[n]
            # 11
            temp[4, 4, 0, 0] = 1.0
            temp[4, 4, 1, 1] = 1.0
            temp[4, 4, 2, 2] = 1.0

            mpo.append(temp)
        temp = np.zeros((5, 1, 3, 3), dtype=dtype)



        # 11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        temp[0,0,2,2]=1.0

        # Sp
        temp[1, 0, 0, 1] = 2**0.5
        temp[1, 0, 1, 2] = 2 ** 0.5

        # Sm
        temp[2, 0, 1, 0] = 2**0.5
        temp[2, 0, 2, 1] = 2 ** 0.5

        # Sz
        temp[3, 0, 0, 0] = 1
        temp[3, 0, 2, 2] = -1
        # small field at the last site For numerical stability
        # D
        temp[4, 0, 0, 0] = D[-1]-0.5
        temp[4, 0, 2, 2] = D[-1]+0.5

        mpo.append(temp)
        super().__init__(tensors=mpo, backend=backend, name=name)


def initialize_Haldane_mpo(Jz: np.ndarray, Jxy: np.ndarray, D: np.ndarray,
                           dtype: Type[np.number], backend: Text):
    """
  Helper function to initialize the XXZ Heisenberg MPO
  for a given backend.

  Args:
    Jz, Jxy, D: Hamiltonian parameters.
    dtype: data type.
    backend: The backend.
  Returns:
    `tn.FiniteMPS`: A Haldane mps for the corresponding backend.
  """

    return FiniteHaldanespin1(Jz, Jxy, D, dtype=dtype, backend=backend)


def run_onesite_dmrg(N: int, bond_dim: int, dtype: Type[np.number], Jz: np.ndarray,
                     Jxy: np.ndarray, D: np.ndarray, num_sweeps: int,
                     backend: Text):
    """
  Run two-site dmrg for the XXZ Heisenberg model using a given backend.

  Args:
    N: Number of spins.
    bond_dim: The bond dimension.
    dtype: The data type of the MPS.
    Jz, Jxy, D: Hamiltonian parameters.
    num_sweeps: Number of DMRG sweeps to perform.
    backend: The backend.

  Returns:
    float/complex: The energy upon termination of DMRG.

  """
    mps = initialize_spin1_mps(N, 32, dtype, backend)
    mpo = initialize_Haldane_mpo(Jz, Jxy, D, dtype, backend)
    dmrg = tn.FiniteDMRG(mps, mpo)
    final_energy = dmrg.run_one_site(num_sweeps=num_sweeps, num_krylov_vecs=10, verbose=1)
    entangled_block = ncon([dmrg.mps.tensors[int(len(mps) / 2 - 1)], dmrg.mps.tensors[int(len(mps) / 2)]],
                           [[-1, -2, 1], [1, -3, -4]])

    u, s, v, _ = dmrg.mps.svd(entangled_block, 2, bond_dim, None)

    return final_energy, dmrg.mps, s


def run_twosite_dmrg(N: int, bond_dim: int, dtype: Type[np.number], Jz: np.ndarray,
                     Jxy: np.ndarray, D: np.ndarray, num_sweeps: int,
                     backend: Text):
    """
  Run two-site dmrg for the Haldane model using a given backend.

  Args:
    N: Number of spins.
    bond_dim: The bond dimension.
    dtype: The data type of the MPS.
    Jz, Jxy, D: Hamiltonian parameters.
    num_sweeps: Number of DMRG sweeps to perform.
    backend: The backend.

  Returns:
    float/complex: The energy upon termination of DMRG.

  """
    mps = initialize_spin1_mps(N, bond_dim, dtype, backend)
    mpo = initialize_Haldane_mpo(Jz, Jxy, D, dtype, backend)
    dmrg = tn.FiniteDMRG(mps, mpo)
    final_energy = dmrg.run_two_site(max_bond_dim=bond_dim, num_sweeps=num_sweeps, num_krylov_vecs=10, verbose=1)

    # mixed canonical

    for site in range(N - 1):
        bondBlock = ncon([dmrg.mps.tensors[site], dmrg.mps.tensors[site + 1]],
                         [[-1, -2, 1], [1, -3, -4]],
                         backend=dmrg.backend.name)
        u, s, vh, _ = dmrg.mps.svd(bondBlock, 2, bond_dim, None)
        s = dmrg.backend.diagflat(s)
        dmrg.mps.tensors[site] = u
        dmrg.mps.tensors[site + 1] = ncon([s, vh], [[-1, 1], [1, -2, -3]],
                                          backend=dmrg.backend.name)
    for site in range(N - 1, int(N / 2 - 1), -1):
        bondBlock = ncon([dmrg.mps.tensors[site - 1], dmrg.mps.tensors[site]],
                         [[-1, -2, 1], [1, -3, -4]],
                         backend=dmrg.backend.name)
        u, s, vh, _ = dmrg.mps.svd(bondBlock, 2, bond_dim, None)
        s = dmrg.backend.diagflat(s)
        dmrg.mps.tensors[site] = vh
        if site == N / 2:
            sigularValue = s

        dmrg.mps.tensors[site - 1] = ncon([u, s], [[-1, -2, 1], [1, -3]],
                                          backend=dmrg.backend.name)
    EntangleS = 0
    for i in range(bond_dim):
        if sigularValue[i][i] > 0:
            s2 = sigularValue[i][i] ** 2
            EntangleS = EntangleS - s2 * np.log(s2)

    return final_energy, dmrg.mps, EntangleS


def computeEntanglemnt(mps: tn.FiniteMPS):
    num_sites = len(mps.tensors)
    Lenv = mps.left_envs(range(num_sites))
    rouB = Lenv[num_sites / 2]

    Entropy = -np.trace(np.dot(rouB, np.log(rouB + 1e-3)))
    return Entropy


if __name__ == '__main__':

    num_sites, bond_dim, datatype = 16, 32, np.float64
    jz = np.ones(num_sites - 1)
    jxy = np.ones(num_sites - 1)
    D = np.ones(num_sites)
    n_sweeps = 12
    energies = {}
    be = 'numpy'

    print(f'\nrunning DMRG for {be} backend')

    energy, groundState, EntangleS = run_twosite_dmrg(
        num_sites,
        bond_dim,
        datatype,
        jz,
        jxy,
        D,
        num_sweeps=n_sweeps,
        backend=be)

    # 临界点lnL-S
    S_array = []
    num_sites = list(np.arange(8, 40, 4))+list(np.arange(40,180,20))
    lnL = np.log(num_sites)

    scatter_size = 100  # 散点大小

    fig, ax = plt.subplots()
    for num_site in num_sites:
        jz = np.ones(num_site - 1)
        jxy = np.ones(num_site - 1)
        D = 0.96*np.ones(num_site)
        energies, groundState, s = run_twosite_dmrg(
            num_site,
            bond_dim,
            datatype,
            jz,
            jxy,
            D,
            num_sweeps=n_sweeps,
            backend=be)
        Entropy = computeEntanglemnt(groundState)
        S_array.append(Entropy)
    ax.scatter(lnL, S_array, c='blue', s=scatter_size)
    plt.xlabel("lnL")
    plt.ylabel("S")
    plt.show()

    '''

    # 不同D下的bond_dimension-S图
    num_sites = 128
    jz = np.ones(num_sites - 1)
    jxy = np.ones(num_sites - 1)
    bz = np.ones(num_sites)
    alpha = np.linspace(start=0.25, stop=1.0, num=4)  # 透明度数组
    scatter_size = 100  # 散点大小数组
    bond_dimension = list(np.arange(2, 10, 2)) + list(np.arange(10, 120, 10))
    Dlist = [2, 1.2, 0.96]
    fig, ax = plt.subplots()
    for i in range(0, 3):
        D = Dlist[i] * bz
        Bond_array = []
        S_array = []
        for bond_dim in bond_dimension:
            energies, groundState,s = run_twosite_dmrg(
                num_sites,
                bond_dim,
                datatype,
                jz,
                jxy,
                D,
                num_sweeps=n_sweeps,
                backend=be)
            Entropy = computeEntanglemnt(groundState)

            Bond_array.append(bond_dim)
            print(f'Dmrg done for bond_dim={bond_dim},D={Dlist[i]}')

            S_array.append(Entropy)

        ax.scatter(Bond_array, S_array, c='blue', alpha=alpha[i], s=scatter_size)
        plt.legend([f'D={Dlist[i]}'])

    plt.xlabel("Bond Dimension")
    plt.ylabel("S")
    plt.show()
    '''
'''
    # 不同L下的D-S图
    alpha = np.linspace(start=0.25, stop=1.0, num=4)  # 透明度数组

    sitenumber = [16,32,64]
    Dlist = np.arange(0, 2.2, 0.1)
    fig, ax = plt.subplots()
    for i in range(len(sitenumber)):
        D_array = []
        S_array = []
        num_sites = sitenumber[i]
        jz = np.ones(num_sites - 1)
        jxy = np.ones(num_sites - 1)

        for d in Dlist:
            D = d * np.ones(num_sites)
            energies, groundState, EntangleS = run_twosite_dmrg(
                num_sites,
                bond_dim,
                datatype,
                jz,
                jxy,
                D,
                num_sweeps=n_sweeps,
                backend=be)
            Entropy = computeEntanglemnt(groundState)

            D_array.append(d)
            print(f'Dmrg done for d={d},L={num_sites}')

            S_array.append(EntangleS)

        ax.scatter(D_array, S_array, c='blue', alpha=alpha[i], s=150)

    plt.xlabel("D")
    plt.ylabel("S")
    plt.show()

    '''
'''
    #D-Energy
    Energies=[]

    Dlist = np.arange(0, 2.2, 0.1)
    for d in Dlist:
        energy, groundState = run_twosite_dmrg(
            num_sites,
            bond_dim,
            datatype,
            jz,
            jxy,
            d*D,
            num_sweeps=n_sweeps,
            backend=be)
        Energies.append(energy)
    plt.scatter(Dlist, Energies, c='blue',s=100)
    plt.show()
'''






