from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from functools import partial
from jax import jit

class TFIM(object):
    """
        1D Transverse Field Ising Model(TFIM)
        Hamiltonian:
        H = - \sum_{i=0}^{N-1} (g \sigma_i^x + \sigma-i^z \sigma_{i+1}^z)
    """
    # @partial(jit, static_argnums=(1,))
    def __init__(self, N,g):
        """
        Initialization of the model
        """
        self.N = N
        self.dim = 2**N
        self.g = g
        self._diags()
        self._flips_basis()
        # print(f"1D lattice size N={self.N} \n"
        #     f"Model initialization completed.")

    def _diags(self):
        """
        Deal with the diagonal elements of the Harmiltonian
        """
        indices = jnp.arange(self.dim)[:,jnp.newaxis]
        bin_reps = (indices >> jnp.arange(self.N)[::-1]) & 1
        spins = 1 - 2 * bin_reps
        spins_prime = jnp.hstack( (spins[:,1:] , spins[:,0:1]) )
        self.diag_elements = -(spins * spins_prime).sum(axis=1)

    def _flips_basis(self):
        """
        Deal with \sigma_i^x
        """
        masks = jnp.array([1 << i for i in range(self.N)], dtype="int64")
        basis = jnp.arange(self.dim)[:,None]
        self.flips_basis = basis ^ masks

    def setpHpg(self):
        """
        Return the direct matrix of
        \partial H / \partial g
        To be used in the full spectrum perturbation formula method
        of calculating chi_F(fidelity susceptibility)
        """
        self.pHpgmatrix = jnp.zeros([self.dim,self.dim], dtype="float64")
        self.pHpgmatrix = self.pHpgmatrix.at[self.flips_basis.T, jnp.arange(self.dim)].set(-1.0)

    def pHpg(self,v):
        """
        Using equation (26):
        pHpg = \partial H / \partial g
            =  - \sum_{i=0}^{N-1} \sigma_i^x
        """
        resultv = -v[self.flips_basis].sum(axis=1)
        return resultv

    def setHmatrix(self):
        """
        Initialize the Hamiltonian 
        To be stored in "self.Hmatrix"
        """
        diagmatrix = jnp.diag(self.diag_elements)
        offdiagmatrix = jnp.zeros([self.dim,self.dim],dtype="float64")
        offdiagmatrix = offdiagmatrix.at[self.flips_basis.T, jnp.arange(self.dim)].set(-self.g)

        #to avoid devided by zero, add random noise into the Hamiltonian
        randommatrix = 1e-12 * jnp.array(np.random.randn(self.dim, self.dim))
        randommatrix = 0.5 * (randommatrix + randommatrix.T)

        self.Hmatrix = diagmatrix + offdiagmatrix + randommatrix

    def H(self, v):
        """
        The sparse Hamiltonian 
        Written in a function representation
        Aka, a "sparse" linear transformation that
        recieves a vector v and returns another vector
        """
        resultv = v * self.diag_elements - self.g * v[self.flips_basis].sum(axis=1)
        return resultv

    def Hadjoint_to_gadjoint(self, v_1, v_2):
        """
        A function that receive the adjoint of the matrix H
        as input, and return the adjoint of g as output.
        """
        return jnp.matmul(self.pHpg(v_2),v_1)[None]


class TFIM_no_diff(object):
    """
        1D Transverse Field Ising Model(TFIM)
        Hamiltonian:
        H = - \sum_{i=0}^{N-1} (g \sigma_i^x + \sigma-i^z \sigma_{i+1}^z)
    """
    
    def __init__(self, N,g):
        """
        Initialization of the model
        """
        self.N = N
        self.dim = 2**N
        self.g = g
        self._diags()
        self._flips_basis()
        # print(f"1D lattice size N={self.N} \n"
        #     f"Model initialization completed.")

    def _diags(self):
        """
        Deal with the diagonal elements of the Harmiltonian
        """
        indices = np.arange(self.dim)[:,np.newaxis]
        bin_reps = (indices >> np.arange(self.N)[::-1]) & 1
        spins = 1 - 2 * bin_reps
        spins_prime = np.hstack( (spins[:,1:] , spins[:,0:1]) )
        self.diag_elements = -(spins * spins_prime).sum(axis=1)

    def _flips_basis(self):
        """
        Deal with \sigma_i^x
        """
        masks = np.array([1 << i for i in range(self.N)], dtype="int64")
        basis = np.arange(self.dim)[:,None]
        self.flips_basis = basis ^ masks

    def setpHpg(self):
        """
        Return the direct matrix of
        \partial H / \partial g
        To be used in the full spectrum perturbation formula method
        of calculating chi_F(fidelity susceptibility)
        """
        self.pHpgmatrix = np.zeros([self.dim,self.dim], dtype="float64")
        self.pHpgmatrix[self.flips_basis.T, np.arange(self.dim)] = -1.0

    def pHpg(self,v):
        """
        Using equation (26):
        pHpg = \partial H / \partial g
            =  - \sum_{i=0}^{N-1} \sigma_i^x
        """
        resultv = -v[self.flips_basis].sum(axis=1)
        return resultv

    def setHmatrix(self):
        """
        Initialize the Hamiltonian 
        To be stored in "self.Hmatrix"
        """
        diagmatrix = np.diag(self.diag_elements)
        offdiagmatrix = np.zeros([self.dim,self.dim],dtype="float64")
        offdiagmatrix[self.flips_basis.T, np.arange(self.dim)] = -self.g

        #to avoid devided by zero, add random noise into the Hamiltonian
        randommatrix = 1e-12 * np.array(np.random.randn(self.dim, self.dim))
        randommatrix = 0.5 * (randommatrix + randommatrix.T)

        self.Hmatrix = diagmatrix + offdiagmatrix + randommatrix

    def H(self, v):
        """
        The sparse Hamiltonian 
        Written in a function representation
        Aka, a "sparse" linear transformation that
        recieves a vector v and returns another vector
        """
        resultv = v * self.diag_elements - self.g * v[self.flips_basis].sum(axis=1)
        return resultv

    def Hadjoint_to_gadjoint(self, v_1, v_2):
        """
        A function that receive the adjoint of the matrix H
        as input, and return the adjoint of g as output.
        """
        return np.matmul(self.pHpg(v_2),v_1)[None]

    
if __name__ == "__main__":
    N = 10
    g = 0.5
    model = TFIM(N,g)

    print("testing method 'setpHpg'\n")
    model.setpHpg()
    print("Done\n")

    print("testing method 'setHmatrix'\n")
    model.setHmatrix()
    print("Done\n")

    print("testing method 'pHpg'\n")
    pHpg_matrix = model.pHpgmatrix
    v = jnp.array(np.random.randn(pHpg_matrix.shape[0]))
    resultv = model.pHpg(v)
    print(f"Done with return {resultv}\n")

    print("testing method 'H'\n")
    pHpg_matrix = model.pHpgmatrix
    v = jnp.array(np.random.randn(pHpg_matrix.shape[0]))
    resultv = model.H(v)
    print(f"Done with return {resultv}\n")

    print("testing method 'Hadjoint_to_gadjoint'\n")
    pHpg_matrix = model.pHpgmatrix
    v_1 = jnp.array(np.random.randn(pHpg_matrix.shape[0]))
    v_2 = jnp.array(np.random.randn(pHpg_matrix.shape[0]))
    resultv = model.Hadjoint_to_gadjoint(v_1,v_2)
    print(f"Done with return {resultv}\n")

    print("Tests passed!")




    