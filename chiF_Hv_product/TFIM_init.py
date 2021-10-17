import re
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jax

def H_u_initialize(N):
    """
    recieving the parameter g for TFIM model
    and a vector u
    and construct the Hamiltonian H,
    return the product of H and u:H*u

    Input: 'g': the parameter for TFIM model
            'u': an (arbitrary) vector
            'N': the number of sites of TFIM model
    Output: 'result_H_u': result_H_u = H*u
            'Hadjoint_to_gadjoint': the translation function
                    to calculate gadjoint form
                    Hadjoint
    """
    dim = 2**N

    #initialize the diagnal of the hamiltonian matrix
    diag_index = jnp.arange(dim)[:, jnp.newaxis]
    diag_bin_reps = (diag_index >> jnp.arange(N)[::-1]) & 1
    diag_spins = 1 - 2 * diag_bin_reps
    diag_spins_prime = jnp.hstack((diag_spins[:,1:],diag_spins[:,0:1]))
    diag_elements = -(diag_spins * diag_spins_prime).sum(axis=1)

    #initialize the basis of flips
    flip_masks = jnp.array([1 << i for i in range(N)], dtype="int64")
    flip_basis = jnp.arange(dim)[:,None]
    flips_basis = flip_basis ^ flip_masks

    #first derivative of H, in a vector product form
    def pHpg(v):
        result_pHpg = -v[flips_basis].sum(axis=1)
        return result_pHpg

    #defining the product function H
    def H_u(g,u):
        result_H_u = u * diag_elements - g * u[flips_basis].sum(axis=1)
        return result_H_u
    
    def Hadjoint_to_gadjoint(v1,v2):
        """
        the adjoint translation function to be 
        used in CG and symeig function

        Input: 'v1': one required vector to calculate A
                'v2': another vector to calculate A
                then A = v1 * v2^T(outer product)
        Output: 'g_adjoint': the adjoint of the parameters(g)
                    with respect to A
        """

        return jnp.matmul(pHpg(v2),v1)  #Here must ensure a scalar output
        
    
    return H_u, Hadjoint_to_gadjoint
    





if __name__ == "__main__":
    N = 10
    g = 0.5





    