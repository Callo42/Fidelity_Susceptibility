import jax.numpy as jnp
from jax import custom_vjp
from jax.config import config
config.update("jax_enable_x64", True)
from Lanczos import symeigLanczos
import numpy as np
from functools import partial

@partial(custom_vjp, nondiff_argnums=(0,1,))
def DominantSparseSymeig(Aadjoint_to_gadjoint, A, g, k, dim):
    """
    Function primitive for dominant real symmetric
    eigensolver,note here A is 'sparse',
    hence in a function representation,
    which is a linear transformation that
    takes a vector as input and returns another
    vector (A*v) as output.
    Input:  'Aadjoint_to_gadjoint': The function that
        recieves the adjoint of A and returns the
        adjoint of the depending parameter g. Note
        here 'Aadjoint_to_gadjoint' is a python callable.
            'A': a linear transformation form of matrix A
            'g': The depending parameter that determines matrix A.
            'k': number of requested Lanczos vectors
            'dim': The dimension of the square matrix A.
    Output: 'eigval': the smallest eigenvalue of A
            'eigvector':corresponding (non-degenerate)
                    eigenvector
    """
    eigval, eigvector = symeigLanczos(g,A,k,extreme="min", sparse=True, dim=dim)
    return (eigval, eigvector)

def DominantSparseSymeig_fwd(Aadjoint_to_gadjoint, A, g, k, dim):
    eigval, eigvector = DominantSparseSymeig(Aadjoint_to_gadjoint, A, g, k, dim)
    res = (g, eigval, eigvector)
    return (eigval, eigvector), res

def DominantSparseSymeig_bwd(Aadjoint_to_gadjoint,A,res, grads):
    from CG import CGSubspaceSparse
    grad_eigval, grad_eigvector = grads
    g, eigval, eigvector = res
    b = grad_eigvector - jnp.matmul(eigvector, grad_eigvector) * eigvector
    lambda_0 = CGSubspaceSparse(Aadjoint_to_gadjoint, A, g, eigval, b, eigvector)
    grad_A = grad_eigval * eigvector - lambda_0, eigvector
    v_1, v_2 = grad_A
    grad_g = Aadjoint_to_gadjoint(v_1,v_2)
    grad_k = grad_dim = None
    return (grad_g, grad_k,grad_dim)

DominantSparseSymeig.defvjp(DominantSparseSymeig_fwd,DominantSparseSymeig_bwd)
