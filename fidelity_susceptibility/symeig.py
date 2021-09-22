import jax.numpy as jnp
from jax import custom_vjp
from jax.config import config
config.update("jax_enable_x64", True)
from Lanczos import symeigLanczos
import numpy as np
from functools import partial
from jax import jit

@custom_vjp
def DominantSymeig(A,k):
    """
    Function primitive of dominant real symmetric
    eigensolver

    Input:  'A': the real symmetric matrix A,
                in a normal matrix form
            'k': number of requested Lanczos vectors
    Output: 'eigval': the smallest eigenvalue of A
            'eigvector':corresponding (non-degenerate)
                    eigenvector
    """
    eigval, eigvector = symeigLanczos(A, k, extreme="min")
    return (eigval, eigvector)

def DominantSymeig_fwd(A,k):
    eigval, eigvector = DominantSymeig(A,k)
    return (eigval, eigvector), (A, eigval, eigvector)

def DominantSymeig_bwd(res, grads):
    from CG import CGSubspace
    grad_eigval, grad_eigvector = grads
    A, eigval, eigvector = res
    Aprime = A - eigval * jnp.array(np.eye(A.shape[0]).astype(A.dtype))
    cg = CGSubspace
    b = grad_eigvector - jnp.matmul(eigvector, grad_eigvector) * eigvector
    lambda_0 = cg(Aprime, b, eigvector)
    grad_A = (grad_eigval * eigvector - lambda_0)[:, None] * eigvector
    grad_k = None
    return (grad_A, grad_k)
    
DominantSymeig.defvjp(DominantSymeig_fwd,DominantSymeig_bwd)