import jax.numpy as jnp
from jax import custom_vjp
import numpy as np
from functools import partial
from jax import jit


def CG_Algorithm(A, b, initial_x, sparse=False):
    """
    Solve for x in linear equation Ax=b
    with an initial choose of x, initial_x.
    Note here A is a matrix and x and b are vectors.
    Using Conjugate Gradient(CG) method.

    Input:
        'A': The real symmetric and positive definite
            matrix A.
        'b': The vector corresponding to the equation Ax=b
        'initial_x': The initial vector of the CG algorithm.
        `sparse`: If sparse = True, means the matrix
                input A is in a linear function representation
                form. In any cases, the dimension of A is 
                inferred from the size of the vector b.
    Output:
        'x': the solution for Ax=b
    """
    if sparse == True:
        Amap = A
    else:
        Amap = lambda v: np.matmul(A,v)
    n = b.shape[0]
    epsilon = 1e-7
    x = initial_x
    r = b - Amap(x)

    if(np.linalg.norm(r) < epsilon):
        return x

    d = r
    alpha = np.matmul(r, r) / np.matmul(Amap(d),d)

    for i in range(n):
        x = x + alpha * d
        r_next = r - alpha * Amap(d)
        if(np.linalg.norm(r_next) < epsilon):
            break
        beta = np.matmul(r_next,r_next) / np.matmul(r,r)
        r = r_next
        d = r + beta * d
        alpha = np.matmul(r,r) / np.matmul(Amap(d), d)
    
    return x


@custom_vjp
def CGSubspace(A, b, alpha):
    """
    Function primitive for low-rank CG linear
    system solver, here A is not 'sparse',
    hence in a normal matrix form.
    Input: 'A': a N-dimensional real symmetric
        matrix of rank N - 1
            'b': The vector satisfying Ax=b
            'alpha': The unique eigenvector of A of eigenvalue
        zero.(The other eigenvalues of A are all greater than zero.)
    Output: the unique solution x of the low-rank 
            linear system Ax = b in addition to
            the condition alpha^T x = 0.
    """
    initial_x = jnp.array(np.random.randn(b.shape[0]).astype(b.dtype))
    initial_x = initial_x - jnp.matmul(alpha, initial_x) * alpha
    x = CG_Algorithm(A, b, initial_x)
    return x

def CGSubspace_fwd(A, b, alpha):
    x = CGSubspace(A,b,alpha)
    return x, (A,alpha,x)

def CGSubspace_bwd(res, grad_x):
    A, alpha, x = res
    CG = CGSubspace
    b = grad_x - jnp.matmul(alpha, grad_x) * alpha
    grad_b = CG(A, b, alpha)
    grad_A = - grad_b[:, None] * x
    grad_alpha = - x * jnp.matmul(alpha, grad_x)
    return (grad_A, grad_b, grad_alpha)

CGSubspace.defvjp(CGSubspace_fwd,CGSubspace_bwd)

