import jax.numpy as jnp
from jax import custom_vjp
import numpy as np
from functools import partial



def CG_Algorithm(g,A, b, initial_x, sparse=False):
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
    r = b - Amap(g,x)

    if(np.linalg.norm(r) < epsilon):
        return x

    d = r
    alpha = np.matmul(r, r) / np.matmul(Amap(g,d),d)

    for i in range(n):
        x = x + alpha * d
        r_next = r - alpha * Amap(g,d)
        if(np.linalg.norm(r_next) < epsilon):
            break
        beta = np.matmul(r_next,r_next) / np.matmul(r,r)
        r = r_next
        d = r + beta * d
        alpha = np.matmul(r,r) / np.matmul(Amap(g,d), d)
    
    return x


@partial(custom_vjp, nondiff_argnums=(0,1,))
def CGSubspaceSparse(Aadjoint_to_gadjoint, A, g, E_0, b, alpha):
    """
    Function primitive for low-rank CG linear
    system solver, here A is 'sparse',
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
            'E_0': smallest eigvalue
            'b': The vector satisfying (A - E_0I)x = b
            'alpha': The unique eigenvector of A w.r.t. E_0
    Output: the unique solution x of the low-rank 
            linear system (A - E_0I)x = b in addition to
            the condition alpha^T x = 0.
    """
    Aprime = lambda g,v: A(g,v) - E_0 * v
    initial_x = jnp.array(np.random.randn(b.shape[0]).astype(b.dtype))
    initial_x = initial_x - jnp.matmul(alpha,initial_x) * alpha
    x = CG_Algorithm(g,Aprime,b,initial_x,sparse=True)
    return x

def CGSubspaceSparse_fwd(Aadjoint_to_gadjoint, A, g, E_0, b, alpha):
    x = CGSubspaceSparse(Aadjoint_to_gadjoint,A,g,E_0,b,alpha)
    res = (g, E_0, alpha,x)
    return x, res

def CGSubspaceSparse_bwd(Aadjoint_to_gadjoint,A,res, grad_x):
    g, E_0, alpha,x = res
    b = grad_x - jnp.matmul(alpha, grad_x) * alpha
    grad_b = CGSubspaceSparse(Aadjoint_to_gadjoint, A, g, E_0, b, alpha)
    v_1, v_2 = - grad_b, x
    grad_alpha = -x * jnp.matmul(alpha, grad_x)
    grad_E_0 = -jnp.matmul(v_1, v_2)
    grad_g = Aadjoint_to_gadjoint(v_1, v_2)
    return (grad_g,grad_E_0,grad_b,grad_alpha)

CGSubspaceSparse.defvjp(CGSubspaceSparse_fwd,CGSubspaceSparse_bwd)

