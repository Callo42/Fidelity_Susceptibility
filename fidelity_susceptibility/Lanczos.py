import jax.numpy as jnp
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

def Lanczos(A, k, *, sparse=False, dim=None):
    """
    Lanczos method to help solve the eigenvalue
    and eigenvector of a real symmetrix matrix.

    Input:  'A': The n times n real symmetrix matrix.
            'k': The number of Lanczos vectors requested.
            'sparse': If sparse = True, means the matrix
                    input A is in a linear function representation
                    form.
            'dim':  If sparse = True, then the integer param 'dim'
                    should be given, indicating the dimension
                    of the square matrix A

    Output: A tuple (Q_k, T): Q_k = (q_1 q_2 ... q_k) is a n*k matrix, 
            whose columns contain k orthomormal Lanczos vectors q1, q2, ..., qk.
            T is the tridiagonal matrix of size k, satisfying Qk^T * Q_k = I_k,
            and T has the same eigenvalue with matrix A, and the eigenvector
            of A could be drawn from the eigenvector of T.( let eigenvector of T
            be x, then y=Q_k*x is the eigenvector of A)
    """
    if sparse == True:
        n = dim
        dtype = jnp.float64
        Amap = A
    else:
        n = A.shape[0]
        dtype = A.dtype
        Amap = lambda v: jnp.matmul(A, v)
    
    Q_k = jnp.zeros((n,k), dtype=dtype)
    alphas = jnp.zeros(k, dtype=dtype)
    betas = jnp.zeros(k - 1, dtype=dtype)
    q = jnp.array(np.random.randn(n).astype(dtype))
    q /= jnp.linalg.norm(q)
    u = Amap(q)
    alpha = jnp.matmul(q,u)
    Q_k = Q_k.at[:, 0].set(q)
    alphas = alphas.at[0].set(alpha)
    beta = 0
    qprime = jnp.array(np.random.randn(n).astype(dtype))
    for i in range(1,k):
        r = u - alpha * q - beta * qprime

        #Reorthogonalization
        r -= jnp.matmul(Q_k[:, :i], jnp.matmul(Q_k[:, :i].T, r))

        qprime = q
        beta = jnp.linalg.norm(r)
        q = r / beta
        u = Amap(q)
        alpha = jnp.matmul(q,u)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i-1].set(beta)
        Q_k = Q_k.at[:, i].set(q)
    #param k in jnp.diag indicates which diagonal to consider
    T = jnp.diag(alphas) + jnp.diag(betas, k=1) + jnp.diag(betas, k=-1)
    return Q_k, T

def symeigLanczos(A, k, extreme="both", *,
                    sparse=False, dim=None):
    """
        Computes the extreme eigenvalues and eigenvectors
    upon request of a symmetric matrix A with Lanczos
    algorithm implemented.

    Input:  'A': The n times n real symmetrix matrix.
        'k': The number of Lanczos vectors requested.
        'sparse': If sparse = True, means the matrix
                input A is in a linear function representation
                form.
        'dim':  If sparse = True, then the integer param 'dim'
                should be given, indicating the dimension
                of the square matrix A
        'extreme':  labels the desired eigenvalues and 
                corresponding eigenvectors to be returned.
                Specificlly,
                "both" -> both min and max.     --Output--> (eigval_min, eigvector_min, eigval_max, eigvector_max)
                "min" -> min.                   --Output--> (eigval_min, eigvector_min)
                "max" -> max.                   --Output--> (eigval_max, eigvector_max)
    Output: See "Input" above.
    """
    Q_k, T = Lanczos(A, k, sparse=sparse, dim=dim)
    eigvalsQ, eigvectorsQ = jnp.linalg.eigh(T,UPLO='U')
    eigvectorsQ = jnp.matmul(Q_k,eigvectorsQ)
    if extreme == "both":
        return eigvalsQ[0], eigvectorsQ[:, 0], eigvalsQ[-1], eigvectorsQ[:, -1]
    elif extreme == "min":
        return eigvalsQ[0], eigvectorsQ[:, 0]
    elif extreme == "max":
        return eigvalsQ[-1], eigvectorsQ[:, -1]



if __name__ == "__main__":
    from TFIM_init import TFIM

    N = 2
    k = 3   
    g = 1
    model = TFIM(N,g)


    model.setHmatrix()
    H = model.Hmatrix
    
    print(f"H for N={N} is \n"
            f"{H}\n")
    print(f"Then testing Lanczos algorithm:\n")

    eigval_min, eigvector_min = symeigLanczos(H, k, extreme="min")
    eigval_max, eigvector_max = symeigLanczos(H, k, extreme="max")
    eigval_both_min, eigvector_both_min, eigval_both_max, eigvector_both_max = symeigLanczos(H, k, extreme="both")

    print(f"param extreme = 'min', giving:\n"
        f"eigval_min = {eigval_min}\n"
        f"eigvector_min = {eigvector_min}\n")
    print(f"param extreme = 'max', giving:\n"
        f"eigval_max = {eigval_max}\n"
        f"eigvector_max = {eigvector_max}\n")
    print(f"param extreme = 'both', giving:\n"
        f"eigval_both_min = {eigval_both_min}\n"
        f"eigvector_both_min = {eigvector_both_min}\n"
        f"eigval_both_max = {eigval_both_max}\n"
        f"eigvector_both_max = {eigvector_both_max}\n")

    print("Test completed.")

    