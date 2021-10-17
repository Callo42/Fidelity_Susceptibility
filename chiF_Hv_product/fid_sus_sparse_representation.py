"""
Computing fidelity susceptibility fid_sus(g)
of 1D TFIM. Fidelity susceptibility is an
important index for occurence of QPT.

In this file the chiF is computed using the 
sparse representation of the hamiltonian
rather than full matrix form

"""
import jax.numpy as jnp
from jax import jit, grad,lax
import numpy as np
from TFIM_init import H_u_initialize


def from_g_to_logproduct_sparse(g, N, k):
    """
    Computing logproduct using DominantSymeig in symeig.py,
    which is to perform AD with direct matrix form.

    Input:  'g': parameter in TFIM
            'g_no_diff': parameter in TFIM from which the
                computed psi_0 will not be applied
                to AD, have the same value with 'g'
            'N': number of sites
            'k': number of Lanczos vectors
    Output: 'log_product': the log product of the two
                ground state wave funciton:
                ln( < psi_0(g) | psi_0(g') > )
                Here the AD is only applied to |psi(g')>
    """
    from symeig import DominantSparseSymeig

    H_u, Hadjoint_to_gadjoint = H_u_initialize(N)
    dim = 2**N
    energy_0, psi_0 = DominantSparseSymeig(Hadjoint_to_gadjoint ,H_u,g,k,dim)
    psi_matmul = jnp.matmul(lax.stop_gradient(psi_0),psi_0)
    log_product = jnp.log(psi_matmul)
    return log_product


def fid_sus_sparse_repre(g,N,k):
    """
        Computing fidelity susceptibility using DominantSparseSymeig in symeig.py,
    which is to perform AD with sparse representation of matrix A.
    """
    dlogdg = grad(from_g_to_logproduct_sparse,argnums=0)
    d2logdg = grad(dlogdg,argnums=0)(g,N,k)
    fid_sus = - d2logdg
    return fid_sus
    



if __name__ == "__main__":
    N = 10
    k = 300
    g_count = 200
    gs = np.linspace(0.5, 1.5, num = g_count)
    fid_sus_from_sparse_repre = np.empty(g_count)
    
    for i in range(g_count):
        fid_sus_from_sparse_repre[i] = fid_sus_sparse_repre(gs[i], N, k)
        print(f"g: {gs[i]}")
        
        # datetime_now = datetime.datetime.now()
        # #save log to file
        # with open('fidelity_susceptibility/fid_sus.log', 'a') as f:
        #     f.write(f"#####################################################################################\n"
        #             f"#####################################################################################\n"
        #             f"Saving at {datetime_now}:\n"
        #             f"N={N}\n"
        #             f"g: {gs[i]} \n"
        #             f"fid_sus_matrixAD: {fid_sus_from_sparse_repre[i]}\n"
        #             f"#####################################################################################\n"
        #             f"#####################################################################################\n")
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(gs, fid_sus_from_sparse_repre, label = "AD: normal representation")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\chi_F$")
    ax.set_title("Fidelity susceptibility of 1D TFIM\n" 
            r"$H = - \sum_{i=0}^{N-1} (g\sigma_i^x + \sigma_i^z \sigma_{i+1}^z)$" "\n"
            f"$N={N}$ \n")
    plt.show()




    