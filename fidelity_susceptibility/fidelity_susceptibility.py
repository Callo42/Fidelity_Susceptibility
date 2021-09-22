"""
Computing fidelity susceptibility fid_sus(g)
of 1D TFIM. Fidelity susceptibility is an
important index for occurence of QPT.

"""
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
from TFIM_init import TFIM, TFIM_no_diff
from functools import partial
import datetime


# @partial(jit, static_argnums = (1,2,3))

def from_g_to_logproduct(g,g_no_diff, N, k):
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
    from symeig import DominantSymeig

    model = TFIM(N,g)
    model.setHmatrix()
    hamiltonian = model.Hmatrix

    model_no_diff = TFIM_no_diff(N,g_no_diff)
    model_no_diff.setHmatrix()
    h_no_diff = model_no_diff.Hmatrix

    energy_0, psi_0 = DominantSymeig(hamiltonian, k)
    energy_0_no_diff, psi_0_no_diff = DominantSymeig(h_no_diff, k)
    psi_matmul = jnp.matmul(psi_0_no_diff,psi_0)
    log_product = jnp.log(psi_matmul)
    return log_product


def fid_sus_matrix_AD(g,N,k):
    """
        Computing fidelity susceptibility using DominantSymeig in symeig.py,
    which is to perform AD with direct matrix form.
    """
    g_diff = g
    g_no_diff = g
    dlogdg = grad(from_g_to_logproduct)
    d2logdg = grad(dlogdg)(g_diff,g_no_diff,N,k)
    fid_sus = - d2logdg
    return fid_sus



if __name__ == "__main__":
    N = 10
    k = 300
    g_count = 5
    gs = np.linspace(0.5, 1.5, num = g_count)
    fid_sus_from_matrix_AD = np.empty(g_count)
    
    for i in range(g_count):
        fid_sus_from_matrix_AD[i] = fid_sus_matrix_AD(gs[i], N, k)
        print(f"g: {gs[i]}")
        
        datetime_now = datetime.datetime.now()
        #save log to file
        with open('fidelity_susceptibility/fid_sus.log', 'a') as f:
            f.write(f"#####################################################################################\n"
                    f"#####################################################################################\n"
                    f"Saving at {datetime_now}:\n"
                    f"N={N}\n"
                    f"g: {gs[i]} \n"
                    f"fid_sus_matrixAD: {fid_sus_from_matrix_AD[i]}\n"
                    f"#####################################################################################\n"
                    f"#####################################################################################\n")
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(gs, fid_sus_from_matrix_AD, label = "AD: normal representation")
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\chi_F$")
    ax.set_title("Fidelity susceptibility of 1D TFIM\n" 
            r"$H = - \sum_{i=0}^{N-1} (g\sigma_i^x + \sigma_i^z \sigma_{i+1}^z)$" "\n"
            f"$N={N}$ \n")
    plt.show()




    