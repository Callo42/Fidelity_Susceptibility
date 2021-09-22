#%%
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from functools import partial


@jit
def f_all_g(g):
    x = jnp.array([1,g**2])
    product = jnp.matmul(x,x)
    return product

@partial(jit, static_argnums=(1,))
def f_one_g(g,g_no_diff):
    x = jnp.array([1,g**2])
    x_no_diff = jnp.array([1,g_no_diff**2])
    product_one = jnp.matmul(x_no_diff,x)
    return product_one

def d2F_all_dg(g):
    dfdg = grad(f_all_g)
    df2dg = grad(dfdg)(g)
    return df2dg

def d2F_one_dg(g):
    g_diff = g
    g_no_diff = g
    dfdg = grad(f_one_g)
    d2fdg = grad(dfdg)(g_diff, g_no_diff)
    return d2fdg


g = 2.0


print(f"f_all_g({g}) = {f_all_g(g)}\n"
    f"grad_f_all_g | g={g} = {grad(f_all_g)(g)}\n"
    f"d2F_all_dg | g = {g} = {d2F_all_dg(g)}")
    
print(f"f_one_g({g}) = {f_one_g(g,g)}\n"
    f"grad_f_one_g | g={g} = {grad(f_one_g)(g,g)}\n"
    f"d2F_one_dg | g={g} = {d2F_one_dg(g)}\n")


# g_count = 5
# gs = jnp.linspace(0.5,1.5, num = g_count)
# d2fdg_all = np.empty(g_count)
# d2fdg_one = np.empty(g_count)
# for i in range(g_count):
#     d2fdg_all[i] = d2F_all_dg(gs[i])
#     d2fdg_one[i] = d2F_one_dg(float(gs[i]))
#     print(f"g: {gs[i]} \n"
#         f"d2fdg_all: {d2fdg_all[i]}\n"
#         f"d2fdg_one: {d2fdg_one[i]}\n") 


#%%
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from functools import partial


@jit
def f_all_g(g):
    x = jnp.array([1,g**2])
    product = jnp.matmul(x,x)
    return product

def f_one_g(g,g_no_diff):
    x = jnp.array([1,g**2])
    x_no_diff = jnp.array([1,g_no_diff**2])
    product_one = jnp.matmul(x_no_diff,x)
    return product_one

def d2F_all_dg(g):
    dfdg = grad(f_all_g)
    df2dg = grad(dfdg)(g)
    return df2dg

def d2F_one_dg(g):
    g_diff = g
    g_no_diff = g
    dfdg = grad(f_one_g)
    d2fdg = grad(dfdg)(g_diff, g_no_diff)
    return d2fdg


g = 2.0


print(f"f_all_g({g}) = {f_all_g(g)}\n"
    f"grad_f_all_g | g={g} = {grad(f_all_g)(g)}\n"
    f"d2F_all_dg | g = {g} = {d2F_all_dg(g)}")
    
print(f"f_one_g({g}) = {f_one_g(g,g)}\n"
    f"grad_f_one_g | g={g} = {grad(f_one_g)(g,g)}\n"
    f"d2F_one_dg | g={g} = {d2F_one_dg(g)}\n")


# g_count = 5
# gs = jnp.linspace(0.5,1.5, num = g_count)
# d2fdg_all = np.empty(g_count)
# d2fdg_one = np.empty(g_count)
# for i in range(g_count):
#     d2fdg_all[i] = d2F_all_dg(gs[i])
#     d2fdg_one[i] = d2F_one_dg(float(gs[i]))
#     print(f"g: {gs[i]} \n"
#         f"d2fdg_all: {d2fdg_all[i]}\n"
#         f"d2fdg_one: {d2fdg_one[i]}\n") 
# %%
