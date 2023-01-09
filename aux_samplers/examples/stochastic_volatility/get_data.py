import numpy as np


def get_data(key, nu, phi, tau, rho, dim, T):
    pass


def get_dynamics(nu, phi, tau, rho, dim):

    F = phi * np.eye(dim)
    Q = stationary_covariance(phi, tau, rho, dim)
    mu = nu * np.ones((dim,))
    b = mu + F @ mu

    m0 = mu
    P0 = Q



def stationary_covariance(phi, tau, rho, dim):
    U = tau * rho * np.ones((dim, dim))
    U[np.diag_indices(dim)] = tau
    vec_U = np.reshape(U, (dim ** 2, 1))
    vec_U_star = vec_U / (1 - phi ** 2)
    U_star = np.reshape(vec_U_star, (dim, dim))
    return U_star
