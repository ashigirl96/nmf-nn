"""semi-NMF and Nonlinear semi-NMF solver by NumPy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from . import utility


def semi_nmf(a, u, v, rcond=1e-14, eps=1e-15, num_iters=1, first_nneg=True):
    """Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        eps:
        num_iters: Number of iterations
        first_nneg: Compute Non-negative matrix first

    Returns:
        u, v
    """
    
    def _compute_u(v):
        svd = utility._low_rank(v, rcond=rcond)
        u_t = np.transpose(svd.u)
        _v = svd.v
        s_inv = np.linalg.inv(svd.s)
        u = ((a @ _v) @ s_inv) @ u_t
        return u
    
    def _compute_v(u, v):
        u_t = np.transpose(u)
        uta = u_t @ a
        u_ta_p = (np.abs(uta) + uta) * 0.5
        u_ta_m = (np.abs(uta) - uta) * 0.5
        utu = u_t @ u
        u_tu_p = (np.abs(utu) + utu) * 0.5
        u_tu_m = (np.abs(utu) - utu) * 0.5
        
        uvm = u_tu_m @ v
        uvp = u_tu_p @ v
        divide = np.divide(u_ta_p + uvm,
                           u_ta_m + uvp + eps)
        # TODO: The divide induce Nan.
        # assert not divide[divide < 0.].sum(), '-1'
        divide[divide < 0.] = 0.
        sqrt = np.sqrt(divide)
        v = np.multiply(v, sqrt)
        return v
    
    for _ in range(num_iters):
        assert not np.isnan(v).any(), utility.have_nan('v', v)
        if first_nneg:
            v = _compute_v(u, v)
            u = _compute_u(v)
        else:
            u = _compute_u(v)
            v = _compute_v(u, v)
    return u, v


def softmax_nmf(a, u, v, rcond=1e-14, eps=1e-15, num_iters=1):
    """Softmax Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        eps:
        num_iters: Number of iterations
        first_nneg: Compute Non-negative matrix first

    Returns:
        u, v
    """
    
    def _compute_u(v):
        svd = utility._low_rank(v, rcond=rcond)
        u_t = np.transpose(svd.u)
        _v = svd.v
        s_inv = np.linalg.inv(svd.s)
        u = ((a @ _v) @ s_inv) @ u_t
        return u
    
    def _compute_v(u, v):
        u_t = np.transpose(u)
        uta = u_t @ a
        u_ta_p = (np.abs(uta) + uta) * 0.5
        u_ta_m = (np.abs(uta) - uta) * 0.5
        utu = u_t @ u
        u_tu_p = (np.abs(utu) + utu) * 0.5
        u_tu_m = (np.abs(utu) - utu) * 0.5
        
        uvm = u_tu_m @ v
        uvp = u_tu_p @ v
        divide = np.divide(u_ta_p + uvm,
                           u_ta_m + uvp + eps)
        # TODO: The divide induce Nan.
        # assert not divide[divide < 0.].sum(), '-1'
        divide[divide < 0.] = 0.
        sqrt = np.sqrt(divide)
        v = np.multiply(v, sqrt)
        return v
    
    for _ in range(num_iters):
        assert not np.isnan(v).any(), utility.have_nan('v', v)
        v = _compute_v(u, v)
        v = utility.softmax(v)
        u = _compute_u(v)
    return u, v


def _nonlin_solve(a, b, x, rcond=1e-14, num_iters=1, solve_ax=True):
    """Nonlinear Solver.
    Args:
        num_iters: Number of iterations each solving.
        solve_ax: Whether to solve min_x || b - f(ax) || or min_x || b - f(xa) ||
    """
    assert not np.isnan(a).any(), utility.have_nan('a', a)
    a_svd = utility._low_rank(a, rcond=rcond)
    u = a_svd.u
    s = a_svd.s
    v = a_svd.v
    
    _omega = 1.0
    
    def _solve_ax(x):
        """
         min_x || b - f(ax) ||
        """
        for _ in range(num_iters):
            r = b - utility.relu(a @ x)
            ur = u.T @ r
            solve = np.linalg.solve(s, ur)
            x = x + _omega * (v @ solve)
            x = utility.relu(x)
        return x
    
    def _solve_xa(x):
        """
         min_x || b - f(xa) ||
        """
        for _ in range(num_iters):
            r = b - utility.relu(x @ a)
            s_inv = np.linalg.inv(s)
            rvs = r @ v @ s_inv
            x = x + _omega * (rvs @ u.T)
        return x
    
    if solve_ax:
        return _solve_ax(x)
    else:
        return _solve_xa(x)


def nonlin_semi_nmf(a, u, v, rcond=1e-14, eps=1e-15, num_iters=1, num_calc_u=1, num_calc_v=1, first_nneg=True):
    """Nonlinear semi-NMF
    
    Args:
        a: Original non-negative matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        num_iters: Number of iterations
        first_nneg: Compute Non-negative matrix first

    Returns:
        Solved u, v
    """
    for _ in range(num_iters):
        if first_nneg:
            # In batch first, v has negative elements.
            v = _nonlin_solve(u, a, v, rcond=rcond, num_iters=num_calc_v, solve_ax=True)
            # In batch first, u has only non-negative elements.
            u = _nonlin_solve(v, a, u, rcond=rcond, num_iters=num_calc_u, solve_ax=False)
        else:
            # In batch first, u has only non-negative elements.
            u = _nonlin_solve(v, a, u, rcond=rcond, num_iters=num_calc_u, solve_ax=False)
            # In batch first, v has negative elements.
            v = _nonlin_solve(u, a, v, rcond=rcond, num_iters=num_calc_v, solve_ax=True)
    return u, v