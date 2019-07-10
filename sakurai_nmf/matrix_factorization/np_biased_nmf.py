"""Biased semi-NMF and Nonlinear semi-NMF solver by NumPy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from . import utility


def semi_nmf(a, u, v, alpha=1e-2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1, first_nneg=True):
    """Biased Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        eps:

    Returns:
        u, v
    """
    n = v.shape[1]
    bias = np.ones((n, 1))
    bias_v = np.vstack((v, bias.T))
    
    def _compute_u(u, bias_v):
        svd = utility._low_rank(bias_v, rcond=rcond)
        u_t = np.transpose(svd.u)
        r = a - u @ bias_v
        rv = r @ svd.v
        s_inv = np.linalg.inv(svd.s)
        u = u + (rv @ s_inv) @ u_t
        ss = np.diag(svd.s)
        ss_square = np.square(ss)
        ss = np.divide(ss_square,
                       (alpha + ss_square))
        u = u @ (svd.u @ np.diag(ss) @ u_t)
        return u
    
    def _compute_v(u, v, bias_v):
        u_org = u[:, :-1]
        u_t = np.transpose(u_org)
        ua = u_t @ a
        uap = (np.abs(ua) + ua) * 0.5
        uam = (np.abs(ua) - ua) * 0.5
        uu = u_t @ u
        uup = (np.abs(uu) + uu) * 0.5
        uum = (np.abs(uu) - uu) * 0.5
        
        divide = np.divide(uap + uum @ bias_v + beta * v,
                           uam + uup @ bias_v + beta * v + eps)
        # TODO: The divide induce Nan.
        divide[divide < 0.] = 0.
        sqrt = np.sqrt(divide)
        v = np.multiply(v, sqrt)
        bias_v = np.vstack((v, bias.T))
        return v, bias_v
    
    for _ in range(num_iters):
        if first_nneg:
            v, bias_v = _compute_v(u, v, bias_v)
            u = _compute_u(u, bias_v)
        else:
            u = _compute_u(u, bias_v)
            v, bias_v = _compute_v(u, v, bias_v)
    return u, v


def softmax_nmf(a, u, v, alpha=1e-2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1):
    """Softmax Biased Semi-NMF
    Args:
        a: Original matrix factorized
        u: Left matrix
        v: Non-negative matrix
        rcond: Reciprocal condition number
        eps:

    Returns:
        u, v
    """
    n = v.shape[1]
    bias = np.ones((n, 1))
    bias_v = np.vstack((v, bias.T))
    
    def _compute_u(u, bias_v):
        svd = utility._low_rank(bias_v, rcond=rcond)
        u_t = np.transpose(svd.u)
        r = a - u @ bias_v
        rv = r @ svd.v
        s_inv = np.linalg.inv(svd.s)
        u = u + (rv @ s_inv) @ u_t
        ss = np.diag(svd.s)
        ss_square = np.square(ss)
        ss = np.divide(ss_square,
                       (alpha + ss_square))
        u = u @ (svd.u @ np.diag(ss) @ u_t)
        return u
    
    def _compute_v(u, v, bias_v):
        u_org = u[:, :-1]
        u_t = np.transpose(u_org)
        ua = u_t @ a
        uap = (np.abs(ua) + ua) * 0.5
        uam = (np.abs(ua) - ua) * 0.5
        uu = u_t @ u
        uup = (np.abs(uu) + uu) * 0.5
        uum = (np.abs(uu) - uu) * 0.5
        
        divide = np.divide(uap + uum @ bias_v + beta * v,
                           uam + uup @ bias_v + beta * v + eps)
        # TODO: The divide induce Nan.
        divide[divide < 0.] = 0.
        sqrt = np.sqrt(divide)
        v = np.multiply(v, sqrt)
        bias_v = np.vstack((v, bias.T))
        return v, bias_v
    
    for _ in range(num_iters):
        v, bias_v = _compute_v(u, v, bias_v)
        v = utility.softmax(v)
        bias_v = utility.softmax(bias_v)
        u = _compute_u(u, bias_v)
    return u, v


def _nonlin_solve(a, b, x, _lambda=1e-2, rcond=1e-14, eps=1e-15, num_iters=1, solve_ax=True):
    """Nonlinear Solver.
    Args:
        num_iters: Number of iterations each solving.
        solve_ax: Whether to solve min_x || b - f(ax) || or min_x || b - f(xa) ||
    """
    _omega = 1.0
    
    def _solve_ax(x):
        """
         min_x || b - f(ax) ||
        """
        a_svd = utility._low_rank(a[:, :-1], rcond=1e-14)
        u = a_svd.u
        s = a_svd.s
        v = a_svd.v
        
        n = a.shape[1]
        bias = np.ones((x.shape[1], 1))
        bias_x = np.vstack((x, bias.T))
        _aa = a[:, :-1].T @ a[:, :-1]
        u_svd = utility._low_rank(_aa, rcond=rcond)
        
        for _ in range(num_iters):
            r = b - utility.relu(a @ bias_x)
            ur = u.T @ r
            sur_solve = np.linalg.solve(s, ur)
            x = x + _omega * (v @ sur_solve)
            _eye = np.eye(n - 1)
            su_solve = np.linalg.solve(u_svd.s, u_svd.u.T)
            vsu = u_svd.v @ su_solve
            _x = _eye + _lambda * vsu
            # TODO: in relu case, cannot solve next code.
            # print("Error: x's rank = {0}".format(np.linalg.matrix_rank(_x)))
            if np.linalg.matrix_rank(_x) == 1:
                print("_x is ", _x)
            x = np.linalg.solve(_x, x)
            x = utility.relu(x)
            bias_x = np.vstack((x, bias.T))
        return x
    
    def _solve_xa(x):
        """
         min_x || b - f(xa) ||
        """
        bias = np.ones((a.shape[1], 1))
        _a = np.vstack((a, bias.T))
        a_svd = utility._low_rank(_a, rcond=rcond)
        u = a_svd.u
        s = a_svd.s
        v = a_svd.v
        
        bias = np.ones((a.shape[1], 1))
        bias_x = np.vstack((a, bias.T))
        for _ in range(num_iters):
            r = b - utility.relu(x @ bias_x)
            rv = r @ v
            s_inv = np.linalg.inv(s)
            rvs = rv @ s_inv
            x = x + _omega * (rvs @ u.T)
            ss = np.diag(s)
            ss = np.divide(
                np.square(ss),
                np.square(ss) + _lambda)
            x = x @ (u @ np.diag(ss) @ u.T)
        return x
    
    if solve_ax:
        return _solve_ax(x)
    else:
        return _solve_xa(x)


def nonlin_semi_nmf(a, u, v, alpha=1e2, beta=1e-2, rcond=1e-14, eps=1e-15, num_iters=1, num_calc_u=1, num_calc_v=1,
                    first_nneg=True, batch_first=True):
    """Biased Nonlinear Semi-NMF
    Args:
        a: Original non-negative matrix factorized
        u: Biased-matrix
        v: Non-negative matrix
        alpha: Coefficient for solve u.
        beta: Coefficient for solve v.
        rcond: Reciprocal condition number
        eps:
        num_iters: Number of iterations
        batch_first: like TensorFlow format.

    Returns:

    """
    if batch_first:
        num_calc_u, num_calc_v = num_calc_v, num_calc_u
    
    for _ in range(num_iters):
        if first_nneg:
            # In batch first, v has negative elements.
            v = _nonlin_solve(u, a, v, _lambda=beta, rcond=rcond, eps=eps, solve_ax=True, num_iters=num_calc_v)
            # In batch first, u has only non-negative elements.
            u = _nonlin_solve(v, a, u, _lambda=alpha, rcond=rcond, eps=eps, solve_ax=False, num_iters=num_calc_u)
        else:
            # In batch first, u has only non-negative elements.
            u = _nonlin_solve(v, a, u, _lambda=alpha, rcond=rcond, eps=eps, solve_ax=False, num_iters=num_calc_u)
            # In batch first, v has negative elements.
            v = _nonlin_solve(u, a, v, _lambda=beta, rcond=rcond, eps=eps, solve_ax=True, num_iters=num_calc_v)
    return u, v