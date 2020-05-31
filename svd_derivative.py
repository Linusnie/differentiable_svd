import numpy


def svd_derivative(a, a_dot):
    """
    Compute the derivative of singular value decomposition of a matrix
    A(k) = U(k)S(k)V(k)^T with respect to k at a point k0.
    Assumes there are no repeated singular values at k0.
    :param a: A(k) evaluated at k0
    :param a_dot: dA(k)/dk evaluated at k0
    :return: dU/dk(k0), dS/dk(k0), dV^T/dk(k0)
    """
    d = min(a.shape[0], a.shape[1])

    u, sigma, vt = numpy.linalg.svd(a, full_matrices=False)
    sigma_matrix = numpy.diag(sigma)

    a_tilde = u.T.dot(a_dot).dot(vt.T)
    sigma_dot = numpy.diag(a_tilde)

    diff_matrix = 1 / (sigma ** 2 - sigma[:, None] ** 2 + numpy.eye(d)) - numpy.eye(d)
    su = diff_matrix * (a_tilde.dot(sigma_matrix) + sigma_matrix.dot(a_tilde.T))
    sv = -diff_matrix * (sigma_matrix.dot(a_tilde) + a_tilde.T.dot(sigma_matrix))

    u_dot = u.dot(su)
    vt_dot = sv.dot(vt)

    return u_dot, sigma_dot, vt_dot
