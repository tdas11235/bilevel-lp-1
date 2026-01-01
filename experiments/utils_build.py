import casadi as ca


def poly_row(q, W1, W2=None):
    """
    Build a single row of A(q):
    - linear:  W1 @ q
    - quadratic: qᵀ W2 q
    """
    lin = W1 @ q
    if W2 is None:
        return lin
    quad = ca.dot(q, W2 @ q)
    return lin + quad
