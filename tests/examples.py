import numpy as np

# f(x) = x^TQx, 3. d. i.
def func_3d_i(x, eval_Hessian=False):
    # evaluate at x:...
    f = None  # scalar function value
    g = None  # gradient vector
    h = None  # Hessian matrix

    # Function definition and properties below: ...

    Q = np.array([[1, 0], [0, 1]])

    f = x.dot(Q @ x)
    g = 2 * (Q @ x)
    if eval_Hessian:
        h = 2 * Q

    return f, g, h


# f(x) = x^TQx, 3. d. ii.
def func_3d_ii(x, eval_Hessian=False):
    # evaluate at x:...
    f = None  # scalar function value
    g = None  # gradient vector
    h = None  # Hessian matrix

    # Function definition and properties below: ...

    Q = np.array([[1, 0], [0, 100]])

    f = x.dot(Q @ x)
    g = 2 * Q @ x
    if eval_Hessian:
        h = 2 * Q

    return f, g, h


# f(x) = x^TQx, 3. d. iii.
def func_3d_iii(x, eval_Hessian=False):
    # evaluate at x:...
    f = None  # scalar function value
    g = None  # gradient vector
    h = None  # Hessian matrix

    # Function definition and properties below: ...

    Q_matrix1 = np.array([[np.sqrt(3) / 2., -0.5], [0.5, np.sqrt(3) / 2.]])
    Q_matrix2 = np.array([[100, 0], [0, 1]])
    Q_matrix3 = np.array([[np.sqrt(3) / 2., -0.5], [0.5, np.sqrt(3) / 2.]])

    Q = Q_matrix1.dot(Q_matrix2 @ Q_matrix3)

    f = x.dot(Q @ x)
    g = 2 * Q @ x
    if eval_Hessian:
        h = 2 * Q

    return f, g, h


# The Rosenbrock Function, 3. e.
def func_3e(x, eval_Hessian=False):
    # evaluate at x:...
    f = None  # scalar function value
    g = None  # gradient vector
    h = None  # Hessian matrix

    # Function definition and properties below: ...

    assert x.size == 2
    x1 = x[0]
    x2 = x[1]

    f = 100 * ((x2 - (x1 ** 2)) ** 2) + ((1 - x1) ** 2)
    g = np.array([((-400 * (x2 - (x1 ** 2)) * x1) - (2 * (1 - x1))), (200 * (x2 - (x1 ** 2)))])
    if eval_Hessian:
        h = np.array([[((-400 * x2) + (1200 * (x1 ** 2)) + 2), (-400 * x1)], [(-400 * x1), (200)]])

    return f, g, h


# Linear Function 3. f.
def func_3f(x, eval_Hessian=False):
    # evaluate at x:...
    f = None  # scalar function value
    g = None  # gradient vector
    h = None  # Hessian matrix

    # Function definition and properties below: ...

    # compute vector a: ...
    a = []
    for i in range(x.size):
        a.append(i)
    a = np.array(a)

    f = a.dot(x)
    g = a
    if eval_Hessian:
        h = np.zeros((g.size, g.size))

    return f, g, h


# Function 3. g.
def func_3g(x, eval_Hessian=False):
    # evaluate at x:...
    f = None  # scalar function value
    g = None  # gradient vector
    h = None  # Hessian matrix

    # Function definition and properties below: ...

    assert x.size == 2
    x1 = x[0]
    x2 = x[1]

    f = (np.exp((x1 + (3 * x2) - 0.1)) + np.exp((x1 - (3 * x2) - 0.1)) + np.exp((-x1 - 0.1)))
    g = np.array([(np.exp((x1 + (3 * x2) - 0.1)) + np.exp((x1 - (3 * x2) - 0.1)) - np.exp((-x1 - 0.1))),
                  ((3 * np.exp((x1 + (3 * x2) - 0.1))) - (3 * np.exp((x1 - (3 * x2) - 0.1))))])
    if eval_Hessian:
        h = np.array([[(np.exp((x1 + (3 * x2) - 0.1)) + np.exp((x1 - (3 * x2) - 0.1)) + np.exp((-x1 - 0.1))),
                       ((3 * np.exp((x1 + (3 * x2) - 0.1))) - (3 * np.exp((x1 - (3 * x2) - 0.1))))],
                      [((3 * np.exp((x1 + (3 * x2) - 0.1))) - (3 * np.exp((x1 - (3 * x2) - 0.1)))),
                       ((9 * np.exp((x1 + (3 * x2) - 0.1))) + (9 * np.exp((x1 - (3 * x2) - 0.1))))]])

    return f, g, h