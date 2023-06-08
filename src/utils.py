import matplotlib.pyplot as plt
import numpy as np
import tests.examples as examples


def generate_graph(function_name, results, limits):

    example_names = {'func_3d_i':' - quadratic example where contour lines are circles', 'func_3d_ii':' - quadratic example where contour lines are axis aligned ellipses',
                     'func_3d_iii':' - quadratic example where contour lines are rotated ellipses', 'func_3e':' - Rosenbrock function, contour lines are banana shaped ellipses',
                     'func_3f':' - linear function, contour lines are straight lines', 'func_3g':' - contour lines look like smoothed corner triangles'}

    f = getattr(examples, function_name)

    min_x1 = limits[0][0]
    max_x1 = limits[0][1]
    min_x2 = limits[1][0]
    max_x2 = limits[1][1]

    x_1_values = [x_1 for x_1 in range(int(min_x1), int(max_x1))]
    x_2_values = [x_2 for x_2 in range(int(min_x2), int(max_x2))]
    x_1_size = len(x_1_values)
    x_2_size = len(x_2_values)

    z = np.zeros([x_1_size, x_2_size])

    row = 0
    column = 0
    for x_1 in x_1_values:
        for x_2 in x_2_values:
            x_val = np.array((x_1, x_2))
            z[row][column] = f(x_val)[0]
            if column == x_1_size - 1:
                row += 1
                column = 0
            else:
                column += 1

    X1, X2 = np.meshgrid(x_1_values, x_2_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(function_name + example_names[function_name], fontsize=16)
    ax1.contour(X1, X2, z, levels=60)

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Contour Lines & Iteration Paths')

    ax2.set_xlabel('iteration')
    ax2.set_ylabel('f(x)')
    ax2.set_title('f(x) v.s. Iteration number')

    legend = []
    for r in results:

        legend.append(r['method'])

        if legend[-1] == 'Newton':
            c = 'blue'
        elif legend[-1] == 'GD':
            c = 'red'
        elif legend[-1] == 'BFGS':
            c = 'purple'
        elif legend[-1] == 'SR1':
            c = 'green'

        x = np.array(r['x_k'])
        x1, x2 = np.split(x, 2, axis=1)
        ax1.plot(x1, x2, color=c, linestyle='dashed')
        ax1.scatter(x1[-1], x2[-1], color=c)

        f_at_x_values = r['f_at_x_k']
        ax2.plot(f_at_x_values, color=c)

    ax1.legend(legend)
    ax2.legend(legend)

    plt.show()
