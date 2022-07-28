from docplex.mp.model import Model
import numpy as np


def initialize_points(d):
    # find the minimum point of every funtion in F, add it to set_of_poits and return it (the ponits are X's)
    list_of_points = [np.zeros(d), np.ones(d)]
    return list_of_points


def CPLEX(F, set_of_points, list_of_y, alphaW):
    m = Model(name="DMA")
    numObjective = len(F)
    rObjective = range(numObjective)
    n = len(set_of_points)

    alpha = m.continuous_var(name='alpha', lb=0)

    x = m.continuous_var(name='x', lb=0) #adjust the right dim

    y = [f(x) for f in F]

    c1 = m.add_constraint(alpha == max([min([y[i] - list_of_y[j][i] for i in rObjective]) for j in range(n)]), ctname='const1')
    m.set_objective("max", alpha * alphaW - sum(y))
    sol = m.solve()
    print(sol.objective_value)
    print(x)
    print(y)

    return x, y


def f1(x):
    return np.dot(x.T, x)


def f2(x):
    return np.dot(x.T - 1, x - 1)


def main():
    max_num_of_points = 15
    F = [f1, f2]  # size 2 or 3
    d = 80
    list_of_points = initialize_points(d)
    list_of_y = [[f(x) for f in F] for x in list_of_points]

    alphaW = 0
    for i in range(len(list_of_y)):
        for j in range(i + 1, len(list_of_y)):
            for k in range(len(F)):
                alphaW = max(alphaW, list_of_y[i][k] - list_of_y[j][k])

    while len(list_of_points) < max_num_of_points: # while not terminate
        x, y = CPLEX(F, list_of_points, list_of_y, alphaW)
        list_of_points.append(x)
        list_of_y.append(y)


main()