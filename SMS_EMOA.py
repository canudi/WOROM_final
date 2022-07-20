import numpy as np
import pandas as pd


def dominate(x1: np.array, x2: np.array):
    return (x1 <= x2).all() and (x1 < x2).any()


def pareto_ranking(ObjFncVectors):
    X = pd.DataFrame(ObjFncVectors)
    return np.array([np.sum(X.T.apply(lambda x: dominate(x, X.T[i]), axis=0)) for i in range(X.shape[0])])


def select(gen, f_vals, mu, v):
    """
    :param gen:
    :param f_vals:
    :param mu:
    :param v:           hyper-volume of last generation.
    :return:
    """
    # first criteria - choose minimum rank samples
    ranks = pareto_ranking(f_vals)
    min_rank = ranks.min()
    selected = gen[ranks == min_rank]
    while selected.shape[0] + gen[ranks == min_rank + 1] < mu:
        min_rank += 1
        selected = np.concatenate((selected, gen[ranks == min_rank]), axis=0)
    if selected.shape[0] == mu:
        if ranks.min() == 0:            # found mu non-dominated samples
            return selected, 0          # let SMS-EMOA return selected
        # else => found mu dominated
        return selected, np.inf         # let SMS-EMOA continue with selected as next generation

    # second criteria - use hyper-volume
    r = 2 * np.max(f_vals, axis=0)     # the worst solutions sample is twice the worst solution found
    contribution, hv = hyper_volume(f_vals[ranks == min_rank], mu - selected.shape[0], r)
    selected = np.concatenate((selected,
                               (gen[ranks == min_rank])
                               [np.argsort(contribution)[-(mu - selected.shape[0]):]]))
    if min_rank == 0:
        return selected, 0
    return selected, v - hv, hv
    # for objective_func in range(f_vals.shape[1]):
    #     next_gen = next_gen.sort(key=lambda x: x[1][objective_func])


def hyper_volume(f_vals, mu, r):
    """
    Calculates the dominated hyper-volume contribution of each individual in the generation.
    :param f_vals:  m objective functions values for each of the n individuals  - ndarray.
    :param r:       a relation f_values point symbolizing the worst solution f_values.
    :param mu:      number of samples to select.
    :return:        the contribution (with - without) of each individual        - ndarray.
    """
    # abs_vals, abs_r = np.abs(f_vals), np.abs(r)
    ribs = np.abs(f_vals - r)               # ribs is a matrix of the ribs of each box
    box_volumes = np.prod(ribs, axis=1)     # d=1
    without = np.array([np.sum(np.delete(box_volumes), i, axis=0) for i in range(box_volumes.shape[0])])
    most_contributing = np.argsort(np.sum(box_volumes) - without)[-mu:]
    volume = np.sum(box_volumes[most_contributing])
    return most_contributing, volume

def select_mates(pop, f_vals):
    """
    todo
    selection with roulette of normalized f_vals
    :param pop:     population to select from.
    :param f_vals:  m objective functions values for each of the n individuals  - ndarray.
    :return:        (x1, x2) - tuple of selected individuals to mate.
    """


def recombine(x1, x2):
    # todo
    pass


def mutate(x):
    # todo
    pass


def variate(curr_gen, f_vals, sigma):
    """

    :param curr_gen:
    :param f_vals:
    :param sigma:
    :return:
    """
    next_gen = np.zeros((sigma, curr_gen.shape[1]))
    for i in range(sigma):
        p1, p2 = select_mates(curr_gen, f_vals)
        offspring = mutate(recombine(p1, p2))
        next_gen[i] = offspring
    return next_gen


def SMS_EMOA(init_pop, F, mu, sigma, epsilon=1e-3):
    """
    :param init_pop:    initial population.
    :param F:           multy-objective functions
    :param mu:
    :param sigma:
    :param epsilon:
    :return:
    """
    # init random population of samples degree d in boundaries - pop
    # gen = np.random.rand(sigma, d) * (boundaries[1] - boundaries[0]) + boundaries[0]
    f_vals = np.array([[f(x) for f in F] for x in init_pop])
    improve = np.inf
    gen = init_pop
    c, v = hyper_volume(f_vals, mu, 2 * np.max(f_vals, axis=0))
    while improve > epsilon:
        next_gen = variate(gen, f_vals, sigma)          # next_gen - sigma new samples
        nf_vals = np.array([[f(x) for f in F] for x in next_gen])
        gen, improve, v = select(np.concatenate((gen, next_gen), axis=0),
                                 np.concatenate((f_vals, nf_vals), axis=0), mu, v)
    return gen
