import itertools
import numpy as np
from scipy.stats import skewnorm
from scipy.special import erf


def get_bags(unique_ncharges):
    combs = list(itertools.combinations(unique_ncharges, r=2))
    combs = [list(x) for x in combs]
    # add self interaction
    self_combs = [[x, x] for x in unique_ncharges]
    combs += self_combs
    return combs


def get_mu_sigma(R):
    mu = R * 0.5
    sigma = R * 0.125
    return mu, sigma


def get_gaussian(x, R):
    mu, sigma = get_mu_sigma(R)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    return g


def get_skew_gaussian(x, R, Z_I, Z_J):
    mu, sigma = get_mu_sigma(R)
    # the same as `Z_I * scipy.stats.skewnorm.pdf(x, Z_J, mu, sigma)` but faster
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    e = 1.0 + erf(Z_J * X)
    return Z_I * g * e


def get_skew_gaussian_l_both(x, R, Z_I, Z_J):
    mu, sigma = get_mu_sigma(R)
    # a = Z_J * scipy.stats.skewnorm.pdf(x, Z_J, mu, sigma)
    # b = Z_I * scipy.stats.skewnorm.pdf(x, Z_I, mu, sigma)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    e = 1.0 + erf(Z_J * X)
    a = Z_J * g * e
    if Z_I==Z_J:
        return a, a
    e = 1.0 + erf(Z_I * X)
    b = Z_I * g * e
    return a, b


def get_skew_gaussian_n_both(x, R, Z_I, Z_J):
    mu, sigma = get_mu_sigma(R)
    # a = Z_I * scipy.stats.skewnorm.pdf(x, Z_J, mu, sigma)
    # b = Z_J * scipy.stats.skewnorm.pdf(x, Z_I, mu, sigma)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    e = 1.0 + erf(Z_J * X)
    a = Z_I * g * e
    if Z_I==Z_J:
        return 2.0*a
    e = 1.0 + erf(Z_I * X)
    b = Z_J * g * e
    return a + b


def get_b2r2_a_molecular(
    ncharges, coords, elements=[1, 6, 7, 8, 9, 17], Rcut=3.5, gridspace=0.03
):
    ncharges = [x for x in ncharges if x in elements]
    bags = get_bags(elements)
    grid = np.arange(0, Rcut, gridspace)
    size = len(grid)
    twobodyrep = np.zeros((len(bags), size))

    for k, bag in enumerate(bags):
        for i, ncharge_a in enumerate(ncharges):
            coords_a = coords[i]
            for j, ncharge_b in enumerate(ncharges):
                if i != j:
                    ncharge_b = ncharges[j]
                    coords_b = coords[j]
                    # check whether to use it
                    bag_candidate = [ncharge_a, ncharge_b]
                    inv_bag_candidate = [ncharge_b, ncharge_a]
                    if bag == bag_candidate or bag == inv_bag_candidate:
                        R = np.linalg.norm(coords_b - coords_a)
                        if R < Rcut:
                            twobodyrep[k] += get_gaussian(grid, R)

    twobodyrep = np.concatenate(twobodyrep)
    return twobodyrep


def get_b2r2_a(
    reactants_ncharges,
    products_ncharges,
    reactants_coords,
    products_coords,
    elements=[1, 6, 7, 8, 9, 17],
    Rcut=3.5,
    gridspace=0.03,
):
    """
    Reactants_ncharges is a list of lists where the outer list is the total number
    of reactions and the inner list is the number of reactants in each reaction
    Same for coords, and for products
    """
#    all_ncharges_reactants = [np.concatenate(x) for x in reactants_ncharges]
 #   u_ncharges_reactants = np.unique(np.concatenate(all_ncharges_reactants))
 #   all_ncharges_products = [np.concatenate(x) for x in products_ncharges]
 #   u_ncharges_products = np.unique(np.concatenate(all_ncharges_products))
 #   u_ncharges = np.unique(np.concatenate((u_ncharges_reactants, u_ncharges_products)))

#    for ncharge in u_ncharges:
  #      if ncharge not in elements:
 #           print("warning!", ncharge, "not included in rep")

    b2r2_a_reactants = np.sum(
        [
            [
                get_b2r2_a_molecular(
                    reactants_ncharges[i][j],
                    reactants_coords[i][j],
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(reactants_ncharges[i]))
            ]
            for i in range(len(reactants_ncharges))
        ],
        axis=1,
    )

    b2r2_a_products = np.sum(
        [
            [
                get_b2r2_a_molecular(
                    products_ncharges[i][j],
                    products_coords[i][j],
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(products_ncharges[i]))
            ]
            for i in range(len(products_ncharges))
        ],
        axis=1,
    )

    b2r2_a = b2r2_a_products - b2r2_a_reactants
    return b2r2_a


def get_b2r2_l_molecular(
    ncharges, coords, elements=[1, 6, 7, 8, 9, 17], Rcut=3.5, gridspace=0.03,
):

    for ncharge in ncharges:
        if ncharge not in elements:
            print("warning!", ncharge, "not included in rep")

    ncharges = [x for x in ncharges if x in elements]

    bags = np.array(elements)
    grid = np.arange(0, Rcut, gridspace)
    size = len(grid)
    twobodyrep = np.zeros((len(bags), size))

    bag_idx = {q: i for i,q in enumerate(bags)}

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < Rcut:
                a, b = get_skew_gaussian_l_both(grid, R, ncharge_a, ncharge_b)
                twobodyrep[bag_idx[ncharge_a]] += a
                twobodyrep[bag_idx[ncharge_b]] += b

    twobodyrep = np.concatenate(twobodyrep)
    return twobodyrep


def get_b2r2_l(
    reactants_ncharges,
    products_ncharges,
    reactants_coords,
    products_coords,
    elements=[1, 6, 7, 8, 9, 17],
    Rcut=3.5,
    gridspace=0.03,
):
    """
    Reactants_ncharges is a list of lists where the outer list is the total number
    of reactions and the inner list is the number of reactants in each reaction
    Same for coords, and for products
    """
    all_ncharges_reactants = [np.concatenate(x) for x in reactants_ncharges]
    u_ncharges_reactants = np.unique(np.concatenate(all_ncharges_reactants))
    all_ncharges_products = [np.concatenate(x) for x in products_ncharges]
    u_ncharges_products = np.unique(np.concatenate(all_ncharges_products))
    u_ncharges = np.unique(np.concatenate((u_ncharges_reactants, u_ncharges_products)))

    for ncharge in u_ncharges:
        if ncharge not in elements:
            print("warning!", ncharge, "not included in rep")

    b2r2_l_reactants = np.sum(
        [
            [
                get_b2r2_l_molecular(
                    reactants_ncharges[i][j],
                    reactants_coords[i][j],
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(reactants_ncharges[i]))
            ]
            for i in range(len(reactants_ncharges))
        ],
        axis=1,
    )

    b2r2_l_products = np.sum(
        [
            [
                get_b2r2_l_molecular(
                    products_ncharges[i][j],
                    products_coords[i][j],
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(products_ncharges[i]))
            ]
            for i in range(len(products_ncharges))
        ],
        axis=1,
    )

    b2r2_l = b2r2_l_products - b2r2_l_reactants
    return b2r2_l


def get_b2r2_n_molecular(
    ncharges, coords, elements=[1, 6, 7, 8, 9, 17], Rcut=3.5, gridspace=0.03
):

    for ncharge in ncharges:
        if ncharge not in elements:
            print("warning!", ncharge, "not included in rep")

    ncharges = [x for x in ncharges if x in elements]

    grid = np.arange(0, Rcut, gridspace)
    size = len(grid)
    twobodyrep = np.zeros(size)

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < Rcut:
                twobodyrep += get_skew_gaussian_n_both(grid, R, ncharge_b, ncharge_a)

    return twobodyrep


def get_b2r2_n(
    reactants_ncharges,
    products_ncharges,
    reactants_coords,
    products_coords,
    elements=[1, 6, 7, 8, 9, 17],
    Rcut=3.5,
    gridspace=0.03,
):
    """
    Reactants_ncharges is a list of lists where the outer list is the total number
    of reactions and the inner list is the number of reactants in each reaction
    Same for coords, and for products
    """
    all_ncharges_reactants = [np.concatenate(x) for x in reactants_ncharges]
    u_ncharges_reactants = np.unique(np.concatenate(all_ncharges_reactants))
    all_ncharges_products = [np.concatenate(x) for x in products_ncharges]
    u_ncharges_products = np.unique(np.concatenate(all_ncharges_products))
    u_ncharges = np.unique(np.concatenate((u_ncharges_reactants, u_ncharges_products)))

    for ncharge in u_ncharges:
        if ncharge not in elements:
            print("warning!", ncharge, "not included in rep")

    b2r2_n_reactants = np.sum(
        [
            [
                get_b2r2_n_molecular(
                    reactants_ncharges[i][j],
                    reactants_coords[i][j],
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(reactants_ncharges[i]))
            ]
            for i in range(len(reactants_ncharges))
        ],
        axis=1,
    )

    b2r2_n_products = np.sum(
        [
            [
                get_b2r2_n_molecular(
                    products_ncharges[i][j],
                    products_coords[i][j],
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(products_ncharges[i]))
            ]
            for i in range(len(products_ncharges))
        ],
        axis=1,
    )

    b2r2_n = b2r2_n_products - b2r2_n_reactants
    return b2r2_n
