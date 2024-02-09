import itertools

import numpy as np
from scipy.stats import skewnorm, norm
from scipy.special import erf


def get_bags(unique_ncharges):
    combs = list(itertools.combinations(unique_ncharges, r=2))
    combs = [list(x) for x in combs]
    # add self interaction
    self_combs = [[x, x] for x in unique_ncharges]
    combs += self_combs
    return combs


def get_mu_sigma(R):
    mu = R / 2
    sigma = R / 8
    return mu, sigma


def get_gaussian(x, R):
    mu, sigma = get_mu_sigma(R)
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def get_skew_gaussian_l(x, R, Z_J):
    mu = 0.5*R
    sigma = 0.125*R
    # 3.2 s for 100 mol
    #return Z_J * skewnorm.pdf(x, Z_J, mu, sigma)
    # 2.4 s for 100 mol
    #return Z_J * (1 + erf(Z_J * (x-mu)/sigma/np.sqrt(2))) * norm.pdf(x, mu, sigma)
    # 1 s for 100 mol
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    e = 1.0 + erf(Z_J * X)
    return Z_J * g * e



def get_skew_gaussian(x, R, Z_I, Z_J, variation="l"):
    mu, sigma = get_mu_sigma(R)
    if variation == "l":
        func = Z_J * skewnorm.pdf(x, Z_J, mu, sigma)
    elif variation == "n":
        func = Z_I * skewnorm.pdf(x, Z_J, mu, sigma)
    return func


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
        for j, ncharge_b in enumerate(ncharges):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < Rcut:
                if i > j:
                        twobodyrep[bag_idx[ncharge_a]] += get_skew_gaussian_l(grid, R, ncharge_b)
                if j < i:
                        twobodyrep[bag_idx[ncharge_b]] += get_skew_gaussian_l(grid, R, ncharge_a)

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
        coords_a = coords[i]
        for j in range(len(ncharges)):
            ncharge_b = ncharges[j]
            coords_b = coords[j]

            if i != j:
                R = np.linalg.norm(coords_b - coords_a)
                if R < Rcut:
                    twobodyrep += get_skew_gaussian(
                        grid, R, ncharge_a, ncharge_b, variation="n"
                    )

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
