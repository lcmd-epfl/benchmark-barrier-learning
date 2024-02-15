import warnings
import itertools
import numpy as np
from scipy.special import erf
import tqdm


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


def get_b2r2_n_molecular(
    ncharges, coords, elements=[1, 6, 7, 8, 9, 17], Rcut=3.5, gridspace=0.03
):

    idx_relevant_atoms = np.where(np.sum(np.array(ncharges)==np.array(elements)[:,None], axis=0))
    ncharges = np.array(ncharges)[idx_relevant_atoms]
    coords = np.array(coords)[idx_relevant_atoms]

    grid = np.arange(0, Rcut, gridspace)
    twobodyrep = np.zeros_like(grid)

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < Rcut:
                twobodyrep += get_skew_gaussian_n_both(grid, R, ncharge_b, ncharge_a)

    return twobodyrep


def get_b2r2_a_molecular(ncharges, coords,
                         elements=[1, 6, 7, 8, 9, 17],
                         Rcut=3.5, gridspace=0.03):

    idx_relevant_atoms = np.where(np.sum(np.array(ncharges)==np.array(elements)[:,None], axis=0))
    ncharges = np.array(ncharges)[idx_relevant_atoms]
    coords = np.array(coords)[idx_relevant_atoms]

    bags = get_bags(elements)
    grid = np.arange(0, Rcut, gridspace)
    twobodyrep = np.zeros((len(bags), len(grid)))

    bag_idx      = {tuple(q1q2): i for i, q1q2 in enumerate(bags)}
    bag_idx.update({tuple(q1q2[::-1]): i for i, q1q2 in enumerate(bags)})

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < Rcut:
                twobodyrep[bag_idx[(ncharge_a, ncharge_b)]] += get_gaussian(grid, R)

    twobodyrep = 2.0*np.concatenate(twobodyrep)
    return twobodyrep



def get_b2r2_l_molecular(ncharges, coords,
                         elements=[1, 6, 7, 8, 9, 17],
                         Rcut=3.5, gridspace=0.03):

    idx_relevant_atoms = np.where(np.sum(np.array(ncharges)==np.array(elements)[:,None], axis=0))
    ncharges = np.array(ncharges)[idx_relevant_atoms]
    coords = np.array(coords)[idx_relevant_atoms]

    bags = np.array(elements)
    grid = np.arange(0, Rcut, gridspace)
    twobodyrep = np.zeros((len(bags), len(grid)))

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


def get_b2r2_l(mols_reactants, mols_products, elements=None, Rcut=3.5, gridspace=0.03):
    return get_b2r2_inner(mols_reactants, mols_products,
                          elements=elements,
                          Rcut=Rcut,
                          gridspace=gridspace,
                          get_b2r2_molecular=get_b2r2_l_molecular,
                          combine=lambda r,p: p-r)

def get_b2r2_a(mols_reactants, mols_products, elements=None, Rcut=3.5, gridspace=0.03):
    return get_b2r2_inner(mols_reactants, mols_products,
                          elements=elements,
                          Rcut=Rcut,
                          gridspace=gridspace,
                          get_b2r2_molecular=get_b2r2_a_molecular,
                          combine=lambda r,p: p-r)

def get_b2r2_n(mols_reactants, mols_products, elements=None, Rcut=3.5, gridspace=0.03):
    return get_b2r2_inner(mols_reactants, mols_products,
                          elements=elements,
                          Rcut=Rcut,
                          gridspace=gridspace,
                          get_b2r2_molecular=get_b2r2_n_molecular,
                          combine=lambda r,p: np.concatenate((r, p), axis=1))

def get_b2r2_inner(
                   mols_reactants, mols_products, elements=None,
                   Rcut=3.5, gridspace=0.03,
                   get_b2r2_molecular=None, combine=None):

    """
    Compute the B2R2 reaction representation for a set of reactions.

    Args:
        mols_reactants : List(List(mol))
        mols_products : List(List(mol))
            List of lists where the outer list is the total number of reactions
            and the inner list is the number of reactants/products in each reaction.
            `mol` is any object that provides fields .nuclear_charges and .coordinates (Å)
        elements : List(int) or 1D int ndarray
            Elements to identify bags. If None, use all the elements in mols_reactants and mols_products.
        Rcut : float
            Cutoff radius (Å)
        gridspace : float
            Grid spacing (Å)
        get_b2r2_molecular : func
            Function to compute the molecular representations, one of `get_b2r2_{l,n,a}_molecular`
        combine : func(r: ndarray, p: ndarray)
            Function to combine the reactants and products representations
    Returns:
        b2r2 : 2D float ndarray (number of reactions × number of features)
            B2R2 reaction representation
    """

    u_ncharges_reactants = np.unique(np.hstack([y.nuclear_charges for x in mols_reactants for y in x]))
    u_ncharges_products  = np.unique(np.hstack([y.nuclear_charges for x in mols_products  for y in x]))
    u_ncharges = np.unique(np.concatenate((u_ncharges_reactants, u_ncharges_products)))

    if elements is None:
        elements=u_ncharges
    else:
        setdiff = set(u_ncharges) - set(elements)
        if setdiff:
            warnings.warn(f"Elements {setdiff} are not included in the representation")

    b2r2_reactants = np.array([ sum(
            get_b2r2_molecular(
                x.nuclear_charges,
                x.coordinates,
                Rcut=Rcut,
                gridspace=gridspace,
                elements=elements,
            ) for x in reactants) for reactants in tqdm.tqdm(mols_reactants) ])

    b2r2_products = np.array([ sum(
            get_b2r2_molecular(
                x.nuclear_charges,
                x.coordinates,
                Rcut=Rcut,
                gridspace=gridspace,
                elements=elements,
            ) for x in products ) for products in tqdm.tqdm(mols_products) ])

    return combine(b2r2_reactants, b2r2_products)
