import numpy as np
import itertools
from scipy.stats import skewnorm

def get_bags(unique_ncharges):
    combs = list(itertools.combinations(unique_ncharges, r=2))
    combs = [list(x) for x in combs]
    # add self interaction
    self_combs = [[x,x] for x in unique_ncharges]
    combs += self_combs
    return combs

def get_mu_sigma(R):
    mu = R / 2
    sigma = R / 8
    return mu, sigma 

def get_gaussian(x, R):
    mu, sigma = get_mu_sigma(R)
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-(x - mu)**2 / (2*sigma**2))

def get_skew_gaussian(x, R, Z_I, Z_J, variation='l'):
    mu, sigma = get_mu_sigma(R)
    if variation == 'l':
        func = Z_J * skewnorm.pdf(x, Z_J, mu, sigma)
    elif variation == 'n':
        func = Z_I * skewnorm.pdf(x, Z_J, mu, sigma)
    return func

def get_bbgr_all_bags(ncharges, coords, elements=[1,6,7,8,9,17],
        Rcut=3.5, gridspace=0.03, onebody=False): 
    for ncharge in ncharges:
        if ncharge not in elements:
            print('warning!', ncharge, 'not included in rep')

    ncharges = [x for x in ncharges if x in elements]
    bags = get_bags(elements)
    grid = np.arange(0, Rcut, gridspace)
    size = len(grid)

    if onebody:
        onebodyrep = np.zeros(len(elements))
        for i, elem in enumerate(elements):
            for ncharge in ncharges:
                if ncharge == elem:
                    onebodyrep[i] += 1

    twobodyrep = np.zeros((len(bags), size))

    for k, bag in enumerate(bags):
        for i, ncharge_a in enumerate(ncharges):
            coords_a = coords[i]
            for j, ncharge_b in enumerate(ncharges):
                if i!=j:
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

    if onebody:
        rep = np.concatenate((onebodyrep, twobodyrep))
    else:
        rep = twobodyrep
    return rep

def get_bbgr_no_bags(ncharges, coords, elements=[1,6,7,8,9,17],
        Rcut=3.5, gridspace=0.03):

    for ncharge in ncharges:
        if ncharge not in elements:
            print('warning!', ncharge, 'not included in rep')

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
                    twobodyrep += get_skew_gaussian(grid, R, ncharge_a, ncharge_b, variation='n')
    return twobodyrep


def get_bbgr_linear_bags(ncharges, coords, elements=[1,6,7,8,9,17],
                        Rcut=3.5, gridspace=0.03):

    for ncharge in ncharges:
        if ncharge not in elements:
            print('warning!', ncharge, 'not included in rep')

    ncharges = [x for x in ncharges if x in elements]

    bags = np.array(elements)
    grid = np.arange(0, Rcut, gridspace)
    size = len(grid)
    twobodyrep = np.zeros((len(bags), size))

    for k, bag in enumerate(bags):
        for i, ncharge_a in enumerate(ncharges):
            coords_a = coords[i]
            for j in range(len(ncharges)):
                if i != j:
                    ncharge_b = ncharges[j]
                    coords_b = coords[j]

                    R = np.linalg.norm(coords_b - coords_a)

                    if R < Rcut:
                        if ncharge_a == bag:
                            twobodyrep[k] += get_skew_gaussian(grid, R, ncharge_a, ncharge_b, variation='l')

    twobodyrep = np.concatenate(twobodyrep)
    return twobodyrep 

