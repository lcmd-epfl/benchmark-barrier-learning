import numpy as np
from periodictable import elements

pt = {}
for el in elements:
    pt[el.symbol] = el.number

def reader(xyz):
    with open(xyz, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    nat = int(lines[0])
    start_idx = 2
    end_idx = start_idx + nat

    atomtypes = []
    coords = []

    for line_idx in range(start_idx, end_idx):
        line = lines[line_idx]
        atomtype, x, y, z = line.split()
        atomtypes.append(str(atomtype))
        coords.append([float(x), float(y), float(z)])

    ncharges = [convert_symbol_to_ncharge(x) for x in atomtypes]

    assert len(atomtypes) == nat
    assert len(coords) == nat
    assert len(ncharges) == nat
    return np.array(atomtypes), np.array(ncharges), np.array(coords)

def convert_symbol_to_ncharge(symbol):
    return pt[symbol]
