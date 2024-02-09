import numpy as np
import pandas as pd
import qml
from glob import glob
from periodictable import elements
import os
from src.fingerprints import get_MFP, get_DRFP
from src.b2r2 import get_b2r2_a_molecular, get_b2r2_l_molecular, get_b2r2_n_molecular

pt = {el.symbol: el.number for el in elements}

def convert_symbol_to_ncharge(symbol):
    return pt[symbol]

def pad_indices(idx):
    idx = str(idx)
    if len(idx) < 6:
        pad_len = 6 - len(idx)
        pad = '0'*pad_len
        idx = pad + idx
    return idx

def check_alt_files(list_files):
    files = []
    if len(list_files) < 3:
        return list_files
    for file in list_files:
        if "_alt" in file:
            dup_file_label = file.split("_alt.xyz")[0]
    for file in list_files:
        if dup_file_label in file:
            if "_alt" in file:
                files.append(file)
        else:
            files.append(file)
    return files

def create_mol_obj(atomtypes, ncharges, coords):
    if len(atomtypes) == 0:
        raise ValueError("mol has no atoms")
    mol = qml.Compound()
    mol.atomtypes = atomtypes
    mol.nuclear_charges = ncharges
    mol.coordinates = coords
    return mol

def reader(xyz):
    if not os.path.exists(xyz):
        return [], [], []
    with open(xyz, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    try:
        nat = int(lines[0])
    except:
        raise ValueError('file', xyz, 'is empty')

    if nat == 0:
        raise ValueError('file', xyz, 'has zero atoms')
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

    if len(atomtypes) == 0:
        raise ValueError('file', xyz, 'has no atoms')

    return np.array(atomtypes), np.array(ncharges), np.array(coords)

class TWODIM:
    """Simple 2D reps based on SMILES"""
    def __init__(self, proparg_stereo=False, proparg_combinatorial=False):
        self.barriers = []
        self.proparg_stereo = proparg_stereo
        self.proparg_combinatorial = proparg_combinatorial
        if proparg_stereo:
            self.proparg_path = "data/proparg/data_fixarom_smiles_stereo.csv"
        elif proparg_combinatorial:
            self.proparg_path = "data/proparg/data_fixarom_smiles.csv"
        else:
            self.proparg_path = "data/proparg/data.csv"

    def get_proparg_MFP(self, subset=None):
        data = pd.read_csv(self.proparg_path, index_col=0)
        self.barriers = data['Eafw'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        if subset:
            rxn_smiles = np.array(rxn_smiles)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(rxn_smiles)
            assert len(self.barriers) == subset
        mfps = [get_MFP(x, self.proparg_stereo or self.proparg_combinatorial) for x in rxn_smiles]
        return np.vstack(mfps)

    def get_proparg_DRFP(self, subset=None):
        data = pd.read_csv(self.proparg_path, index_col=0)
        self.barriers = data['Eafw'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        if subset:
            rxn_smiles = np.array(rxn_smiles)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(rxn_smiles)
            assert len(self.barriers) == subset
        drfps = [get_DRFP(x) for x in rxn_smiles]
        return np.vstack(drfps)

    def get_cyclo_MFP(self, subset=None):
        data = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = data['G_act'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        if subset:
            rxn_smiles = np.array(rxn_smiles)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(rxn_smiles)
            assert len(self.barriers) == subset
        mfps = [get_MFP(x) for x in rxn_smiles]
        return np.vstack(mfps)

    def get_cyclo_DRFP(self, subset=None):
        data = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = data['G_act'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        if subset:
            rxn_smiles = np.array(rxn_smiles)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(rxn_smiles)
            assert len(self.barriers) == subset
        drfps = [get_DRFP(x) for x in rxn_smiles]
        return np.vstack(drfps)

    def get_gdb_MFP(self, subset=None):
        data = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv", index_col=0)
        self.barriers = data['dE0'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        if subset:
            rxn_smiles = np.array(rxn_smiles)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(rxn_smiles)
            assert len(self.barriers) == subset
        mfps = [get_MFP(x) for x in rxn_smiles]
        return np.vstack(mfps)

    def get_gdb_DRFP(self, subset=None):
        data = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv", index_col=0)
        self.barriers = data['dE0'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        if subset:
            rxn_smiles = np.array(rxn_smiles)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(rxn_smiles)
            assert len(self.barriers) == subset
        drfps = [get_DRFP(x) for x in rxn_smiles]
        return np.vstack(drfps)

class QML:
    """class for 3d reps: slatm and b2r2"""
    def __init__(self):
        self.ncharges = []
        self.unique_ncharges = []
        self.max_natoms = 0
        self.atomtype_dict = {"H": 0, "C": 0, "N": 0, "O": 0, "S": 0, "Cl":0,
                                "F":0}
        self.mols_products = []
        self.mols_reactants = [[]]
        return

    def get_proparg_data(self, xtb=False, subset=None):
        df = pd.read_csv("data/proparg/data.csv", index_col=0)
        if xtb:
            data_dir = 'data/proparg/xyz-xtb/'
        else:
            data_dir = 'data/proparg/xyz/'
        indices = [''.join(x) for x in zip(df['mol'].to_list(), df['enan'].to_list())]
        if subset:
            indices = np.array(indices)[:subset]

        reactants_files = []
        products_files = []
        for idx in indices:
            r_xyz, p_xyz = [f'{data_dir}{idx}.{x}.xyz' for x in ('r', 'p')]
            reactants_files.append(r_xyz)
            products_files.append(p_xyz)

        all_mols = [qml.Compound(x) for x in reactants_files + products_files]
        self.barriers = df.Eafw.to_numpy()
        if subset:
            self.barriers = self.barriers[:subset]
            assert len(self.barriers) == len(reactants_files)
            assert len(self.barriers) == subset
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [[qml.Compound(x)] for x in reactants_files]
        self.mols_products = [[qml.Compound(x)] for x in products_files]

        return

    def get_proparg_data_xtb(self):
        self.get_proparg_data(xtb=True)
        return

    def get_GDB7_ccsd_data(self, subset=None):
        df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
        self.barriers = df['dE0'].to_numpy()
        indices = df['idx'].apply(pad_indices).to_list()
        if subset:
            self.barriers = self.barriers[:subset]
            indices = indices[:subset]
            assert len(self.barriers) == subset
            assert len(self.barriers) == len(indices)

        r_mols = []
        p_mols = []
        for idx in indices:
            filedir = 'data/gdb7-22-ts/xyz/'+idx
            rfile = filedir + '/r' + idx + '.xyz'
            r_atomtypes, r_ncharges, r_coords = reader(rfile)
            r_coords = r_coords * 0.529177 # bohr to angstrom
            r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
            r_mols.append([r_mol])

            # multiple p files
            pfiles = glob(filedir+'/p*.xyz')
            sub_pmols = []
            for pfile in pfiles:
                p_atomtypes, p_ncharges, p_coords = reader(pfile)
                p_coords = p_coords * 0.529177
                p_mol = create_mol_obj(p_atomtypes, p_ncharges, p_coords)
                sub_pmols.append(p_mol)
            p_mols.append(sub_pmols)
        self.mols_reactants = r_mols
        self.mols_products = p_mols
        all_r_mols = np.concatenate(r_mols)
        self.ncharges = [x.nuclear_charges for x in all_r_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        return

    def get_GDB7_xtb_data(self, subset=None):
        df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
        self.barriers = df['dE0'].to_numpy()
        indices = df['idx']
        if subset:
            self.barriers = self.barriers[:subset]
            indices = np.array(indices)[:subset]
            assert len(self.barries) == len(indices)
            assert len(self.barriers) == subset

        r_mols = []
        p_mols = []
        good_indices = []
        for i, idx in enumerate(indices):
            filedir = 'data/gdb7-22-ts/xyz-xtb/' + str(idx)
            rfile = filedir + '/Reactant_' + str(idx) + '_0_opt.xyz'
            r_atomtypes, r_ncharges, r_coords = reader(rfile)
            r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)

            # multiple p files
            pfiles = glob(filedir + '/Product_*.xyz')
            sub_pmols = []
            for pfile in pfiles:
                p_atomtypes, p_ncharges, p_coords = reader(pfile)
                p_mol = create_mol_obj(p_atomtypes, p_ncharges, p_coords)
                sub_pmols.append(p_mol)
            if len(sub_pmols) == 0:
                continue
            r_mols.append([r_mol])
            p_mols.append(sub_pmols)
            good_indices.append(i)
        self.mols_reactants = r_mols
        self.mols_products = p_mols

        assert len(self.mols_reactants) == len(self.mols_products)
        all_r_mols = np.concatenate(r_mols)
        self.ncharges = [x.nuclear_charges for x in all_r_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        self.barriers = self.barriers[good_indices]

        assert len(self.barriers) == len(self.mols_reactants)
        assert len(self.mols_reactants) == len(self.mols_products)
        return


    def get_cyclo_data(self, subset=None):
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = df['G_act'].to_numpy()
        indices = df['rxn_id'].to_list()
        self.indices = indices
        rxns = ["data/cyclo/xyz/"+str(i) for i in indices]
        if subset:
            rxns = np.array(rxns)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(rxns) == subset
            assert len(rxns) == len(self.barriers)

        reactants_files = []
        products_files = []
        for rxn_dir in rxns:
            reactants = glob(rxn_dir+"/r*.xyz")
            reactants = check_alt_files(reactants)
            assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"
            reactants_files.append(reactants)
            products = glob(rxn_dir+"/p*.xyz")
            products_files.append(products)

        mols_reactants = []
        mols_products = []
        ncharges_products = []
        for i in range(len(rxns)):
            mols_r = []
            mols_p = []
            ncharges_p = []
            for reactant in reactants_files[i]:
                mol = qml.Compound(reactant)
                mols_r.append(mol)
            for product in products_files[i]:
                mol = qml.Compound(product)
                mols_p.append(mol)
                ncharges_p.append(mol.nuclear_charges)
            ncharges_p = np.concatenate(ncharges_p)
            ncharges_products.append(ncharges_p)
            mols_reactants.append(mols_r)
            mols_products.append(mols_p)
        self.ncharges = ncharges_products
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges, axis=0))
        self.mols_reactants = mols_reactants
        self.mols_products = mols_products
        return

    def get_cyclo_xtb_data(self, subset=None):
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = df['G_act'].to_numpy()
        indices = df['rxn_id'].to_list()
        self.indices = indices
        if subset:
            self.indices = np.array(self.indices)[:subset]
            self.barriers = self.barriers[:subset]
            assert len(self.indices) == subset
            assert len(self.barriers) == len(self.indices)
        keep_indices = []
        reactants_files = []
        products_files = []
        for i, idx in enumerate(indices):
            rxn_dir = "data/cyclo/xyz-xtb/"
            reactants = glob(rxn_dir + f"Reactant_{idx}_*.xyz")
            if len(reactants) == 0:
                continue
            assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"
            reactants_files.append(reactants)
            products = [rxn_dir + f"Product_{idx}.xyz"]
            products_files.append(products)
            keep_indices.append(i) # keep those that are not skipped bc zero

        self.barriers = self.barriers[keep_indices]

        mols_reactants = []
        mols_products = []
        ncharges_products = []
        for i in range(len(keep_indices)):
            mols_r = []
            mols_p = []
            ncharges_p = []
            for reactant in reactants_files[i]:
                mol = qml.Compound(reactant)
                mols_r.append(mol)
            for product in products_files[i]:
                mol = qml.Compound(product)
                mols_p.append(mol)
                ncharges_p.append(mol.nuclear_charges)
            ncharges_p = np.concatenate(ncharges_p)
            ncharges_products.append(ncharges_p)
            mols_reactants.append(mols_r)
            mols_products.append(mols_p)
        self.ncharges = ncharges_products
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges, axis=0))
        self.mols_reactants = mols_reactants
        self.mols_products = mols_products

        assert len(self.mols_reactants) == len(self.mols_products)
        assert len(self.mols_reactants) == len(self.barriers), f"size mismatch between n mols {len(self.mols_reactants)} and n barriers {len(self.barriers)}"
        return

    def get_SLATM(self):
        mbtypes = qml.representations.get_slatm_mbtypes(self.ncharges)

        slatm_reactants = [
            np.array(
                [
                    qml.representations.generate_slatm(
                        x.coordinates, x.nuclear_charges, mbtypes, local=False
                    )
                    for x in reactants
                ]
            )
            for reactants in self.mols_reactants
        ]
        slatm_reactants_sum = np.array([sum(x) for x in slatm_reactants])

        slatm_products = [
            np.array(
                [
                    qml.representations.generate_slatm(
                        x.coordinates, x.nuclear_charges, mbtypes, local=False
                    )
                    for x in products
                ]
            )
            for products in self.mols_products
        ]

        ys = []
        for i, prod in enumerate(slatm_products):
            if prod.shape[0] == 0:
                print(i)
            y = sum(prod)
            ys.append(y)
        pys = np.array(ys)

        slatm_products = np.array([sum(x) for x in slatm_products])
        slatm_diff = slatm_products - slatm_reactants_sum

        return slatm_diff


    def get_b2r2_l(self, Rcut=3.5, gridspace=0.03):
        return self.get_b2r2_inner(Rcut=Rcut,
                                   gridspace=gridspace,
                                   get_b2r2_molecular=get_b2r2_l_molecular,
                                   combine=lambda r,p: p-r)


    def get_b2r2_a(self, Rcut=3.5, gridspace=0.03):
        return self.get_b2r2_inner(Rcut=Rcut,
                                   gridspace=gridspace,
                                   get_b2r2_molecular=get_b2r2_a_molecular,
                                   combine=lambda r,p: p-r)


    def get_b2r2_n(self, Rcut=3.5, gridspace=0.03):
        return self.get_b2r2_inner(Rcut=Rcut,
                                   gridspace=gridspace,
                                   get_b2r2_molecular=get_b2r2_n_molecular,
                                   combine=lambda r,p: np.concatenate((r, p), axis=1))


    def get_b2r2_inner(self, Rcut=3.5, gridspace=0.03, get_b2r2_molecular=None, combine=None):

        b2r2_reactants = np.array([ sum(
                get_b2r2_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=self.unique_ncharges,
                ) for x in reactants) for reactants in self.mols_reactants ])

        b2r2_products = np.array([ sum(
                get_b2r2_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=self.unique_ncharges,
                ) for x in products ) for products in self.mols_products ])

        return combine(b2r2_reactants, b2r2_products)
