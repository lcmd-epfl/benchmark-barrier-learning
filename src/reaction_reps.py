import numpy as np
import pandas as pd
import qml
from glob import glob
from periodictable import elements
import os
from src.fingerprints import get_MFP, get_DRFP

pt = {}
for el in elements:
    pt[el.symbol] = el.number

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
        return None
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
        print('file', xyz, 'is empty')
        return [], [], [] 
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

class TWODIM:
    """Simple 2D reps based on SMILES"""

    def __init__(self):
        self.barriers = []

    def get_proparg_MFP(self):
        data = pd.read_csv("data/proparg/data.csv", index_col=0)
        self.barriers = data['dErxn'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        mfps = [get_MFP(x) for x in rxn_smiles]
        return np.vstack(mfps)

    def get_proparg_DRFP(self):
        data = pd.read_csv("data/proparg/data.csv", index_col=0)
        self.barriers = data['dErxn'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        drfps = [get_DRFP(x) for x in rxn_smiles]
        return np.vstack(drfps)

    def get_cyclo_MFP(self):
        data = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = data['G_act'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        mfps = [get_MFP(x) for x in rxn_smiles]
        return np.vstack(mfps)

    def get_cyclo_DRFP(self):
        data = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = data['G_act'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        drfps = [get_DRFP(x) for x in rxn_smiles]
        return np.vstack(drfps)

    def get_gdb_MFP(self):
        data = pd.read_csv("data/gdb7-22-ts/ccsdf12_dz.csv", index_col=0)
        self.barriers = data['dE0'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        mfps = [get_MFP(x) for x in rxn_smiles]
        return np.vstack(mfps)

    def get_gdb_DRFP(self):
        data = pd.read_csv("data/gdb7-22-ts/ccsdf12_dz.csv", index_col=0)
        self.barriers = data['dE0'].to_numpy()
        rxn_smiles = data['rxn_smiles']
        drfps = [get_DRFP(x) for x in rxn_smiles]
        return np.vstack(drfps)

class B2R2:
    """Reps available in B2R2 series"""

    def __init__(self):
        self.unique_ncharges = []
        self.barriers = []
        self.energies = []
        self.mols_reactants = [[]]
        self.mols_products = [[]]
        return

    def get_proparg_data(self):
        data = pd.read_csv("data/proparg/data.csv", index_col=0)
        reactants_files = [
            "data/proparg/data_react_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        products_files = [
            "data/proparg/data_prod_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        all_mols = [qml.Compound(x) for x in reactants_files + products_files]
        self.barriers = data.dErxn.to_numpy()
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [[qml.Compound(x)] for x in reactants_files]
        self.mols_products = [[qml.Compound(x)] for x in products_files]

        return

    def get_hydroform_data(self):
        co_df = pd.read_csv("data/hydroform/Co_clean.csv")
        names = co_df["name"].to_list()
        labels = [name[3:] for name in names]
        co_reactants = [
            "data/hydroform/geometries/co/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        co_products = [
            "data/hydroform/geometries/co/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.co_barriers = co_df["f_barr"].to_numpy()
        self.mols_reactants_co = [[qml.Compound(x)] for x in co_reactants]
        self.mols_products_co = [[qml.Compound(x)] for x in co_products]

        ir_df = pd.read_csv("data/hydroform/Ir_clean.csv")
        names = ir_df["name"].to_list()
        labels = [name[3:] for name in names]
        ir_reactants = [
            "data/hydroform/geometries/ir/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        ir_products = [
            "data/hydroform/geometries/ir/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.ir_barriers = ir_df["f_barr"].to_numpy()
        self.mols_reactants_ir = [[qml.Compound(x)] for x in ir_reactants]
        self.mols_products_ir = [[qml.Compound(x)] for x in ir_products]

        rh_df = pd.read_csv("data/hydroform/Rh_clean.csv")
        names = rh_df["name"].to_list()
        labels = [name[3:] for name in names]
        rh_reactants = [
            "data/hydroform/geometries/rh/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        rh_products = [
            "data/hydroform/geometries/rh/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.rh_barriers = rh_df["f_barr"].to_numpy()
        self.mols_reactants_rh = [[qml.Compound(x)] for x in rh_reactants]
        self.mols_products_rh = [[qml.Compound(x)] for x in rh_products]

        self.barriers = np.concatenate(
            (self.co_barriers, self.ir_barriers, self.rh_barriers), axis=0
        )

        all_reactants = co_reactants + ir_reactants + rh_reactants
        all_products = co_products + ir_products + rh_products
        list_reactants = [qml.Compound(x) for x in all_reactants]

        self.mols_reactants = [[qml.Compound(x)] for x in all_reactants]
        self.mols_products = [[qml.Compound(x)] for x in all_products]

        ncharges = [mol.nuclear_charges for mol in list_reactants]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        return

    def get_GDB7_ccsd_data(self):
        df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
        self.barriers = df['dE0'].to_numpy()
        indices = df['idx'].apply(pad_indices).to_list()

        r_mols = []
        p_mols = []
        for idx in indices:
            filedir = 'data/gdb7-22-ts/xyz/' + idx
            rfile = filedir + '/r' + idx + '.xyz'
            r_atomtypes, r_ncharges, r_coords = reader(rfile)
            r_coords = r_coords * 0.529177  # bohr to angstrom
            r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
            r_mols.append([r_mol])

            # multiple p files
            pfiles = glob(filedir + '/p*.xyz')
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

    def get_cyclo_data(self):
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = df['G_act'].to_numpy()
        indices = df['rxn_id'].to_list()
        self.indices = indices
        rxns = ["data/cyclo/xyz/" + str(i) for i in indices]

        reactants_files = []
        products_files = []
        for rxn_dir in rxns:
            reactants = glob(rxn_dir + "/r*.xyz")
            reactants = check_alt_files(reactants)
            assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"
            reactants_files.append(reactants)
            products = glob(rxn_dir + "/p*.xyz")
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

    def get_cyclo_xtb_data(self):
        # test set at lower quality geometry
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        barriers = df['G_act'].to_numpy()

        filedir = 'data/cyclo/xyz-xtb/'
        files = glob(filedir + "*.xyz")
        indices = np.unique([x.split("/")[-1].split("_")[1].strip('.xyz') for x in files])
        self.indices = [int(x) for x in indices]
        reactants_files = []
        products_files = []

        r_mols = []
        p_mols = []
        barriers = []
        for i, idx in enumerate(indices):
            idx = str(idx)
            # multiple r files
            if os.path.exists(filedir + "Reactant_" + idx + ".xyz"):
                rfile = filedir + "Reactant_" + idx + ".xyz"
                r_atomtypes, r_ncharges, r_coords = reader(rfile)
                r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
                sub_rmols.append(r_mol)
            elif os.path.exists(filedir + "Reactant_" + idx + "_0.xyz"):
                rfiles = glob(filedir + 'Reactant_' + idx + '_*.xyz')
                sub_rmols = []
                for rfile in rfiles:
                    r_atomtypes, r_ncharges, r_coords = reader(rfile)
                    r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
                    sub_rmols.append(r_mol)
            else:
                print("cannot find rfile")
                sub_rmols.append([None])

            sub_pmols = []
            pfile = filedir + "Product_" + idx + ".xyz"
            p_atomtypes, p_ncharges, p_coords = reader(pfile)
            p_mol = create_mol_obj(p_atomtypes, p_ncharges, p_coords)
            sub_pmols.append(p_mol)

            if None not in sub_pmols and None not in sub_rmols:
                r_mols.append(sub_rmols)
                p_mols.append(sub_pmols)
                barrier = df[df['rxn_id'] == int(idx)]['G_act'].item()
                barriers.append(barrier)
            else:
                print("skipping r mols", sub_rmols, 'and p mols', sub_pmols, 'for idx', idx)

        assert len(r_mols) == len(p_mols)
        assert len(r_mols) == len(barriers)
        self.mols_reactants = r_mols
        self.mols_products = p_mols
        self.barriers = barriers
        all_r_mols = np.concatenate(r_mols)
        self.ncharges = [x.nuclear_charges for x in all_r_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
        return

    def get_b2r2_l(self, Rcut=3.5, gridspace=0.03):
        b2r2_reactants = [
            [
                get_b2r2_l_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=self.unique_ncharges,
                )
                for x in reactants
            ]
            for reactants in self.mols_reactants
        ]
        # first index is reactants
        b2r2_reactants_sum = np.array([sum(x) for x in b2r2_reactants])

        b2r2_products = [
            [
                get_b2r2_l_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=self.unique_ncharges,
                )
                for x in products
            ]
            for products in self.mols_products
        ]
        b2r2_products_sum = np.array([sum(x) for x in b2r2_products])

        b2r2_diff = b2r2_products_sum - b2r2_reactants_sum

        return b2r2_diff

    def get_b2r2_a(self, Rcut=3.5, gridspace=0.03):
        elements = self.unique_ncharges
        b2r2_reactants = [
            [
                get_b2r2_a_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for x in reactants
            ]
            for reactants in self.mols_reactants
        ]
        # first index is reactants
        b2r2_reactants_sum = np.array([sum(x) for x in b2r2_reactants])

        b2r2_products = [
            [
                get_b2r2_a_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for x in products
            ]
            for products in self.mols_products
        ]
        b2r2_products_sum = np.array([sum(x) for x in b2r2_products])

        b2r2_diff = b2r2_products_sum - b2r2_reactants_sum

        return b2r2_diff

    def get_b2r2_n(self, Rcut=3.5):
        b2r2_reactants = [
            [
                get_b2r2_n_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    elements=self.unique_ncharges,
                )
                for x in reactants
            ]
            for reactants in self.mols_reactants
        ]
        # first index is reactants
        b2r2_reactants_sum = np.array([sum(x) for x in b2r2_reactants])

        b2r2_products = [
            [
                get_b2r2_n_molecular(
                    x.nuclear_charges,
                    x.coordinates,
                    Rcut=Rcut,
                    elements=self.unique_ncharges,
                )
                for x in products
            ]
            for products in self.mols_products
        ]
        b2r2_products_sum = np.array([sum(x) for x in b2r2_products])

        return np.concatenate((b2r2_reactants_sum, b2r2_products_sum), axis=1)

class QML:
    def __init__(self):
        self.ncharges = []
        self.unique_ncharges = []
        self.max_natoms = 0
        self.atomtype_dict = {"H": 0, "C": 0, "N": 0, "O": 0, "S": 0, "Cl":0,
                                "F":0}
        self.mols_products = []
        self.mols_reactants = [[]]
        return

    def get_proparg_data(self):
        data = pd.read_csv("data/proparg/data.csv", index_col=0)
        reactants_files = [
            "data/proparg/data_react_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        products_files = [
            "data/proparg/data_prod_xyz/"
            + data.mol.values[i]
            + data.enan.values[i]
            + ".xyz"
            for i in range(len(data))
        ]
        all_mols = [qml.Compound(x) for x in reactants_files + products_files]
        self.barriers = data.dErxn.to_numpy()
        self.ncharges = [mol.nuclear_charges for mol in all_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        self.mols_reactants = [[qml.Compound(x)] for x in reactants_files]
        self.mols_products = [[qml.Compound(x)] for x in products_files]

        return

    def get_hydroform_data(self):
        co_df = pd.read_csv("data/hydroform/Co_clean.csv")
        names = co_df["name"].to_list()
        labels = [name[3:] for name in names]
        co_reactants = [
            "data/hydroform/geometries/co/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        co_products = [
            "data/hydroform/geometries/co/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.co_barriers = co_df["f_barr"].to_numpy()
        self.mols_reactants_co = [[qml.Compound(x)] for x in co_reactants]
        self.mols_products_co = [[qml.Compound(x)] for x in co_products]

        ir_df = pd.read_csv("data/hydroform/Ir_clean.csv")
        names = ir_df["name"].to_list()
        labels = [name[3:] for name in names]
        ir_reactants = [
            "data/hydroform/geometries/ir/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        ir_products = [
            "data/hydroform/geometries/ir/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.ir_barriers = ir_df["f_barr"].to_numpy()
        self.mols_reactants_ir = [[qml.Compound(x)] for x in ir_reactants]
        self.mols_products_ir = [[qml.Compound(x)] for x in ir_products]

        rh_df = pd.read_csv("data/hydroform/Rh_clean.csv")
        names = rh_df["name"].to_list()
        labels = [name[3:] for name in names]
        rh_reactants = [
            "data/hydroform/geometries/rh/r/" + label + "_reactant.xyz"
            for label in labels
        ]
        rh_products = [
            "data/hydroform/geometries/rh/p/" + label + "_product.xyz"
            for label in labels
        ]
        self.rh_barriers = rh_df["f_barr"].to_numpy()
        self.mols_reactants_rh = [[qml.Compound(x)] for x in rh_reactants]
        self.mols_products_rh = [[qml.Compound(x)] for x in rh_products]

        self.barriers = np.concatenate(
            (self.co_barriers, self.ir_barriers, self.rh_barriers), axis=0
        )

        all_reactants = co_reactants + ir_reactants + rh_reactants
        all_products = co_products + ir_products + rh_products
        list_reactants = [qml.Compound(x) for x in all_reactants]

        self.mols_reactants = [[qml.Compound(x)] for x in all_reactants]
        self.mols_products = [[qml.Compound(x)] for x in all_products]

        ncharges = [mol.nuclear_charges for mol in list_reactants]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))

        return

    def get_GDB7_ccsd_data(self):
        df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
        self.barriers = df['dE0'].to_numpy()
        indices = df['idx'].apply(pad_indices).to_list()

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


    def get_cyclo_data(self):
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        self.barriers = df['G_act'].to_numpy()
        indices = df['rxn_id'].to_list()
        self.indices = indices
        rxns = ["data/cyclo/xyz/"+str(i) for i in indices]
        
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

    def get_cyclo_xtb_data(self):
        # test set at lower quality geometry
        df = pd.read_csv("data/cyclo/full_dataset.csv", index_col=0)
        barriers = df['G_act'].to_numpy()

        filedir = 'data/cyclo/xyz-xtb/'
        files = glob(filedir+"*.xyz")
        indices = np.unique([x.split("/")[-1].split("_")[1].strip('.xyz') for x in files])
        self.indices = [int(x) for x in indices]
        reactants_files = []
        products_files = []

        r_mols = []
        p_mols = []
        barriers = []
        for i, idx in enumerate(indices):
            idx = str(idx)
            # multiple r files
            if os.path.exists(filedir+"Reactant_"+idx+".xyz"):
                rfile = filedir+"Reactant_"+idx+".xyz"
                r_atomtypes, r_ncharges, r_coords = reader(rfile)
                r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
                sub_rmols.append(r_mol)
            elif os.path.exists(filedir+"Reactant_"+idx+"_0.xyz"):
                rfiles = glob(filedir+'Reactant_'+idx+'_*.xyz')
                sub_rmols = []
                for rfile in rfiles:
                    r_atomtypes, r_ncharges, r_coords = reader(rfile)
                    r_mol = create_mol_obj(r_atomtypes, r_ncharges, r_coords)
                    sub_rmols.append(r_mol)
            else:
                print("cannot find rfile")
                sub_rmols.append([None])

            sub_pmols = []
            pfile = filedir+"Product_"+idx+".xyz"
            p_atomtypes, p_ncharges, p_coords = reader(pfile)
            p_mol = create_mol_obj(p_atomtypes, p_ncharges, p_coords)
            sub_pmols.append(p_mol)

            if None not in sub_pmols and None not in sub_rmols: 
                r_mols.append(sub_rmols)
                p_mols.append(sub_pmols)
                barrier = df[df['rxn_id'] == int(idx)]['G_act'].item()
                barriers.append(barrier)
            else:
                print("skipping r mols", sub_rmols, 'and p mols', sub_pmols, 'for idx', idx) 

        assert len(r_mols) == len(p_mols)
        assert len(r_mols) == len(barriers)
        self.mols_reactants = r_mols
        self.mols_products = p_mols
        self.barriers = barriers
        all_r_mols = np.concatenate(r_mols)
        self.ncharges = [x.nuclear_charges for x in all_r_mols]
        self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
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
        slatm_products = np.array([sum(x) for x in slatm_products])
        slatm_diff = slatm_products - slatm_reactants_sum

        return slatm_diff

