from glob import glob
import numpy as np
import pandas as pd
import ase.io
import qml
from src.fingerprints import get_MFP, get_DRFP
from src.b2r2 import get_b2r2_a, get_b2r2_l, get_b2r2_n
import tqdm


def read_mol(xyz, input_bohr=False):
    asemol = ase.io.read(xyz)
    mol = qml.Compound()
    mol.atomtypes = asemol.get_chemical_symbols()
    mol.nuclear_charges = asemol.numbers
    mol.coordinates = asemol.positions * 0.529177 if input_bohr else asemol.positions
    return mol


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
        self.mols_products = []
        self.mols_reactants = [[]]

        self.init_gdb_loaders()
        self.init_cyclo_loaders()
        self.init_proparg_loaders()


    def init_gdb_loaders(self):

        def get_gdb_xyz_files(idx):
            r = [f'data/gdb7-22-ts/xyz/{idx}/r{idx}.xyz']
            p = sorted(glob(f'data/gdb7-22-ts/xyz/{idx}/p{idx}*.xyz'))
            return r, p

        def get_gdb_xtb_xyz_files(idx):
            r = [f'data/gdb7-22-ts/xyz-xtb/{idx}/Reactant_{idx}_0_opt.xyz']
            p = sorted(glob(f'data/gdb7-22-ts/xyz-xtb/{idx}/Product_{idx}_?_opt.xyz'))
            return r, p

        pad_indices = lambda idx: f'{idx:06}'

        self.get_GDB7_xtb_data = self.get_data_template(csv_path='data/gdb7-22-ts/ccsdtf12_dz.csv',
                                                        csv_column_target='dE0',
                                                        input_bohr=False,
                                                        bad_idx_path='data/gdb7-22-ts/xtb_bad_idx.dat',
                                                        get_xyz_files=get_gdb_xtb_xyz_files,
                                                        get_idx = lambda df: df['idx'])

        self.get_GDB7_ccsd_data = self.get_data_template(csv_path='data/gdb7-22-ts/ccsdtf12_dz.csv',
                                                         csv_column_target='dE0',
                                                         input_bohr=True,
                                                         get_xyz_files=get_gdb_xyz_files,
                                                         get_idx = lambda df: df['idx'].apply(pad_indices).to_list())

        self.get_GDB7_ccsd_subset_data = self.get_data_template(csv_path='data/gdb7-22-ts/ccsdtf12_dz.csv',
                                                                csv_column_target='dE0',
                                                                input_bohr=True,
                                                                bad_idx_path='data/gdb7-22-ts/xtb_bad_idx.dat',
                                                                get_xyz_files=get_gdb_xyz_files,
                                                                get_idx = lambda df: df['idx'].apply(pad_indices).to_list())

    def init_cyclo_loaders(self):

        def get_cyclo_xyz_files(idx):

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

            r = sorted(glob(f'data/cyclo/xyz/{idx}/r*.xyz'))
            r = check_alt_files(r)
            assert len(r)==2, f"Inconsistent length of {len(r)} for {idx}"
            p = sorted(glob(f'data/cyclo/xyz/{idx}/p*.xyz'))
            return r, p

        def get_cyclo_xtb_xyz_files(idx):
            r = sorted(glob(f"data/cyclo/xyz-xtb/Reactant_{idx}_*.xyz"))
            p = [f"data/cyclo/xyz-xtb/Product_{idx}.xyz"]
            return r, p

        self.get_cyclo_data = self.get_data_template(csv_path='data/cyclo/full_dataset.csv',
                                                     csv_column_target='G_act',
                                                     input_bohr=False,
                                                     get_xyz_files=get_cyclo_xyz_files,
                                                     get_idx = lambda df: df['rxn_id'].to_list())

        self.get_cyclo_xtb_data = self.get_data_template(csv_path='data/cyclo/full_dataset.csv',
                                                         csv_column_target='G_act',
                                                         input_bohr=False,
                                                         bad_idx_path='data/cyclo/xtb_bad_idx.dat',
                                                         get_xyz_files=get_cyclo_xtb_xyz_files,
                                                         get_idx = lambda df: df['rxn_id'].to_list())

        self.get_cyclo_subset_data = self.get_data_template(csv_path='data/cyclo/full_dataset.csv',
                                                            csv_column_target='G_act',
                                                            input_bohr=False,
                                                            bad_idx_path='data/cyclo/xtb_bad_idx.dat',
                                                            get_xyz_files=get_cyclo_xyz_files,
                                                            get_idx = lambda df: df['rxn_id'].to_list())

    def init_proparg_loaders(self):

        get_proparg_xyz_files     = lambda idx: ([f'data/proparg/xyz/{idx}.r.xyz'],     [f'data/proparg/xyz/{idx}.p.xyz'])
        get_proparg_xtb_xyz_files = lambda idx: ([f'data/proparg/xyz-xtb/{idx}.r.xyz'], [f'data/proparg/xyz-xtb/{idx}.p.xyz'])
        get_proparg_idx = lambda df: [''.join(x) for x in zip(df['mol'].to_list(), df['enan'].to_list())]

        self.get_proparg_data = self.get_data_template(csv_path='data/proparg/data.csv',
                                                       csv_column_target='Eafw',
                                                       get_xyz_files=get_proparg_xyz_files,
                                                       get_idx = get_proparg_idx)

        self.get_proparg_data_xtb = self.get_data_template(csv_path='data/proparg/data.csv',
                                                           csv_column_target='Eafw',
                                                           get_xyz_files=get_proparg_xtb_xyz_files,
                                                           get_idx = get_proparg_idx)


    def get_data_template(self,
                          csv_path,
                          csv_column_target,
                          bad_idx_path=None,
                          input_bohr=False,
                          get_xyz_files=None,
                          get_idx=None):

        def get_data(subset=None):
            df = pd.read_csv(csv_path, index_col=0)
            self.barriers = df[csv_column_target].to_numpy()
            indices = get_idx(df)

            bad_idx = np.loadtxt(bad_idx_path, dtype=int) if bad_idx_path else None

            r_mols = []
            p_mols = []
            good_indices = []
            for i, idx in tqdm.tqdm(enumerate(indices)):

                if (bad_idx is not None) and (idx in bad_idx):
                    continue

                def read_mols(rfiles):
                    sub_rmols = []
                    for f in rfiles:
                        mol = read_mol(f, input_bohr=input_bohr)
                        sub_rmols.append(mol)
                    return sub_rmols

                rfiles, pfiles = get_xyz_files(idx)
                r_mols.append(read_mols(rfiles))
                p_mols.append(read_mols(pfiles))

                good_indices.append(i)
                if subset and len(good_indices)==subset:
                    break

            self.barriers = self.barriers[good_indices]
            self.mols_reactants = r_mols
            self.mols_products  = p_mols
            self.ncharges = [x.nuclear_charges for x in np.concatenate(r_mols)]
            self.unique_ncharges = np.unique(np.concatenate(self.ncharges))
            return

        return get_data


    def get_SLATM(self):
        mbtypes = qml.representations.get_slatm_mbtypes(self.ncharges)
        slatm_reactants = np.array([sum(
                            qml.representations.generate_slatm(
                                x.coordinates,
                                x.nuclear_charges,
                                mbtypes,
                                local=False
                                ) for x in reactants) for reactants in tqdm.tqdm(self.mols_reactants)])
        slatm_products = np.array([sum(
                            qml.representations.generate_slatm(
                                x.coordinates,
                                x.nuclear_charges,
                                mbtypes,
                                local=False
                                ) for x in products) for products in tqdm.tqdm(self.mols_products)])
        slatm_diff = slatm_products - slatm_reactants
        return slatm_diff


    def get_b2r2(self, Rcut=3.5, gridspace=0.03, variant='l'):
        if variant=='l':
            return get_b2r2_l(self.mols_reactants, self.mols_products, self.unique_ncharges, Rcut=Rcut, gridspace=gridspace)
        if variant=='a':
            return get_b2r2_a(self.mols_reactants, self.mols_products, self.unique_ncharges, Rcut=Rcut, gridspace=gridspace)
        if variant=='n':
            return get_b2r2_n(self.mols_reactants, self.mols_products, self.unique_ncharges, Rcut=Rcut, gridspace=gridspace)
