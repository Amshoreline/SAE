import rdkit
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.DataStructs import BulkTanimotoSimilarity, BulkDiceSimilarity, BulkCosineSimilarity, BulkTverskySimilarity, BulkSokalSimilarity
import numpy as np


def get_MACCS(smi):  # 167
    mol = Chem.MolFromSmiles(smi)
    return MACCSkeys.GenMACCSKeys(mol)


def get_RDKFP(smi):  # 2048
    mol = Chem.MolFromSmiles(smi)
    return Chem.RDKFingerprint(mol)


# ECFP4
def get_Morgan(smi):  # 2048
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)


def get_Avalon(smi):  # 512
    mol = Chem.MolFromSmiles(smi)
    return pyAvalonTools.GetAvalonFP(mol)


def get_Pairs(smi):
    mol = Chem.MolFromSmiles(smi)
    return Pairs.GetAtomPairFingerprintAsBitVect(mol)


def get_Torsions(smi):
    mol = Chem.MolFromSmiles(smi)
    return Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)  # Int


name2sim_func = {
    'tani': BulkTanimotoSimilarity,
    'dice': BulkDiceSimilarity,
    'cosine': BulkCosineSimilarity,
    'sokal': BulkSokalSimilarity,
}

name2fp_func = {
    'maccs': get_MACCS,
    'rdkfp': get_RDKFP,
    'morgan': get_Morgan,
    'avalon': get_Avalon,
    'pairs': get_Pairs,
    'torsions': get_Torsions,
}


def get_sim(smis_0, smis_1, sim_name='tani', fp_name='morgan'):
    sim_func = name2sim_func[sim_name]
    fp_func = name2fp_func[fp_name]
    #
    fps_0 = [fp_func(smi) for smi in smis_0]
    fps_1 = [fp_func(smi) for smi in smis_1]
    sims = [sim_func(fp_0, fps_1) for fp_0 in fps_0]
    sims = np.array(sims)
    return sims


if __name__ == '__main__':
    print(get_sim(['CCC', 'CCCCC'], ['COC',' CCOC', 'OCCO']))
