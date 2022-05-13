import numpy as np
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import sys

path_to_checkpoint = 'molbert_100epochs/checkpoints/last.ckpt'
smiles_file = sys.argv[1]
output_file = sys.argv[2]

# Pooled embedding is the default, but we make it explicit here.
# Note, the MolBERT paper uses this pooling for generating embeddings
# for use in downstream tasks.
featurizer = MolBertFeaturizer(path_to_checkpoint, embedding_type='pooled')

with open(smiles_file, 'r') as f:
    smiles_strings = [x.strip() for x in f.readlines() if x.strip()]

features, masks = featurizer.transform(smiles_strings)
assert all(masks)

np.savetxt(output_file, features, delimiter=',')
