import numpy as np
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

path_to_checkpoint = 'molbert_100epochs/checkpoints/last.ckpt'
f = MolBertFeaturizer(path_to_checkpoint)
features, masks = f.transform(np.asarray(['C']))
assert all(masks)
print(features)
