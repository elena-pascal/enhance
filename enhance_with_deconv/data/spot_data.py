# Graeme's data spot example
# normally this info would be read programmatically from a dials file

import numpy as np
intensity_map = np.load('data/pixel_array.npy')[0]
s1_vector = np.mean(np.load('data/s1_vectors.npy'), axis=0)
