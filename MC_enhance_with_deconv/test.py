import numpy as np
from enhance import Detector, Spot, MCProjection


def test_mc_result():
    # pixel depth
    depth = 1  # mm

    # pixel size
    px_x = 0.172  # mm
    px_y = 0.172  # mm

    # absorption factor
    mu = 0.6245  # mm^-1

    detector = Detector(px_x, px_y, depth, mu)

    intensity_all = np.load('data/pixel_array.npy')
    intensity_first, _, _ = intensity_all.astype(np.int8)

    s1_all = np.load('data/s1_vectors.npy')
    s1_first = s1_all[:49].reshape(7, 7, 3)

    spot = Spot(intensity_first, s1_first, detector)

    np.random.seed(0)
    projection = MCProjection(spot, detector, 100)

    computed_projection = np.load('data/deterministic_prob_map.npy')

    assert np.array_equal(projection.prob_map, computed_projection)
