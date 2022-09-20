import numpy as np
from spot_enhance import Detector, Spot
import pytest


def test_psf():
    import detector_data
    pilatus_det = Detector(pixel_size=detector_data.pixel_size,
                           depth=detector_data.depth,
                           mu=detector_data.mu,
                           normal=detector_data.normal)

    import spot_data
    spot = Spot(intensity_map=spot_data.intensity_map,
                s1=spot_data.s1_vector,
                detector=pilatus_det)

    psf = spot.psf

    expected_psf = np.load('data/expected_psf.npy')

    assert np.array_equal(psf, expected_psf)
