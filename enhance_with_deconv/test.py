import numpy as np
from spot_enhance import Detector, Spot
from digitise_absorption import digitise_path, Point, Pixel
import pytest


@pytest.mark.parametrize("start, end, expected", [(Point(0.5, 0.5), Point(3.4, 0.4),
                                                   [Pixel(0, 0), Pixel(1, 0), Pixel(2, 0), Pixel(3, 0)])])
def test_digitise(start, end, expected):
    path, _ = digitise_path(start, end, np.array([1, 1, 1]))
    assert path == expected


def test_psf():
    from data import detector_data, spot_data
    pilatus_det = Detector(pixel_size=detector_data.pixel_size,
                           depth=detector_data.depth,
                           mu=detector_data.mu,
                           normal=detector_data.normal)

    spot = Spot(intensity_map=spot_data.intensity_map,
                s1=spot_data.s1_vector,
                detector=pilatus_det)

    psf = spot.psf

    expected_psf = np.load('data/expected_psf.npy')

    assert np.array_equal(psf, expected_psf)
