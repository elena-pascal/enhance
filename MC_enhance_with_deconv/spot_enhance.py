"""
Remove parallax effects introduced by the detector thickness.
This is achieved by deconvolving the point spread function of the
detector. This in turn, is just the digitised absorption of a beam of
unit intensity hitting the center of a pixel and travelling along s1
through the detector.

Usage:
    > spot_enhance.py indexed.expt indexed.refl

Because this requires s1 vectors and information about detector geometry,
this state of the code assumes the indexed data is available.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from skimage import restoration

from digitise_absorbtion import get_abs_per_pixel


class Detector:
    def __init__(self, pixel_size, depth, mu, normal):
        self.pixel_size = pixel_size
        self.depth = depth
        self.mu = mu
        # detector normal in lab frame normalised
        self.normal_in_lab_frame = normal/np.linalg.norm(normal)

        # we will normalise computations in pixel voxel units
        self.px_norm = np.array([*self.pixel_size, self.depth])

    def lab_to_detector(self, vector: np.ndarray) -> np.ndarray:
        """
        For a given vector in lab frame, return its coordinates in detector frame.
        We can do this because we know the detector normal in lab frame.
        Note, the detector frame z is conveniently the detector normal.

        The rotation that rotates lab z axis to detector z in lab frame:
         z_det_in_det_frame = R * z_det_in_lab_frame.
         R = z_det_in_det_frame * z_det_in_lab_frame.T
         R = [0, 0, 1] * self.normal_in_lab_frame.T
        """
        r_lab_to_det = R.from_matrix(np.outer(np.array([0, 0, 1]), self.normal_in_lab_frame))
        return r_lab_to_det.apply(vector)

    def compute_psf(self, intensity_map: np.array, s1_lab: np.array, pad: int):
        """
        Compute the effect of the detector thickness on a delta like
        beam hitting the center of a pixel.

        The pixel is positioned in the middle of the intensity_map.

        :param intensity_map: refl_loader.intensity_map
        :param s1_lab: array of size 3
            diffraction vector in lab frame
        :param pad: int
            data is padded such that pad_data_pad
        """
        intensity_map_size = max(intensity_map.shape[:2]) * 2 + 1

        s1_detector = self.lab_to_detector(s1_lab)
        pixels, abs_per_pixel = get_abs_per_pixel(s1=s1_detector,
                                                  px_norm=self.px_norm,
                                                  spot_size=intensity_map_size,
                                                  mu=self.mu)

        psf = np.zeros((intensity_map_size, intensity_map_size))

        fast = [pixel.x for pixel in pixels]
        slow = [pixel.y for pixel in pixels]

        psf[slow, fast] = abs_per_pixel
        psf = psf/np.sum(psf)
        Spot.display(psf, 'psf')
        return psf


class Spot:
    def __init__(self, intensity_map: np.array, s1: np.array, detector: Detector):
        self.intensity_map = intensity_map
        self.detector = detector
        self.s1 = self._normalise_s1(s1)  # in pixel units

    def _normalise_s1(self, unnormed_s1: np.array) -> np.array:
        """
        For some reasons s1 is saved in indexed.refl non-normalised
        """
        normed_s1 = unnormed_s1/np.linalg.norm(unnormed_s1)
        return normed_s1/self.detector.px_norm

    @staticmethod
    def display(spot_map, title=None, ax=None, pad=None):
        size = spot_map.shape
        if ax is None:
            _fig, ax = plt.subplots()
        ax.imshow(spot_map,
                  extent=(0, size[0], size[1], 0),
                  origin='upper',
                  cmap='viridis')
        if title:
            ax.set_title(title)
        ax.axis('off')

        if pad is not None:
            # add bbox
            ax.plot([pad, size[0]-pad, size[0]-pad, pad, pad],
                    [pad, pad, size[1]-pad, size[1]-pad, pad])
        return ax

    def enhance(self, pad=2):
        """
        compute the detector psf for this spot and then
        deconvolve it from the measured spot to get
        an "enhanced" spot.
        """
        psf = self.detector.compute_psf(self.intensity_map, self.s1, pad)

        padded_spot = np.pad(self.intensity_map, ((pad, pad), (pad, pad)),)
        recovered_spot = restoration.richardson_lucy(padded_spot,
                                                     psf,
                                                     num_iter=500,
                                                     clip=False,
                                                     filter_epsilon=0.00)
        return recovered_spot


if __name__ == "__main__":
    # make detector instance
    import detector_data
    pilatus_det = Detector(pixel_size=detector_data.pixel_size,
                           depth=detector_data.depth,
                           mu=detector_data.mu,
                           normal=detector_data.normal)

    # make a spot instance
    import spot_data
    spot = Spot(intensity_map=spot_data.intensity_map,
                s1=spot_data.s1_vector,
                detector=pilatus_det)

    # enhance it
    pad_size = 2
    enhanced = spot.enhance(pad=pad_size)

    # plot for visual inspection
    _fig, (ax1, ax2) = plt.subplots(1, 2)
    padded = np.pad(spot_data.intensity_map, ((pad_size, pad_size), (pad_size, pad_size)))
    Spot.display(padded, title='as measured', ax=ax1, pad=pad_size)
    Spot.display(enhanced, title='enhanced', ax=ax2, pad=pad_size)
    plt.show()

