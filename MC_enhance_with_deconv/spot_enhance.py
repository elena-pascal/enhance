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
import logging
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from skimage import restoration
from typing import List
from refl_loader import Shoebox
from refl_loader import load as refl_load

import dials
from dials.util.options import ArgumentParser, flatten_experiments
from libtbx.phil import parse, scope
from digitise_absorbtion import get_abs_per_pixel

# Define a logger.
logger = logging.getLogger("dials.command_line.spot_enhance")

phil_scope = parse(
    """
    output {
        log = dials.spot_enhance.log
           .type = path
    }
    """
)


class Detector:
    def __init__(self, dials_detector):
        self.pixel_size = dials_detector.get_pixel_size()
        self.depth = dials_detector.get_thickness()
        self.mu = dials_detector.get_mu()
        # detector normal in lab frame
        normal = dials_detector.get_normal()
        self.normal_in_lab_frame = normal/np.linalg.norm(normal)

        # we will renormalise some computations in pixel voxel units
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

    def compute_psf(self, shoebox: Shoebox, s1_lab: np.array, pad: int):
        """
        Compute the effect of the detector thickness on a delta like
        beam hitting the center of a pixel.

        The pixel is positioned in the middle of the shoebox.

        :param shoebox: refl_loader.Shoebox
        :param s1_lab: array of size 3
            diffraction vector in lab frame
        :param pad: int
            data is padded such that pad_data_pad
        """
        shoebox_size = max(shoebox.data.shape[:2]) * 2 + 1

        s1_detector = self.lab_to_detector(s1_lab)
        pixels, abs_per_pixel = get_abs_per_pixel(s1=s1_detector,
                                                  px_norm=self.px_norm,
                                                  spot_size=shoebox_size,
                                                  mu=self.mu)

        psf = np.zeros((shoebox_size, shoebox_size))

        fast = [pixel.x for pixel in pixels]
        slow = [pixel.y for pixel in pixels]

        psf[fast, slow] = abs_per_pixel
        psf = psf/np.sum(psf)
        Spot.display(psf, 'psf')
        return psf


class Spot:
    def __init__(self, shoebox: Shoebox, s1: np.array, detector: Detector):
        self.shoebox = shoebox
        self.detector = detector
        self.s1 = self._normalise_s1(s1)  # in pixel units

    def _normalise_s1(self, unnormed_s1: np.array) -> np.array:
        """
        For some reasons s1 is saved in indexed.refl non-normalised
        """
        normed_s1 = unnormed_s1/np.linalg.norm(unnormed_s1)
        return normed_s1/self.detector.px_norm

    @staticmethod
    def display(spot, title=None, ax=None, pad=None):
        size = spot.shape
        if ax is None:
            _fig, ax = plt.subplots()
        ax.imshow(spot,
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
        psf = self.detector.compute_psf(self.shoebox, self.s1, pad)

        # use first z=0 spot
        padded = np.pad(self.shoebox.data[0], ((pad, pad), (pad, pad)), 'constant', constant_values=0)
        recovered_spot = restoration.richardson_lucy(padded,
                                                     psf,
                                                     num_iter=500,
                                                     clip=False,
                                                     filter_epsilon=0.00)
        return recovered_spot


# make it dials compatible
@dials.util.show_mail_handle_errors()
def run(args: List[str] = None, phil: scope = phil_scope) -> None:
    usage = "$ dials.spot_enhance indexed.expt indexed.refl [options]"

    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        read_experiments=True,
        read_reflections=True,
        check_format=True,
        epilog=__doc__
    )

    parameters, options = parser.parse_args(args=args, show_diff_phil=False)

    experiments = flatten_experiments(parameters.input.experiments)

    # define the detector
    expt = experiments[0]
    detector = Detector(expt.detector[0])

    # is there a better way to get the refl filename from params?
    # probably since this must be the absolute strangest way
    refl_filename = parameters.input.reflections[0][0]
    data = refl_load(refl_filename)
    shoeboxes, s1 = data['shoebox'], data['s1']

    # pick a spot
    spot_number = 588

    spot = Spot(shoeboxes[spot_number], s1[spot_number], detector)
    pad = 4
    # enhance it
    enhanced = spot.enhance(pad=pad)

    # plot to check
    _fig, (ax1, ax2) = plt.subplots(1, 2)
    padded = np.pad(shoeboxes[spot_number].data[0], ((pad, pad), (pad, pad)))
    Spot.display(padded, title='as measured', ax=ax1, pad=pad)
    Spot.display(enhanced, title='enhanced', ax=ax2, pad=pad)
    plt.show()


if __name__ == "__main__":
    run()
