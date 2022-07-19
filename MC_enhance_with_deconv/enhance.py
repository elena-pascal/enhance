import numpy as np
import numpy.ma as ma
from collections import Counter
from scipy.spatial.transform import Rotation
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from refl_loader import Shoebox, load


def get_refl(refl_file) -> Tuple[List[Shoebox], List[np.array], List[np.array]]:
    """
    return useful data from a reflection table

    Params
    ------
    shoeboxes = list
        list of refl_loader Shoeboxes

    Return
    -----
    shoeboxes : List[Shoebox]
        List of spot shoeboxes
    bboxes : List[bboxes]
        List of bboxes
    s1_vectors : List[s1]
        List of s1 vectors
    """
    data = load(refl_file)
    return data['shoebox'], data['bbox'], data['s1']


def lab_to_detector(vectors: np.ndarray) -> np.ndarray:
    """
    From lab to detector there is a pi rotation around x
    """
    r = Rotation.from_rotvec(np.pi * np.array([1, 0, 0]))
    return r.apply(vectors)


class Detector:
    def __init__(self, px_x, px_y, depth, mu):
        self.px_x = px_x
        self.px_y = px_y
        self.depth = depth
        self.s = 1 / mu

    def map_mm_to_px(self, position_array) -> list:
        """
        Map 3D position vectors to detector pixels

        :param
            position_array : array
            This should be raveled
        :return
            pixels : list
        """
        if position_array.shape[1] == 3:
            position_array = position_array.T

        index_x = ((position_array[0]) // self.px_x).astype(int)
        index_y = ((position_array[1]) // self.px_y).astype(int)

        pixels = list(zip(index_x, index_y))
        return pixels


class Spot:
    def __init__(self, intensity, s1, detector, pos_map=1, padding=4):
        self.pos_map = pos_map
        self.padding = padding
        self.detector = detector

        self.num_pix = len(intensity)
        self.image = self.pad(intensity)
        self.num_pad_pix = len(self.image)

        self.px_pos = self.transform(self.position_grid)
        self.intensity = self.transform(self.pad(intensity))

        self.s1 = lab_to_detector(self.transform(self.pad(s1)))
        self.non_zero = ma.masked_values(self.intensity, 0)

    @property
    def position_grid(self) -> np.ndarray:
        """
        Make a position grid for s1 origins.
        The distances are in the padded spot frame.
        :return:
        px_pos
        """
        x = self.detector.px_x
        y = self.detector.px_x
        px_pos_x, px_pos_y = np.meshgrid(np.arange(x / (2 * self.pos_map),
                                                   x * self.num_pad_pix,
                                                   x / self.pos_map),
                                         np.arange(y / (2 * self.pos_map),
                                                   y * self.num_pad_pix,
                                                   y / self.pos_map))
        px_pos_z = np.zeros((self.num_pad_pix * self.pos_map,
                             self.num_pad_pix * self.pos_map))

        # numpy arrays have axis y before x: v[y,x]
        px_pos = np.stack((px_pos_y, px_pos_x, px_pos_z)).T
        return px_pos

    def transform(self, array: np.ndarray) -> np.ndarray:
        """
        Do the following transformations to the input parameter arrays:
        1. Match grids to the choice of positions per pixel
        2. Ravel out the arrays. Since they should all have the same shape and
        sampling on the detector, ravelling the 2D to 1D and tracking position
        with the pos array will not loose any info

        :param
        array:
            array to be transformed
        :return:
        array
            transformed array, will have different shape and size
        """

        matched = self.match_to_px_grid(array)
        return self.ravel(matched)

    def pad(self, array: np.ndarray) -> np.ndarray:
        p = self.padding
        if len(array.shape) == 2:
            return np.pad(array, ((p, p), (p, p)))
        elif len(array.shape) == 3:
            return np.pad(array, ((p, p), (p, p), (0, 0)))
        else:
            raise ValueError('Array has unexpected shape in pad')

    @staticmethod
    def ravel(array: np.ndarray) -> np.ndarray:
        """
        Ravel array such that dimension one is along all the valid pixels
        """
        if len(array.shape) == 2:
            return array.ravel()
        elif len(array.shape) == 3:
            return array.reshape(-1, array.shape[-1])

    def match_to_px_grid(self, array: np.ndarray) -> np.ndarray:
        """
        If more than one position per pixel is considered
        match the parameter arrays to the position array

        :param
        array:
            array to be repeated

        :return:
            repeated array
        """
        return np.repeat(np.repeat(array, self.pos_map, axis=0),
                         self.pos_map, axis=1)


class MCProjection:
    """
    The stochastic X-ray absorption of X-rays in pixelated detectors
    introduces both parallax and diffuseness. The measured counts in each
    pixel contains contributions from X-ray entering the detector through
    neighbouring pixels. The aim of this class is to use MC to predict these
    effects.

    Attributes
    ----------
    spot : Spot
        spot info contained in a shoebox
    detector : Detector
        info about the detector
    num_mc : int
        number of MC runs needed.
    prob_map: np.array
         the probability map on the back of the detector

    Methods
    -------
    plot_spot
        Plots of spot used
    plot_prob_map
        Plots the probability map calculated
    plot_s1_in_det_frame
        Plots the s1 vector in detector frame

    Example
    -------
    # pixel depth
    d = 1  # mm

    # pixel size
    p_x = 0.172  # mm
    p_y = 0.172  # mm

    # absorption factor
    abs_f = 0.6245  # mm^-1

    pilatus = Detector(p_x, p_y, d, abs_f)

    # these lines should be replaced with dials
    intensity_all = np.load('data/pixel_array.npy')
    intensity_first, _, _ = intensity_all.astype(np.int8)

    s1_all = np.load('data/s1_vectors.npy')
    s1_first = s1_all[:49].reshape(7, 7, 3)

    first_spot = Spot(intensity_first, s1_first, pilatus)

    projection = MCProjection(first_spot, pilatus, 100)

    print(projection.prob_map)
    projection.plot_spot()
    projection.plot_prob_map()
    projection.plot_s1_in_det_frame()
    """
    def __init__(self, spot, detector, num_mc=10000):
        self.spot = spot
        self.detector = detector
        self.num_mc = num_mc

        # square spot for now
        self._num_pix = self.spot.num_pix
        self._num_pad_pix = self.spot.num_pad_pix

        # only do the work for non zero intensity pixels
        self._nonzero_int = self._apply_nonzero_mask(self.spot.intensity)
        self._nonzero_s1 = self._apply_nonzero_mask(self.spot.s1)
        self._nonzero_pos = self._apply_nonzero_mask(self.spot.px_pos)

        self.prob_map = self.get_prob_map()

    @property
    def path_travelled(self) -> np.ndarray:
        summed_int = np.sum(self._nonzero_int)

        rnd = np.random.rand(summed_int * self.num_mc)
        path_l = -self.detector.s * np.log(rnd)
        return path_l

    def _apply_nonzero_mask(self, array) -> np.ndarray:
        return array[~self.spot.non_zero.mask]

    def _repeat_per_trial(self, array: np.ndarray) -> np.ndarray:
        """
        Repeat array entry per intensity value
        and per number of MC runs
        :param array:
        :return: trials_array:
        """
        per_hit = np.repeat(array, self._nonzero_int, axis=0)
        per_mc = np.repeat(per_hit, self.num_mc, axis=0)
        return per_mc

    @property
    def last_position(self) -> np.ndarray:
        s1 = self._repeat_per_trial(self._nonzero_s1)
        final_pos = s1.T * self.path_travelled
        pos = self._repeat_per_trial(self._nonzero_pos)
        return pos + final_pos.T

    def _inside_detector_region(self, coords) -> np.ma:
        """
        Mask for only photons stopping withing the detector depth
        and chosen padded spot region
        :return: np.ma
        """

        last_x, last_y, last_z = coords.T

        # ignore x-rays that escape through the back of detector
        escape_z_plus = ma.masked_greater(last_z, self.detector.depth)

        # ignore x-rays that travel outside padded region
        escape_x_plus = ma.masked_greater(last_x, self.detector.px_x * self._num_pad_pix)
        escape_x_minus = ma.masked_less_equal(last_x, 0)
        escape_y_plus = ma.masked_greater(last_y, self.detector.px_y * self._num_pad_pix)
        escape_y_minus = ma.masked_less_equal(last_y, 0)

        # mask all escaped
        escaping_mask = escape_x_minus.mask | escape_x_plus.mask | \
            escape_y_minus.mask | escape_y_plus.mask | \
            escape_z_plus.mask

        return escaping_mask

    def _coords_to_counts(self, coords) -> np.ndarray:
        """
        Translate coordinates in mm to counts per pixel
        :return:
        prob_map
        """
        pixels = self.detector.map_mm_to_px(coords)
        counts_on_back = np.zeros((self._num_pad_pix, self._num_pad_pix))

        # count hits per pixel
        c = Counter(pixels)
        key_x, key_y = zip(*c.keys())

        # populate count array with counts
        counts_on_back[[key_x], [key_y]] = list(c.values())
        return counts_on_back

    def get_prob_map(self) -> np.ndarray:
        last_coords = self.last_position
        valid_last_coords = last_coords[~self._inside_detector_region(last_coords)]
        counts = self._coords_to_counts(valid_last_coords)

        prob_map = counts/(self.num_mc * self.spot.pos_map**2)
        return prob_map.T  # store as x,y instead of y,x

    def _plot_view_in_det_frame(self, intensity_map, title, zero=np.nan, show_pad=True, static=False):
        # set zero values to required value or non-value
        intensity_map = np.where(intensity_map == 0, zero, intensity_map)
        fig = px.imshow(intensity_map, title=title)

        if show_pad:
            pad_edge = self.spot.padding - 0.5
            size = self._num_pix
            fig.add_scatter(x=[pad_edge, pad_edge + size, pad_edge + size, pad_edge, pad_edge],
                            y=[pad_edge, pad_edge, pad_edge + size, pad_edge + size, pad_edge],
                            mode='lines')

        if static:
            fig.show("png")
        else:
            fig.show()

    def plot_spot(self, **kwargs):
        self._plot_view_in_det_frame(intensity_map=self.spot.image,
                                     title='Spot on front of detector',
                                     **kwargs)

    def plot_prob_map(self, **kwargs):
        self._plot_view_in_det_frame(intensity_map=self.prob_map,
                                     title='Spot on back of detector',
                                     **kwargs)

    def plot_s1_in_det_frame(self, scale=2):
        vector = np.array([[self._nonzero_s1[0, 0]], [self._nonzero_s1[0, 1]]])
        origin = np.array([[0], [0]])  # origin point
        plt.quiver(*origin, vector[0], -vector[1], scale=scale)
        plt.gca().invert_yaxis()
        plt.title('s1 vector in detector frame')
        plt.show()


class PSF:
    """
    Will name `point spread function` the effect of detector thickness
    as described in MCProjection. This is modelled as the probability map
    on the back of the detector generated by signal hitting only one pixel.

    Using only one s1 position per pixel results in an under-sampled psf
    (depending on the pixel size). In order to avoid this, grid s1 positions
    over one pixel and look for psf convergence.

    Attributes
    ----------
    detector : Detector
        info about the detector
    s1 : np.array
        s1 vector
    psf: np.array
    """
    def __init__(self, detector, s1, tau=0.1):
        self.detector = detector
        self.s1 = s1
        self.tau = tau

    @property
    def distribution(self):
        one_px = Spot()
        px_proj = MCProjection(one_px, self.detector)
        return px_proj.prob_map
