import numpy as np
from typing import NamedTuple


class Point(NamedTuple):
    """
    a point in 2D with x, y coords
    """

    x: float
    y: float


class Pixel(NamedTuple):
    """
    a point in 2D with x, y coords
    """

    x: int
    y: int


def digitise_path(entry_pos, exit_pos, px_norm):
    """
    For a given entry and exit position, find the pixels
    traversed and the path in those pixels.

    A notebook exploring benefits and limitation of a number of
    different algorithms can be found at:
    https://github.com/DiamondLightSource/dials_research/blob/ssx_work/ssx_work/detector_effects/MC_notebook/line_digitisation.ipynb

    This implemented algorithm follows the ray equation:
            𝑝𝑜𝑖𝑛𝑡(𝑡)=𝑢+𝑡𝑣
    where u is the entry position and v is the direction of the ray.
    Thinking in units of t, we use the following parameters:
    t_delta_x, t_delta_y : pixel dimensions in t units
    t_max_x, t_max_y: path dimensions in pixel in t units
    step_x, step_y: pixel dimensions, for simplicity and speed
    pixels dimensions are unity.

    Because we use normalised units the last step in computing paths
    per pixel is to convert pixel units to mm units.
    """
    dx = abs(exit_pos.x - entry_pos.x)
    dy = abs(exit_pos.y - entry_pos.y)
    norm = np.sqrt(dx * dx + dy * dy)

    start_pixel = Pixel(int(entry_pos.x), int(entry_pos.y))
    end_pixel = Pixel(int(exit_pos.x), int(exit_pos.y))

    x, y = start_pixel.x, start_pixel.y

    # predict number of pixels traversed
    n = 0

    if dx == 0:
        step_x = 0
        t_delta_x = np.inf
        t_delta_y = dy
        t_max_x = np.inf

    elif dy == 0:
        step_y = 0
        t_delta_x = dx
        t_delta_y = np.inf
        t_max_y = np.inf

    else:
        # deltas are just the inverse component of d
        t_delta_x = norm/dx
        t_delta_y = norm/dy

    # set the travelling direction quadrant based on d
    if exit_pos.x < entry_pos.x:
        step_x = -1
        t_max_x = abs((entry_pos.x - np.floor(entry_pos.x)) * t_delta_x)
        n += x - end_pixel.x

    elif exit_pos.x > entry_pos.x:
        step_x = 1
        t_max_x = abs((np.ceil(entry_pos.x) - entry_pos.x) * t_delta_x)
        n += end_pixel.x - x

    if exit_pos.y < entry_pos.y:
        step_y = -1
        t_max_y = abs((entry_pos.y - np.floor(entry_pos.y)) * t_delta_y)
        n += y - end_pixel.y

    elif exit_pos.y > entry_pos.y:
        step_y = 1
        t_max_y = abs((np.ceil(entry_pos.y) - entry_pos.y) * t_delta_y)
        n += end_pixel.y - y

    if dx == dy == 0:
        # avoid 0/0
        v = [0, 0]
    else:
        v = [dx/norm, dy/norm]

    x, y = int(entry_pos.x), int(entry_pos.y)
    pixels = [start_pixel]

    path_lengths = []
    for _ in range(n):
        pl = t_max_x

        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            x += step_x
        elif t_max_x > t_max_y:
            pl = t_max_y
            t_max_y += t_delta_y
            y += step_y
        else:
            x += step_x
            y += step_y

        pixels.append(Pixel(x, y))
        path_lengths.append(pl)

    # for final pixel append total path length in t units
    path_lengths.append(norm)

    # change v units from px to mm
    v = v * px_norm[:2]
    px_path_lens = np.linalg.norm(np.multiply.outer(path_lengths, v), axis=1)
    return pixels, px_path_lens


def get_exit_pos(entry_pos, s1):
    return entry_pos + s1[:2] / abs(s1[2])


def pathlength_from_proj(px_path_lens_proj, px_norm, s1):
    """
    From path length projection on detector surface
    return the full path length inside detector
    """
    # angle between s1 and detector normal
    s_dot_n = s1.dot(np.array([0, 0, -1]))

    if s_dot_n == 1:
        # avoid undefined division
        return px_norm[2]  # D
    else:
        return px_path_lens_proj / np.sqrt(1 - s_dot_n * s_dot_n)


def get_abs_per_pixel(s1, px_norm, spot_size, mu):
    """
    Compute intensity absorption per pixel (ie digitise)
    Using the path per pixel compute then the absorption per pixel
    from the Beer-Lambert law
    I_l = I_0 exp(-l * mu)
    where s is the mean free path
    and l is the path travelled

    Params
    -------
    s1: s1 vector in detector frame
    spot_size: side edge of spot
    """
    entry_pos = np.array([spot_size//2+0.5, spot_size//2+0.5])

    exit_pos = get_exit_pos(entry_pos, s1)

    pixels, px_path_lens_proj = digitise_path(Point(*entry_pos), Point(*exit_pos), px_norm)
    px_path_lens = pathlength_from_proj(px_path_lens_proj, px_norm, s1)

    int_per_pixel = np.exp(-mu * px_path_lens)

    abs_per_pixel = np.diff(1 - int_per_pixel, prepend=[0])
    return pixels, abs_per_pixel
