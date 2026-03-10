#!/usr/bin/env python
"""Numpy implemetation of the diamond square algorithm."""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from helpers import generate_indices, get_collapsor_array, reshape_to_square


def diamond_step(
    heightmap: np.ndarray,
    iteration: int,
) -> np.ndarray:
    """Perform the first step of the diamond-square algorith.

    Fills the heightmap in place.

    Parameters
    ----------
    heightmap : numpy.ndarray
        The heightmap to fill.
    iteration : int
        Iteration number of the diamond-square algorithm. Such that step 1 is
        the first one (that takes in as input the array with only the four
        corners filled).

    Returns
    -------
    numpy.ndarray
        The indices that were filled during this step, as a `(2, num_indices)`
        numpy array.

    """
    width = heightmap.shape[0]
    log_width = int(np.log2(heightmap.shape[0] - 1))  # width = 2^log_width + 1
    step = 2 ** (log_width - iteration)

    # Indices that are filled at this point
    filled_idx = generate_indices(
        start=0,
        stop=width,
        step=step,
    )

    # Extract only the filled subarray
    filled_subarray = reshape_to_square(heightmap[*filled_idx])

    # Indices to fill in this step
    tofill_idx = generate_indices(
        start=step // 2,
        stop=width,
        step=step,
    )

    # Use the collapsor array to sum all 2 by 2 blocks
    collapsor = get_collapsor_array(filled_subarray)
    heightmap[*tofill_idx] = (
        collapsor @ filled_subarray @ collapsor.T
    ).reshape(
        -1,
        order="F",
    ) / 4.0

    return tofill_idx


def square_step(
    heightmap: np.ndarray,
    iteration: int,
) -> np.ndarray:
    """Perform the second step of the diamond-square algorith.

    Fills the heightmap in place.

    Parameters
    ----------
    heightmap : numpy.ndarray
        The heightmap to fill.
    iteration : int
        Iteration number of the diamond-square algorithm. Such that step 1 is
        the first one (that takes in as input the array with only the four
        corners filled).

    Returns
    -------
    numpy.ndarray
        The indices that were filled during this step, as a `(2, num_indices)`
        numpy array.

    """
    width = heightmap.shape[0]
    log_width = int(np.log2(heightmap.shape[0] - 1))  # width = 2^log_width + 1
    step = 2 ** (log_width - iteration)

    # The filled subarray so far, excluding the values from the diamond step
    filled_idx = generate_indices(
        start=0,
        stop=width,
        step=step,
    )
    filled_subarray = reshape_to_square(heightmap[*filled_idx])

    # The subarray containing the values filled in the diamond step
    diamond_idx = generate_indices(
        start=step // 2,
        stop=width,
        step=step,
    )
    diamond_subarray = reshape_to_square(heightmap[*diamond_idx])

    # The same array but padded horizontally and vertically
    diamond_subarray_pad_h = np.pad(diamond_subarray, ((0, 0), (1, 1)))
    diamond_subarray_pad_v = np.pad(diamond_subarray, ((1, 1), (0, 0)))

    # Get collapsor arrays
    collapsor_filled = get_collapsor_array(filled_subarray)
    collapsor_diamond_h = get_collapsor_array(diamond_subarray_pad_h.T).T
    collapsor_diamond_v = get_collapsor_array(diamond_subarray_pad_v)

    # Fill the first part (horizontal part)
    tofill_idx_h = generate_indices(
        start=(0, step // 2),
        stop=width,
        step=step,
    )
    heightmap[*tofill_idx_h] = (
        collapsor_filled @ filled_subarray
        + diamond_subarray_pad_h @ collapsor_diamond_h
    ).reshape(-1, order="F")

    # Fill the first part (horizontal part)
    tofill_idx_v = generate_indices(
        start=(step // 2, 0),
        stop=width,
        step=step,
    )
    heightmap[*tofill_idx_v] = (
        filled_subarray @ collapsor_filled.T
        + collapsor_diamond_v @ diamond_subarray_pad_v
    ).reshape(-1, order="F")

    # Divide to transorm sums into averages
    tofill_idx = np.concat([tofill_idx_h, tofill_idx_v], axis=1)
    tofill_idx_outer = tofill_idx[
        :,
        np.any((tofill_idx == 0) | (tofill_idx == width - 1), axis=0),
    ]
    tofill_idx_inner = tofill_idx[
        :,
        np.all((tofill_idx > 0) & (tofill_idx < width - 1), axis=0),
    ]
    heightmap[*tofill_idx_outer] /= 3.0
    heightmap[*tofill_idx_inner] /= 4.0

    return tofill_idx


def generate_heightmap(
    log_width: int,
    log_decay: float,
    initial_multiplier: float,
    initial_heights: tuple[int, int, int, int],
    seed: int | None = None,
) -> np.ndarray:
    """Generate heightmap using the diamond-square algorithm.

    Parameters
    ----------
    log_width : int
        The width of the heightmap will be `2 ** log_width + 1`.
    log_decay : int
        The random noise added to the averages is multiplied by
        a multiplier which decays by a factor of `2 ** log_decay` after
        each iteration.
    initial_multiplier : float
        The initial value of the noise multiplier.
    initial_heights : tuple of 4 floats
        The heights of the four corners of the terrain in the following order:
        * Top left
        * Top right
        * Bottom right
        * Bottom left
    seed : int | None
        Seed for the random number generator. If None, no seed is set.

    Returns
    -------
    numpy.ndarray
        The generated heightmap, of shape
        `(2 ** log_width + 1, 2 ** log_width + 1)`.

    """
    rng = np.random.default_rng(seed=seed)
    heightmap = np.zeros((2**log_width + 1, 2**log_width + 1))
    heightmap[[0, 0, -1, -1], [0, -1, -1, 0]] = initial_heights
    scale = initial_multiplier
    for iteration in range(log_width):
        idx = np.concat(
            [
                diamond_step(heightmap, iteration),
                square_step(heightmap, iteration),
            ],
            axis=1,
        )
        heightmap[*idx] += rng.normal(scale=scale, size=heightmap[*idx].shape)
        scale /= 2**log_decay
    return heightmap


def plot_heightmap(
    heightmap: np.ndarray,
    figsize: tuple[int, int] | None = None,
    cmap_name: str = "viridis",
    sea_color: str | None = "c",
) -> None:
    """Plot the heightmap and send it to stdout.

    Parameters
    ----------
    heightmap : numpy.ndarray
        The heightmap to plot.
    figsize : tuple[int, int], optional
        Figure size. The default is (9, 9).
    cmap_name : str
        The color map to use. The default is "viridis".
    sea_color : str, optional
        A custom color for negative heights, used to simulate oceans. The
        default is "c" (for cyan). If None, negative heights are treated just
        like other values.

    """
    if figsize is None:
        figsize = (9, 9)
    _, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps.get_cmap(cmap_name)
    if sea_color:
        cmap.set_under(sea_color)
    if sea_color:
        ax.imshow(heightmap, cmap=cmap, vmin=0)
    else:
        ax.imshow(heightmap, cmap=cmap)
    ax.axis("off")
    ax.margins(0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(sys.stdout.buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a map and send it to the standard output",
    )
    parser.add_argument(
        "-w",
        "--log-width",
        type=int,
        default=8,
        help="log of the width of the generated map",
    )
    parser.add_argument(
        "-m",
        "--initial-multiplier",
        type=float,
        default=10.0,
        help="lnitial random noise multiplier",
    )
    parser.add_argument(
        "-d",
        "--log-decay",
        type=float,
        default=0.9,
        help="log of the noise decay factor",
    )
    parser.add_argument(
        "-i",
        "--initial-heights",
        nargs=4,
        default=(0, 0, 0, 0),
        help="initial height for the four corners (TL, TR, BR, BL)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="seed for the random number generator",
    )
    args = parser.parse_args()

    plot_heightmap(
        generate_heightmap(
            log_width=args.log_width,
            log_decay=args.log_decay,
            initial_multiplier=args.initial_multiplier,
            initial_heights=args.initial_heights,
            seed=args.seed,
        ),
    )
