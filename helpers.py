"""Helper for the diamond_square module."""

import numpy as np


def generate_indices(
    start: int | tuple[int, int],
    stop: int | tuple[int, int],
    step: int | tuple[int, int],
) -> np.ndarray:
    """Generate grid of (i, j) indices.

    start : int or tuple[int, int]
        If tuple, starting index for rows and columns, consecutively.
        If integer, starteing index for both rows and columns.
    stop : int or tuple[int, int]
        If tuple, stopping index for rows and columns, consecutively.
        If integer, stopping index for both rows and columns.
    step : int or tuple[int, int]
        If tuple, step for rows and columns, consecutively.
        If integer, step for both rows and columns.

    """
    if isinstance(start, int):
        start_i = start_j = start
    else:
        start_i, start_j = start
    if isinstance(stop, int):
        stop_i = stop_j = stop
    else:
        stop_i, stop_j = stop
    if isinstance(step, int):
        step_i = step_j = step
    else:
        step_i, step_j = step
    return np.array(
        np.meshgrid(
            np.arange(start_j, stop_j, step_j),
            np.arange(start_i, stop_i, step_i),
        ),
    ).reshape(2, -1)


def reshape_to_square(
    array: np.ndarray,
) -> np.ndarray:
    """Reshape an array to a square array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to reshape.

    Returns
    -------
    numpy.ndarray
        The same array, reshaped into a square array.

    Raises
    ------
    ValueError
        If the size of the array is not a perfect square.

    Notes
    -----
    The reshaping is done in Fortran order.

    """
    return array.reshape(int(np.sqrt(array.size)), -1, order="F")


def get_collapsor_array(
    array: np.ndarray,
) -> np.ndarray:
    """Return an array that sums `array`'s cosecutive rows.

    If array is of shape (m, n), `get_collapsor_array(array) @ array` is an
    array of shape (m - 1, n), such that row i is the sum of rows i and i + 1 of
    `array`.

    Parameters
    ----------
    array : numpy.array
        The array to "collapse".

    Returns
    -------
    numpy.ndarray
        The collapsor array, to multiply by `array`, in order to sum consecutive
        rows.

    """
    row = np.zeros(array.shape[0])
    row[:2] = 1
    return np.vstack([np.roll(row, i) for i in range(array.shape[0] - 1)])
