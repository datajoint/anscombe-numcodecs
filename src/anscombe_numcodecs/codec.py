"""
Numcodecs Codec implementation for Anscombe Transform for photon-limited data.
"""

import numpy as np
import numcodecs
from numcodecs.abc import Codec


def make_anscombe_lookup(
    sensitivity: float,
    input_max: int = 0x7FFF,
    zero_level: int = 0,
    beta: float = 0.5,
    output_type="uint8",
):
    """
    Compute the Anscombe lookup table.
    The lookup converts a linear grayscale image into a uniform variance image.
    :param sensitivity: the size of one photon in the linear input image.
    :param input_max: the maximum value in the input
    :param beta: the grayscale quantization step expressed in units of noise std dev
    """
    xx = (
        np.r_[: input_max + 1] - zero_level
    ) / sensitivity  # input expressed in photon rates
    zero_slope = 1 / beta / np.sqrt(3 / 8)  # slope for negative values
    offset = zero_level * zero_slope / sensitivity
    lookup_table = np.round(
        offset
        + (xx < 0) * (xx * zero_slope)
        + (xx >= 0)
        * (2.0 / beta * (np.sqrt(np.maximum(0, xx) + 3 / 8) - np.sqrt(3 / 8)))
    )
    lookup = lookup_table.astype(output_type)
    assert np.diff(lookup_table).min() >= 0, "non-monotonic lookup generated"
    return lookup


def make_inverse_lookup(lookup_table: np.ndarray, output_type="int16") -> np.ndarray:
    """Compute the inverse lookup table for a monotonic forward lookup table."""
    _, inv1 = np.unique(lookup_table, return_index=True)  # first entry
    _, inv2 = np.unique(lookup_table[::-1], return_index=True)  # last entry
    inverse = (inv1 + lookup_table.size - 1 - inv2) / 2
    return inverse.astype(output_type)


def lookup(movie: np.ndarray, lookup_table: np.ndarray) -> np.ndarray:
    """Apply lookup table to movie"""
    return lookup_table[np.maximum(0, np.minimum(movie, lookup_table.size - 1))]


class AnscombeCodec(Codec):
    """Codec for 3-dimensional Filter. The codec assumes that input data are of shape:
    (time, x, y).

    Parameters
    ----------
    zero_level : float
        Signal level when no photons are recorded.
        This should pre-computed or measured directly on the instrument.
    photon_sensitivity : float
        Conversion scalor to convert the measure signal into absolute photon numbers.
        This should pre-computed or measured directly on the instrument.
    """

    codec_id = "anscombe-v1"

    def __init__(
        self,
        zero_level,
        photon_sensitivity,
        encoded_dtype="int8",
        decoded_dtype="int16",
    ):
        self.zero_level = zero_level
        self.photon_sensitivity = photon_sensitivity
        self.encoded_dtype = encoded_dtype
        self.decoded_dtype = decoded_dtype

    def encode(self, buf: np.ndarray) -> np.ndarray:
        lookup_table = make_anscombe_lookup(
            self.photon_sensitivity,
            output_type=self.encoded_dtype,
            zero_level=self.zero_level,
        )
        encoded = lookup(buf, lookup_table)
        shape = [encoded.ndim] + list(encoded.shape)
        shape = np.array(shape, dtype="uint32")
        return shape.tobytes() + encoded.astype(self.encoded_dtype).tobytes()

    def decode(self, buf: bytes, out=None) -> np.ndarray:
        lookup_table = make_anscombe_lookup(
            self.photon_sensitivity,
            output_type=self.encoded_dtype,
            zero_level=self.zero_level,
        )
        inverse_table = make_inverse_lookup(
            lookup_table, output_type=self.decoded_dtype
        )
        ndims = np.frombuffer(buf[:4], "uint32")[0]
        shape = np.frombuffer(buf[4 : 4 * (ndims + 1)], "uint32")
        decoded = np.frombuffer(
            buf[(ndims + 1) * 4 :], dtype=self.encoded_dtype
        ).reshape(shape)
        return lookup(decoded, inverse_table).astype(self.decoded_dtype)


numcodecs.register_codec(AnscombeCodec)
