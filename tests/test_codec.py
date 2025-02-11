from anscombe_numcodecs import AnscombeCodec
import numpy as np
import pytest

DEBUG = False

def make_poisson_ramp_signals(shape=(10, 1, 1), min_rate=1, max_rate=5, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    x, y, times = shape
    output_array = np.zeros(shape, dtype=dtype)
    for x_ind in range(x):
        for y_ind in range(y):
            output_array[x_ind, y_ind, :] = sensitivity * np.random.poisson(
                np.arange(min_rate, max_rate, (max_rate-min_rate)/times))
    return output_array.astype('int16')

sensitivity = 100.0

@pytest.fixture
def test_data(dtype="int16"):
    test2d =  make_poisson_ramp_signals(shape=(50, 1, 1), min_rate=1, max_rate=5, dtype=dtype)
    test2d_long = make_poisson_ramp_signals(shape=(1, 50, 1), min_rate=1, max_rate=5, dtype=dtype)
    return [test2d, test2d_long]

def test_poisson_encode_decode(test_data):
    codec = AnscombeCodec(
        zero_level=0,
        photon_sensitivity=sensitivity,
        encoded_dtype='uint8', 
        decoded_dtype='int16'
    )
    
    for example_data in test_data:
        encoded = codec.encode(example_data)
        decoded = codec.decode(encoded)
        recoded = codec.decode(codec.encode(decoded))
        assert np.abs(decoded - example_data).max() < sensitivity / 2
        assert (decoded == recoded).all()
         
if __name__ == '__main__':
    list_data = test_data("int16")
    test_poisson_encode_decode(list_data)
