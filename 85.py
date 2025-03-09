import numpy as np

def pos_encoding(position: int, d_model: int):
    positions = np.arange(position)[:, np.newaxis]

    angle_rads = positions * (1 / np.power(10000, (2 * np.arange(d_model//2)) / np.float32(d_model)))[np.newaxis, :]
    sines = np.sin(angle_rads)
    cosines = np.cos(angle_rads)

    pos_encoding = np.zeros((position, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines

    pos_encoding = pos_encoding[np.newaxis, :, :]
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding

print(pos_encoding(2, 8))
