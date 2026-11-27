import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from keras import initializers

# Predefined edge and line filters adopted from Linse et al. (2024),
# Figure 2, p. 5,
# https://arxiv.org/abs/2410.16897
EDGE_BIAS_KERNELS = (
    # 1
    ((-0.17,  0.00,  0.17),
     (-0.17,  0.00,  0.17),
     (-0.17,  0.00,  0.17)),

    # 2
    ((-0.29, -0.11,  0.00),
     (-0.11,  0.00,  0.11),
     ( 0.00,  0.11,  0.29)),

    # 3
    ((-0.17, -0.17, -0.17),
     ( 0.00,  0.00,  0.00),
     ( 0.17,  0.17,  0.17)),

    # 4
    (( 0.00, -0.11, -0.29),
     ( 0.11,  0.00, -0.11),
     ( 0.29,  0.11,  0.00)),

    # 5
    (( 0.08, -0.17,  0.08),
     ( 0.08, -0.17,  0.08),
     ( 0.08, -0.17,  0.08)),

    # 6
    (( 0.17, -0.05, -0.14),
     (-0.05,  0.17, -0.05),
     (-0.14, -0.05,  0.17)),

    # 7
    (( 0.08,  0.08,  0.08),
     (-0.17, -0.17, -0.17),
     ( 0.08,  0.08,  0.08)),

    # 8
    ((-0.14, -0.05,  0.17),
     (-0.05,  0.17, -0.05),
     ( 0.17, -0.05, -0.14)),

    # 9
    ((-0.06, -0.06, -0.06),
     (-0.06,  0.50, -0.06),
     (-0.06, -0.06, -0.06))
)

def generate_prior_kernels(arrays : list | tuple = EDGE_BIAS_KERNELS) -> np.ndarray:
    """
    Returns
    -------
    NDArray[np.float32]
        Convolutional kernels of shape
        (height, width, input_channels, num_filters)
    """

    prior_kernels_positive = np.stack(
        arrays,
        axis=-1
    )[:,:,np.newaxis,:].astype(np.float32)

    prior_kernels = np.concatenate(
        [prior_kernels_positive, - prior_kernels_positive],
        axis=-1
    )

    print(f"Prior kernel array shape: {prior_kernels.shape}")

    return prior_kernels

def generate_random_kernels(shape: tuple, initializer = initializers.GlorotUniform) -> np.ndarray:
    """
    Returns
    -------
    NDArray[np.float32]
        Convolutional kernels of shape
        (height, width, input_channels, num_filters)
    """
    random_kernels = np.asarray(
        initializer()(shape=shape),
        dtype=np.float32
    )
    print(f"Random kernel array shape: {random_kernels.shape}")

    return np.asarray(
        initializer()(shape=shape),
        dtype=np.float32
    )

def construct_initializer(prior_kernels: np.ndarray, random_kernels: np.ndarray) -> np.ndarray:
    """
    Returns
    -------
    NDArray[np.float32]
        Convolutional kernels of shape
        (height, width, input_channels, num_filters)
    """
    return np.concatenate(
        [prior_kernels, random_kernels],
        axis=-1
    )