from keras import initializers
from comp7017.config import generate_pfm

import numpy as np

glorot_kernels = initializers.GlorotUniform()(
    shape = (3,3,1,28)
).numpy()

pfm = generate_pfm()
print(type(pfm), pfm.shape, pfm.dtype)
print(type(glorot_kernels), glorot_kernels.shape, glorot_kernels.dtype)
kernel_module = np.concatenate(
    [pfm, glorot_kernels],
    axis=-1
)