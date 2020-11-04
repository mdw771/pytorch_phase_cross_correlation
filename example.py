from PIL import Image
from numpy.fft import *
import scipy.ndimage as ndi
from registration import phase_correlation
import matplotlib.pyplot as plt

import torch as tc
import numpy as np

img_0 = Image.open('cameraman.jpg')
img_0 = np.array(img_0).astype('float32')
# Shift image using Fourier shift.
img_shifted = ifft2(ndi.fourier_shift(fft2(img_0), [10.5, 25.8])).real

device = tc.device('cuda:0')
# Use zeros for imaginary parts. 
img_0_tc = tc.tensor(img_0, device=device)
img_0_tc = tc.stack([img_0_tc, tc.zeros_like(img_0_tc)], axis=-1)
img_shifted_tc = tc.tensor(img_shifted, device=device)
img_shifted_tc = tc.stack([img_shifted_tc, tc.zeros_like(img_shifted_tc)], axis=-1)
print(img_shifted_tc.shape, img_0_tc.shape)
shifts = phase_correlation(img_shifted_tc, img_0_tc, upsample_factor=10)
print(shifts)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_0)
axes[1].imshow(img_shifted)
plt.show()
