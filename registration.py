import torch as tc
import numpy as np

def complex_mul(a_real, a_imag, b_real, b_imag):
    return (a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real)


def fft2(var_real, var_imag, axes=(-2, -1), normalize=False):
    var = tc.stack([var_real, var_imag], dim=-1)
    var = tc.fft(var, signal_ndim=2, normalized=normalize)
    var_real, var_imag = tc.split(var, 1, dim=-1)
    slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
    return var_real[tuple(slicer)], var_imag[tuple(slicer)]


def ifft2(var_real, var_imag, axes=(-2, -1), normalize=False):
    var = tc.stack([var_real, var_imag], dim=-1)
    var = tc.ifft(var, signal_ndim=2, normalized=normalize)
    var_real, var_imag = tc.split(var, 1, dim=-1)
    slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
    return var_real[tuple(slicer)], var_imag[tuple(slicer)]


def tensordot(a, b, axes=None, override_backend=None):
    """
    :param axes: Comply to Numpy format.
    """
    dims = axes
    if isinstance(axes, (list, tuple)):
        if isinstance(axes[0], int):
            dims = []
            for i in axes:
                dims.append((axes[i],))
    return tc.tensordot(a, b, dims=dims)


def exp_complex(var_real, var_imag):
    if not isinstance(var_real, tc.Tensor):
        var_real = tc.tensor(var_real)
    if not isinstance(var_imag, tc.Tensor):
        var_real = tc.tensor(var_imag)
    e = tc.exp(var_real)
    return e * tc.cos(var_imag), e * tc.sin(var_imag)


def _upsampled_dft(data_real, data_imag, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    A wrapped version of the synonym function in scikit-image.
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)
    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * len(data_real.shape)
    else:
        if len(upsampled_region_size) != len(data_real.shape):
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * len(data_real.shape)
    else:
        if len(axis_offsets) != len(data_real.shape):
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    im2pi_imag = 2 * np.pi

    dim_properties = list(zip(data_real.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        ffreq = tc.tensor(np.fft.fftfreq(n_items, upsample_factor), device=data_real.device)
        a = (tc.arange(ups_size) - ax_offset)[:, None]
        a = tc.tensor(a, device=data_real.device)
        kernel = a * ffreq
        kernel_real, kernel_imag = exp_complex(0., -im2pi_imag * kernel)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        # data = np.tensordot(kernel, data, axes=(1, -1))
        d_real = tensordot(kernel_real, data_real, axes=(1, -1)) - tensordot(kernel_imag, data_imag, axes=(1, -1))
        d_imag = tensordot(kernel_real, data_imag, axes=(1, -1)) + tensordot(kernel_imag, data_real, axes=(1, -1))
        data_real = d_real
        data_imag = d_imag
    return data_real, data_imag


def phase_correlation(img, ref, upsample_factor=1):
    """
    An adaption of skimage.registration.phase_cross_correlation for Pytorch, enabling GPU support.
    Perform phase correlation to find relative translational shift.

    :param img: Tensor. In shape [y, x, 2], where the last dimension holds real and imaginary parts.
    :param ref: Tensor. In shape [y, x, 2], where the last dimension holds real and imaginary parts.
    :param upsample_factor: Int. Images will be registered to within `1 / upsample_factor` of a pixel.
    :return: Shift as [dy, dx]. It is the relative shift of img with regards to ref. In other words, you can shift
             img by -shifts to get ref.
    """
    img_shape = img.shape[:2]
    size = img_shape[0] * img_shape[1]
    f_img_real, f_img_imag = fft2(img[:, :, 0], img[:, :, 1])
    f_ref_real, f_ref_imag = fft2(ref[:, :, 0], ref[:, :, 1])
    prod_real, prod_imag = complex_mul(f_img_real, f_img_imag, f_ref_real, -f_ref_imag)
    cc_real, cc_imag = ifft2(prod_real, prod_imag)
    cc = cc_real ** 2 + cc_imag ** 2
    shifts = tc.argmax(cc)
    shifts = tc.tensor([shifts // img_shape[1], shifts % img_shape[1]], device=img.device).float()

    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = tc.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        upsampled_region_size = tc.tensor([upsampled_region_size, upsampled_region_size], device=img.device)
        # Center of output array at dftshift + 1
        dftshift = tc.trunc(upsampled_region_size / 2.0)
        normalization = (size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cc_real, cc_imag = _upsampled_dft(prod_real, -prod_imag,
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset)
        cc_imag = -cc_imag
        cc_real /= normalization
        cc_imag /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = tc.argmax(cc_real ** 2 + cc_imag ** 2)
        maxima = [maxima // cc_real.shape[1], maxima % cc_real.shape[1]]

        maxima = tc.tensor(maxima, device=shifts.device) - dftshift

        shifts = shifts + maxima / upsample_factor
    return shifts
