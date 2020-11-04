Subpixel phase correlation for translational image registration, adapted from `skimage.registration.phase_cross_correlation` for Pytorch API with GPU support. 

The function allows for complex input. In that case, the real part and imaginary part of each image should be held in the last dimension of the image tensor - *e.g.*, the shape of the image tensor should be `[y, x, 2]`.

Note: This function may not work with automatic differentiation. I use it only for fast image registration on GPU outside of the graph. 
