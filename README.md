Subpixel phase correlation for translational image registration, adapted from `skimage.registration.phase_cross_correlation` for Pytorch API with GPU support. 

Note: This function may not work with automatic differentiation. I use it only for fast image registration on GPU outside of the graph. 
