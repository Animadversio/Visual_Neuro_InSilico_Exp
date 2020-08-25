#%%
from GAN_utils import upconvGAN, np
G = upconvGAN("fc6")
W = G.G.deconv0.weight.data.numpy()
#%%
def SingularValues(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)

SV = SingularValues(W.transpose((2,3,0,1)), (128,128))
