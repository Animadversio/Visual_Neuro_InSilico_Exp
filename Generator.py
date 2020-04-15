import numpy as np
import net_utils


class Generator:
    '''Load CaffeNet generator

    Major use is to "visualize", detransform the code to the image
    '''
    def __init__(self, name="fc6"):
        if name == "fc6":
            generator = net_utils.load('generator')
        elif name == "fc7":
            generator = net_utils.load('generator-fc7')
        elif name == "fc8":
            generator = net_utils.load('generator-fc8')
        detransformer = net_utils.get_detransformer(generator)
        self._GNN = generator
        self._detransformer = detransformer

    def visualize(self, code, scale=255):
        x = self._GNN.forward(feat=code.reshape(1, 4096))['deconv0']
        x = self._detransformer.deprocess('data', x)
        x = np.clip(x, 0, 1)  # use clip to bound all the image output in interval [0,1]
        if scale == 255:
            return (x * 255).astype('uint8')  # rescale to uint in [0,255]
        else:
            return x

    def render(self, codes, scale=255):
        '''Render a list of codes to list of images'''
        if type(codes) is list:
            images = [self.visualize(codes[i], scale) for i in range(len(codes))]
        else:
            images = [self.visualize(codes[i, :], scale) for i in range(codes.shape[0])]
        return images

    def visualize_norm(self, code):
        """Add to visualize the un-cropped but min-max normalized image distribution"""
        x = self._GNN.forward(feat=code.reshape(1, 4096))['deconv0']
        x = self._detransformer.deprocess('data', x)
        # x = np.clip(x, 0, 1)  # use clip to bound all the image output in interval [0,1]
        x = (x - x.min()) / (x.max() - x.min())
        return (x * 255).astype('uint8')  # rescale to uint in [0,255]

    def raw_output(self, code):
        x = self._GNN.forward(feat=code.reshape(-1, 4096))['deconv0']
        return x