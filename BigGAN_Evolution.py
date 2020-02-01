from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
import torch
import numpy as np
# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')


# Prepare a input
batch_size = 5
truncation = 0.4
class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=batch_size)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

# If you have a GPU, put everything on cuda
noise_vector = noise_vector.to('cuda')
class_vector = class_vector.to('cuda')
model.to('cuda')

# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)


def render(codes, scale=255):
    '''Render a list of codes to list of images'''
    if type(codes) is list:
        images = [generator.visualize(codes[i], scale) for i in range(len(codes))]
    else:
        images = [generator.visualize(codes[i, :], scale) for i in range(codes.shape[0])]
    return images

# Generate Gaussian on simplex...?
# Generate Gaussian noise, (With Small enough sigma and variance)
class CNNmodel_Torch:
    """ Basic CNN scorer
    Demo:
        CNN = CNNmodel('caffe-net')  #
        CNN.select_unit( ('caffe-net', 'conv1', 5, 10, 10) )
        scores = CNN.score(imgs)
        # if you want to record all the activation in conv1 layer you can use
        CNN.set_recording( 'conv1' ) # then CNN.artiphys = True
        scores, activations = CNN.score(imgs)

    """
    def __init__(self, model_name):
        if model_name == "caffe-net":
            self._classifier = load_caffenet()
        self._transformer = net_utils.get_transformer(self._classifier, scale=1)
        self.artiphys = False

    def select_unit(self, unit_tuple):
        self._classifier_name = str(unit_tuple[0])
        self._net_layer = str(unit_tuple[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self._net_iunit = int(unit_tuple[2])
        # this index is used to extract the scalar response `self._net_iunit`
        if len(unit_tuple) == 5:
            self._net_unit_x = int(unit_tuple[3])
            self._net_unit_y = int(unit_tuple[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

    def set_recording(self, record_layers):
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers = record_layers
        self.recordings = {}
        for layername in record_layers:  # will be arranged in a dict of lists
            self.recordings[layername] = []

    # def forward(self, imgs):
    #     return recordings

    def score(self, images, with_grad=False):
        scores = np.zeros(len(images))
        for i, img in enumerate(images):
            # Note: now only support single repetition
            tim = self._transformer.preprocess('data', img)  # shape=(3, 227, 227) # assuming input scale is 0,1 output will be 0,255
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)  # propagate the image the target layer
            # record only the neuron intended
            score = self._classifier.blobs[self._net_layer].data[0, self._net_iunit]
            if self._net_unit_x is not None:
                # if `self._net_unit_x/y` (inside dimension) are provided, then use them to slice the output score
                score = score[self._net_unit_x, self._net_unit_y]
            scores[i] = score
            if self.artiphys:  # record the whole layer's activation
                for layername in self.record_layers:
                    score_full = self._classifier.blobs[layername].data[0, :]
                    # self._pattern_array.append(score_full)
                    self.recordings[layername].append(score_full.copy())
        if self.artiphys:
            return scores, self.recordings
        else:
            return scores


