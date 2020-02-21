import sys
from os.path import join
# sys.path.append("D:\Github\pytorch-pretrained-BigGAN")
# sys.path.append("E:\Github_Projects\pytorch-pretrained-BigGAN")
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images)
import torch
import numpy as np
import matplotlib.pylab as plt
#%%
from scipy.stats import truncnorm
def convert_to_images(obj):
    """ Convert an output tensor from BigGAN in a list of images.
        Params:
            obj: tensor or numpy array of shape (batch_size, channels, height, width)
        Output:
            list of Pillow Images of size (height, width)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Please install Pillow to use images: pip install Pillow")

    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()

    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(Image.fromarray(out_array))
    return img

def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

#%%
# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')
model.to('cuda')
def BigGAN_render(class_vector, noise_vector, truncation):
    if class_vector.shape[0] == 1:
        class_vector = np.tile(class_vector, [noise_vector.shape[0], 1])
    if noise_vector.shape[0] == 1:
        noise_vector = np.tile(noise_vector, [class_vector.shape[0], 1])
    class_vector = torch.from_numpy(class_vector.astype(np.float32)).to('cuda')
    noise_vector = torch.from_numpy(noise_vector.astype(np.float32)).to('cuda')
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)
    imgs = convert_to_images(output.cpu())
    return imgs
#%%
# Prepare a input
batch_size = 3
truncation = 0.5
class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=batch_size)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)

noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
# All in tensors
#noise_vector = torch.from_numpy(np.ones([3, 128]).astype(np.float32)) #
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)
# If you have a GPU, put everything on cuda
noise_vector = noise_vector.to('cuda')
class_vector = class_vector.to('cuda')
model.to('cuda')
# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)
imgs = convert_to_images(output.cpu())
#%% 1d interpolation
truncation = 0.7
batch_size = 11
class_vector = one_hot_from_names(['mushroom']*batch_size, batch_size=1)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
scale_vec = np.arange(-1, 1.1, 0.2)
noise_vec_scale = scale_vec.reshape([-1, 1])*noise_vector
imgs = BigGAN_render(class_vector, noise_vec_scale, truncation=truncation)
#%
figh = plt.figure(figsize=[25, 3])
gs = figh.add_gridspec(1, len(imgs)) # 1d interpolation
for i, img in enumerate(imgs):
    plt.subplot(gs[i])
    plt.imshow(img)
    plt.axis('off')
    plt.title("{0:.2f}".format(scale_vec[i]), fontsize=15,)
plt.show()
#%%
savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
truncation = 0.7
# batch_size = 11
classname = 'goldfish'
class_vector = one_hot_from_names([classname], batch_size=1)
#%% 1d interpolation and save
truncation = 0.7
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
scale_UL = 1; scale_BL = -scale_UL; sample_n = 11
scale_vec = np.linspace(scale_BL, scale_UL, sample_n)
# scale_vec = np.linspace(-2.5, -0.9, sample_n)
noise_vec_scale = scale_vec.reshape([-1, 1])*noise_vector
imgs = BigGAN_render(class_vector, noise_vec_scale, truncation=truncation)
figh = plt.figure(figsize=[25, 3])
gs = figh.add_gridspec(1, len(imgs)) # 1d interpolation
for i, img in enumerate(imgs):
    plt.subplot(gs[i])
    plt.imshow(img)
    plt.axis('off')
    plt.title("{0:.1f}".format(scale_vec[i]), fontsize=15,)
plt.savefig(join(savedir, "%s_UL%.1f_BL%.1f_trunc%.1f_%04d.png" % (classname, scale_UL, scale_BL, truncation, np.random.randint(10000))))
plt.show()
#%%
from numpy.linalg import norm
def orthonorm(ref, vec2):
    res = vec2 - vec2 @ ref.T * ref / norm(ref, axis=1)**2
    return res / norm(res) * norm(ref)
#%% 2d linear interpolation through center
savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
truncation = 0.7
# batch_size = 11
classname = 'goldfish'
class_vector = one_hot_from_names([classname], batch_size=1)
truncation = 0.7
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=2)
vec1 = noise_vector[0:1, :]
vec2 = orthonorm(vec1, noise_vector[1:2, :])
xlim = (-1, 1)
ylim = (-1, 1); sample_n = 11
x_scale_vec = np.linspace(*xlim, sample_n)
y_scale_vec = np.linspace(*ylim, sample_n)
# scale_vec = np.linspace(-2.5, -0.9, sample_n)
imgs = []
for ydeg in y_scale_vec:
    noise_vec_scale = x_scale_vec[:, np.newaxis] * vec1 + ydeg * vec2
    img_row = BigGAN_render(class_vector, noise_vec_scale, truncation=truncation)
    imgs.append(img_row)
#%
figh = plt.figure(figsize=[25, 25])
gs = figh.add_gridspec(len(y_scale_vec), len(x_scale_vec))  # 2d interpolation
for i, img_row in enumerate(imgs):
    for j, img in enumerate(img_row):
        plt.subplot(gs[i, j])
        plt.imshow(img)
        plt.axis('off')
        plt.title("%.1f, %.1f"%(x_scale_vec[i], y_scale_vec[j]), fontsize=15,)
plt.tight_layout()
plt.savefig(join(savedir, "%s_[%.1f-%.1f]_[%.1f-%.1f]_trunc%.1f_%04d.png" % (classname, *xlim, *ylim, truncation, np.random.randint(10000))))
plt.show()

#%% 2d interpolation in sphere
savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
truncation = 0.4
# batch_size = 11
classname = 'goldfish'
class_vector = one_hot_from_names([classname], batch_size=1)
truncation = 0.4
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)
vec1 = noise_vector[0:1, :]
vec2 = orthonorm(vec1, noise_vector[1:2, :])
vec3 = orthonorm(vec2, noise_vector[2:3, :])
vec3 = orthonorm(vec1, vec3)

sample_n = 11
phi_scale_vec = np.linspace(-90, 90, sample_n)
theta_scale_vec = np.linspace(-90, 90, sample_n)
# scale_vec = np.linspace(-2.5, -0.9, sample_n)
imgs = []
for phi in phi_scale_vec:
    phi = phi / 180 * np.pi
    theta = theta_scale_vec[:, np.newaxis] / 180 * np.pi
    noise_vec_arc = np.cos(phi) * np.cos(theta) * vec1 + \
                      np.cos(phi) * np.sin(theta) * vec2 + \
                      np.sin(phi) * vec3
    img_row = BigGAN_render(class_vector, noise_vec_arc, truncation=truncation)
    imgs.append(img_row)
#%
figh = plt.figure(figsize=[25, 25])
gs = figh.add_gridspec(len(theta_scale_vec), len(phi_scale_vec))  # 2d interpolation
for i, img_row in enumerate(imgs):
    for j, img in enumerate(img_row):
        plt.subplot(gs[i, j])
        plt.imshow(img)
        plt.axis('off')
        plt.title("%.1f, %.1f"%(theta_scale_vec[i], phi_scale_vec[j]), fontsize=15,)
plt.tight_layout()
plt.savefig(join(savedir, "%s_hemisph_trunc%.1f_%04d.png" % (classname, truncation, np.random.randint(10000))))
plt.show()
#%% Multi class sampling
from scipy.special import softmax
#%%
savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
truncation = 0.7
# batch_size = 11
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
classname = 'goldfish'
class_vector = np.ones([1, 1000])  # one_hot_from_names([classname], # batch_size=1)
sample_n = 11
mu = 40
class_vectors = class_vector + np.random.randn(sample_n, 1000) / 32 * mu
class_vectors = np.concatenate((class_vector, class_vectors), axis=0)
# class_vectors = np.clip(class_vectors, 0, np.inf)
# class_vectors = class_vectors / 1000
class_vectors = 2*softmax(class_vectors, axis=1)
    # np.exp(class_vectors) / np.sum(np.exp(class_vectors), axis=1)
imgs = BigGAN_render(class_vectors, noise_vector, truncation=truncation)
figh = plt.figure(figsize=[25, 3])
gs = figh.add_gridspec(1, len(imgs)) # 1d interpolation
for i, img in enumerate(imgs):
    plt.subplot(gs[i])
    plt.imshow(img)
    plt.axis('off')
    #plt.title("{0:.1f}".format(scale_vec[i]), fontsize=15,)
plt.savefig(join(savedir, "%s_multiclass_softmax_mu%s_trunc%.1f_%04d.png" % (classname, mu, truncation, np.random.randint(10000))))
plt.show()
#%% Directly manipulate the embedding
savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
truncation = 0.7
muembd = 0.02
classname = 'goldfish'
class_vector = one_hot_from_names([classname],  batch_size=1)
ebd_class = model.cpu().embeddings(torch.from_numpy(class_vector))
ebd_vecs = muembd * torch.randn((10, ebd_class.shape[1])) + ebd_class
output = model.cuda().generator(torch.cat((torch.zeros_like(ebd_vecs.cuda()), ebd_vecs.cuda()), dim=1), truncation)
imgs = convert_to_images(output.cpu())
figh = plt.figure(figsize=[25, 3])
gs = figh.add_gridspec(1, len(imgs)) # 1d interpolation
for i, img in enumerate(imgs):
    plt.subplot(gs[i])
    plt.imshow(img)
    plt.axis('off')
    #plt.title("{0:.1f}".format(scale_vec[i]), fontsize=15,)
plt.savefig(join(savedir, "%s_embdspace_mu%s_trunc%.1f_%04d.png" % (classname, muembd, truncation, np.random.randint(10000))))
plt.show()
#%%
plt.figure(figsize=[4, 4])
plt.imshow(imgs[0])
plt.axis("image")
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
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


