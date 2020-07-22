# this is the python library created for using BigGAN in evolution.
import sys
from os.path import join
sys.path.append("C:/Users/zhanq/OneDrive - Washington University in St. Louis/GitHub/pytorch-pretrained-BigGAN")
# sys.path.append("E:\Github_Projects\pytorch-pretrained-BigGAN")
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, one_hot_from_int, truncated_noise_sample, convert_to_images)
import torch
import numpy as np
import matplotlib.pylab as plt
#%%
#%%
from numpy.linalg import norm
def orthonorm(ref, vec2):
    res = vec2 - vec2 @ ref.T * ref / norm(ref, axis=1)**2
    return res / norm(res) * norm(ref)
#%%
from scipy.stats import truncnorm
def convert_to_images_np(obj, scale=1.0):
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
    obj = np.clip(((obj + 1) / 2.0) * scale, 0, scale)
    img = []
    for i, out in enumerate(obj):
        img.append(out)
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

def BigGAN_embed_render(embed_vecs, noise_vecs=None, truncation=0.7, scale=255.0, batch=5):
    if embed_vecs.shape[1] == 256:
        input_vecs = torch.from_numpy(embed_vecs)
    elif embed_vecs.shape[1] == 128:
        if noise_vecs is None:
            embed_vecs = torch.from_numpy(embed_vecs)
            input_vecs = torch.cat((torch.zeros_like(embed_vecs), embed_vecs), dim=1)
        else:
            assert noise_vecs.shape[1] == 128
            if noise_vecs.shape[0] == embed_vecs[0]:
                input_vecs = torch.cat((torch.from_numpy(noise_vecs), torch.from_numpy(embed_vecs)), dim=1)
            else:
                assert noise_vecs.shape[0] == 1
                noise_vecs = np.tile(noise_vecs, [embed_vecs.shape[0], 1])
                input_vecs = torch.cat((torch.from_numpy(noise_vecs), torch.from_numpy(embed_vecs)), dim=1)
    sample_n = input_vecs.shape[0]
    imgs_all = []
    csr = 0
    csr_end = 0
    while csr_end < sample_n:
        csr_end = min(csr + batch, sample_n)
        with torch.no_grad():
            output = model.generator(input_vecs[csr:csr_end, :].float().cuda(), truncation)
            # imgs = convert_to_images(output.cpu())
            # imgs = [np.array(img).astype(np.float64) / 255 * scale for img in imgs]
            imgs = convert_to_images_np(output.cpu(), scale)
            imgs_all.extend(imgs)
        csr = csr_end
    return imgs_all

if __name__=="__main__":
    # %%
    # Prepare a input
    batch_size = 3
    truncation = 0.5
    class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=batch_size)
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)

    #noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
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
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
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

    #%% 2d linear interpolation through center
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
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
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
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
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
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
    #%% Directly manipulate the 128D embedding space
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
    truncation = 0.7
    muembd = 0.06
    classname = 'goldfish'
    # one_hot_from_int
    class_vector = one_hot_from_names([classname],  batch_size=1)
    ebd_class = model.embeddings(torch.from_numpy(class_vector).cuda())
    ebd_vecs = muembd * torch.randn((10, ebd_class.shape[1])).cuda() + ebd_class  # add Gaussian perturbation around
    with torch.no_grad():
        output = model.generator(torch.cat((torch.zeros_like(ebd_vecs), ebd_vecs), dim=1), truncation)
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

    #%% Linear 2d interpolation in the embedding space
    truncation = 0.7
    muembd = 0.1
    classname = 'dog'
    class_vector = one_hot_from_names([classname],  batch_size=1)
    ebd_class = model.embeddings(torch.from_numpy(class_vector).cuda())
    perturb_vector = muembd * np.random.randn(2, ebd_class.shape[1]) # truncated_noise_sample(truncation=truncation, batch_size=2)
    vec1 = perturb_vector[0:1, :]
    vec2 = orthonorm(vec1, perturb_vector[1:2, :])
    xlim = (-1, 1)
    ylim = (-1, 1); sample_n = 11
    x_scale_vec = np.linspace(*xlim, sample_n)
    y_scale_vec = np.linspace(*ylim, sample_n)
    # scale_vec = np.linspace(-2.5, -0.9, sample_n)
    imgs = []
    for ydeg in y_scale_vec:
        perturb_vec_scale = (x_scale_vec[:, np.newaxis] * vec1 + ydeg * vec2).astype(np.float32)
        emb_class_vec = torch.from_numpy(perturb_vec_scale).cuda() + ebd_class
        #ebd_vecs = muembd * torch.randn((10, ebd_class.shape[1])).cuda() + ebd_class  # add Gaussian perturbation around
        with torch.no_grad():
            output = model.generator(torch.cat((torch.zeros_like(emb_class_vec), emb_class_vec), dim=1), truncation)
        img_row = convert_to_images(output.cpu())
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
    plt.savefig(join(savedir, "%s_embdspace_mu%s_[%.1f-%.1f]_[%.1f-%.1f]__trunc%.1f_%04d.png" % (classname, muembd, *xlim, *ylim, truncation, np.random.randint(10000))))
    plt.show()
    #%%
    plt.figure(figsize=[4, 4])
    plt.imshow(imgs[0])
    plt.axis("image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
#%%
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
    truncation = 0.7
    muembd = 0.1
    classname = [1,2];
    class_vector = one_hot_from_int([classname], batch_size=len(classname))

    ebd_class = model.embeddings(torch.from_numpy(class_vector).cuda())
    ebd_vecs = muembd * torch.randn((10, ebd_class.shape[1])).cuda() + ebd_class  # add Gaussian perturbation around
    with torch.no_grad():
        output = model.generator(torch.cat((torch.zeros_like(ebd_vecs), ebd_vecs), dim=1), truncation)
    imgs = convert_to_images(output.cpu())
    figh = plt.figure(figsize=[25, 3])
    gs = figh.add_gridspec(1, len(imgs))  # 1d interpolation
    for i, img in enumerate(imgs):
        plt.subplot(gs[i])
        plt.imshow(img)
        plt.axis('off')
        # plt.title("{0:.1f}".format(scale_vec[i]), fontsize=15,)
    plt.savefig(join(savedir, "%s_embdspace_mu%s_trunc%.1f_%04d.png" % (
    classname, muembd, truncation, np.random.randint(10000))))
    plt.show()
#%%
    savedir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN256"
    truncation = 0.7
    muembd = 0.1

    classname = 'dog'
    class_vector = one_hot_from_int([], batch_size=1)
    model.embeddings
    classname = [1,2];
    class_vector = one_hot_from_int([classname], batch_size=len(classname))

    ebd_class = model.embeddings(torch.from_numpy(class_vector).cuda())
    ebd_vecs = muembd * torch.randn((10, ebd_class.shape[1])).cuda() + ebd_class  # add Gaussian perturbation around
    with torch.no_grad():
        output = model.generator(torch.cat((torch.zeros_like(ebd_vecs), ebd_vecs), dim=1), truncation)
    imgs = convert_to_images(output.cpu())
    figh = plt.figure(figsize=[25, 3])
    gs = figh.add_gridspec(1, len(imgs))  # 1d interpolation
    for i, img in enumerate(imgs):
        plt.subplot(gs[i])
        plt.imshow(img)
        plt.axis('off')
        # plt.title("{0:.1f}".format(scale_vec[i]), fontsize=15,)
    plt.savefig(join(savedir, "%s_embdspace_mu%s_trunc%.1f_%04d.png" % (
    classname, muembd, truncation, np.random.randint(10000))))
    plt.show()

