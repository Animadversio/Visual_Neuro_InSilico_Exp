from NN_PC_visualize.NN_PC_lib import *
from tqdm import tqdm
from glob import glob
from easydict import EasyDict
import scipy.optimize as opt
from robustCNN_utils import load_pretrained_robust_model
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square, twoD_Gaussian, fit_2dgauss
def show_gradmap(gradAmpmap, pop_str, rfdir):
    plt.figure(figsize=[5.8, 5])
    plt.imshow(gradAmpmap)
    plt.colorbar()
    plt.title(f"{pop_str}", fontsize=14)
    plt.savefig(join(rfdir, f"{pop_str}_gradAmpMap.png"))
    plt.show()


rfdir = r"H:\CNN-PCs\RF_estimate"
#%% ResNet50 Robust
netname = "resnet50_linf8"
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
reclayers = [".layer2.Bottleneck2",
             ".layer3.Bottleneck2",
             ".layer4.Bottleneck2"]
for layer in reclayers:
    pop_str = f"{netname}-{layer}"
    cent_pos = get_cent_pos(model, layer, imgfullpix=256)
    gradAmpmap = grad_RF_estimate(model, layer, (slice(None), cent_pos[0], cent_pos[1]),
                          input_size=(3, 256, 256), device="cuda", show=False, reps=200, batch=4)
    show_gradmap(gradAmpmap, pop_str, rfdir)
    fitdict = fit_2dgauss(gradAmpmap, pop_str, rfdir)

#%% VGG16
_, model = load_featnet("vgg16")
model.eval().cuda()
netname = "vgg16"
reclayers = [".features.ReLU8", ".features.ReLU15", ".features.ReLU29",
             ".classifier.ReLU1", ".classifier.ReLU4"]
for layer in reclayers[3:]:
    pop_str = f"{netname}-{layer}"
    if ".classifier" not in layer:
        # get center of the feature map
        cent_pos = get_cent_pos(model, layer, imgfullpix=256)
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), cent_pos[0], cent_pos[1]),
                                      input_size=(3, 256, 256), device="cuda", show=False, reps=200, batch=4)
    else:
        # Use the full fc layer to find RF ?
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), ),
                                      input_size=(3, 256, 256), device="cuda", show=False, reps=200, batch=4)
    show_gradmap(gradAmpmap, pop_str, rfdir)
    fitdict = fit_2dgauss(gradAmpmap, pop_str, rfdir)

#%% DenseNet Robust
netname = "densenet_robust"
model = load_pretrained_robust_model("densenet")
model.eval().cuda()
reclayers = [".features._DenseBlockdenseblock1", #".features.transition1.AvgPool2dpool",
             ".features._DenseBlockdenseblock2", #".features.transition2.AvgPool2dpool",
             ".features._DenseBlockdenseblock3", #".features.transition3.AvgPool2dpool",
             ".features._DenseBlockdenseblock4",
             ]
for layer in reclayers:
    pop_str = f"{netname}-{layer}"
    if ".classifier" not in layer:
        # get center of the feature map
        cent_pos = get_cent_pos(model, layer, imgfullpix=256)
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), cent_pos[0], cent_pos[1]),
                                      input_size=(3, 256, 256), device="cuda", show=False, reps=200, batch=4)
    else:
        # Use the full fc layer to find RF ?
        gradAmpmap = grad_RF_estimate(model, layer, (slice(None), ),
                                      input_size=(3, 256, 256), device="cuda", show=False, reps=200, batch=4)
    show_gradmap(gradAmpmap, pop_str, rfdir)
    fitdict = fit_2dgauss(gradAmpmap, pop_str, rfdir)

#%%
