"""Small lib to calculate RF by back prop towards the image."""
import numpy as np
import matplotlib.pylab as plt
from layer_hook_utils import register_hook_by_module_names, get_module_names
import torch, torchvision
from insilico_Exp_torch import get_activation, activation


def grad_RF_estimate(model, target_layer, target_unit, input_size=(3,227,227), device="cuda", show=True, reps=200, batch=1):
    # (slice(None), 7, 7)
    handle, module_names, module_types = register_hook_by_module_names(target_layer,
        get_activation("record", target_unit,ingraph=True), model,
        input_size, device=device, )

    cnt = 0
    # graddata = torch.zeros((1, 3, 227, 227)).cuda()
    gradabsdata = torch.zeros(input_size).cuda()
    for i in range(reps):
        intsr = torch.rand((batch, *input_size)).cuda() * 2 - 1
        intsr.requires_grad_(True)
        model(intsr)
        act_vec = activation['record']
        if act_vec.numel() > 1:
            act = act_vec.mean()
        else:
            act = act_vec
        if not torch.isclose(act, torch.tensor(0.0)):
            act.backward()
            # graddata += intsr.grad
            gradabsdata += intsr.grad.abs().mean(dim=0)
            cnt += 1
        else:
            continue

    for h in handle:
        h.remove()
    gradAmpmap = gradabsdata.permute([1, 2, 0]).abs().mean(dim=2).cpu() / cnt
    if show:
        plt.figure()
        plt.pcolor(gradAmpmap)
        plt.axis("image")
        plt.title("L %s Unit %s"%(target_layer, target_unit))
        plt.show()
        plt.figure()
        plt.hist(np.log10(1E-15 + gradAmpmap.flatten().cpu().numpy()), bins=100)
        plt.title("L %s Unit %s"%(target_layer, target_unit))
        plt.show()
    return gradAmpmap.numpy()


def gradmap2RF_square(gradAmpmap, absthresh=None, relthresh=0.01, square=True):
    maxAct = gradAmpmap.max()
    relthr = maxAct * relthresh
    if absthresh is None:
        thresh = relthr
    else:
        thresh = max(relthr, absthresh)
    Yinds, Xinds = np.where(gradAmpmap > thresh)
    Xlim = (Xinds.min(), Xinds.max()+1)
    Ylim = (Yinds.min(), Yinds.max()+1)
    if square:
        Xrng = Xlim[1] - Xlim[0]
        Yrng = Ylim[1] - Ylim[0]
        if Xrng == Yrng:
            pass
        elif Xrng > Yrng:
            print("Modify window to be square, before %s, %s"%(Xlim, Ylim))
            incre = (Xrng - Yrng) // 2
            Ylim = (Ylim[0] - incre, Ylim[1] + (Xrng - Yrng - incre))
            if Ylim[1] > gradAmpmap.shape[0]:
                offset = Ylim[1] - gradAmpmap.shape[0]
                Ylim = (Ylim[0] - offset, Ylim[1] - offset)
            if Ylim[0] < 0:
                offset = 0 - Ylim[0]
                Ylim = (Ylim[0] + offset, Ylim[1] + offset)
            print("After %s, %s"%(Xlim, Ylim))
        elif Yrng > Xrng:
            print("Modify window to be square, before %s, %s" % (Xlim, Ylim))
            incre = (Yrng - Xrng) // 2
            Xlim = (Xlim[0] - incre, Xlim[1] + (Yrng - Xrng - incre))
            if Xlim[1] > gradAmpmap.shape[1]:
                offset = Xlim[1] - gradAmpmap.shape[1]
                Xlim = (Xlim[0] - offset, Xlim[1] - offset)
            if Xlim[0] < 0:
                offset = 0 - Xlim[0]
                Xlim = (Xlim[0] + offset, Xlim[1] + offset)
            print("After %s, %s" % (Xlim, Ylim))
    return Xlim, Ylim


if __name__ == "__main__":
    from collections import OrderedDict
    resnet101 = torchvision.models.resnet101(pretrained=True).cuda()
    for param in resnet101.parameters():
        param.requires_grad_(False)
    resnet101.eval()
    # module_names, module_types = get_module_names(resnet101, (3,227,227))
    resnet_feat = torch.nn.Sequential(OrderedDict({"conv1": resnet101.conv1,
                                                   "bn1": resnet101.bn1,
                                                   "relu": resnet101.relu,
                                                   "maxpool": resnet101.maxpool,
                                                   "layer1": resnet101.layer1,
                                                   "layer2": resnet101.layer2,
                                                   "layer3": resnet101.layer3,
                                                   "layer4": resnet101.layer4}))
    unit_list = [("resnet101", ".ReLUrelu", 5, 56, 56),
                 ("resnet101", ".layer1.Bottleneck0", 5, 28, 28),
                 ("resnet101", ".layer1.Bottleneck1", 5, 28, 28),
                 ("resnet101", ".layer2.Bottleneck0", 5, 14, 14),
                 ("resnet101", ".layer2.Bottleneck3", 5, 14, 14),
                 ("resnet101", ".layer3.Bottleneck0", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck2", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck6", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck10", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck14", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck18", 5, 7, 7),
                 ("resnet101", ".layer3.Bottleneck22", 5, 7, 7), ]
    for unit in unit_list:
        print("Unit %s" % (unit,))
        gradAmpmap = grad_RF_estimate(resnet_feat, unit[1], (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                      device="cuda", show=True, reps=40, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
        Xlim, Ylim, (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0]), (Xlim[0], Ylim[0])))