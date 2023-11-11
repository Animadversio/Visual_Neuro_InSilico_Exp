import torch
import torchvision
from os.path import join
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
ckpt_root = r"D:\DL_Projects\Vision\AdvPretrained_models"
pytorch_models = {
        'alexnet': torchvision.models.alexnet,
        'vgg16': torchvision.models.vgg16,
        'vgg16_bn': torchvision.models.vgg16_bn,
        'squeezenet': torchvision.models.squeezenet1_0,
        'densenet': torchvision.models.densenet161,
        'shufflenet': torchvision.models.shufflenet_v2_x1_0,
        'mobilenet': torchvision.models.mobilenet_v2,
        'resnet18' : torchvision.models.resnet18,
        'resnet50' : torchvision.models.resnet50,
        'wide_resnet50_2' : torchvision.models.wide_resnet50_2,
        # 'wide_resnet50_4' : missing in torchvision
        'resnext50_32x4d': torchvision.models.resnext50_32x4d,
        'mnasnet': torchvision.models.mnasnet1_0,
    }

ckptpaths = {
     'vgg16_bn' : 'vgg16_bn_l2_eps3.ckpt',
     'densenet' : 'densenet_l2_eps3.ckpt',
     'mobilenet' : 'mobilenet_l2_eps3.ckpt',
     'shufflenet' : 'shufflenet_l2_eps3.ckpt',
     'resnext50_32x4d' : 'resnext50_32x4d_l2_eps3.ckpt',
     'resnet18' : 'resnet18_l2_eps3.ckpt',
     'resnet50' : 'resnet50_l2_eps3.ckpt',
     'wide_resnet50_2' : 'wide_resnet50_2_l2_eps3.ckpt',
     'wide_resnet50_4' : 'wide_resnet50_4_l2_eps3.ckpt',  # this one is current not loadable
     'mnasnet' : 'mnasnet_l2_eps3.ckpt',  # this one is current not loadable
 }


def load_pretrained_robust_model(arch, ckptname=None, parallel=False):
    constructor = pytorch_models[arch]
    if ckptname is None:
        ckptname = ckptpaths[arch]

    if arch in ["resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "mnasnet"]:
        add_custom_forward = False
        # for these models there happen to be one less wrapper in the ckpt.
    else:
        add_custom_forward = True

    model, _ = make_and_restore_model(arch=constructor(pretrained=False),
              dataset=ImageNet("."), add_custom_forward=add_custom_forward, parallel=parallel,
              resume_path=join(ckpt_root, ckptname))
    if add_custom_forward:
        return model.model.model
    else:
        return model.model


# Function to load the model from the saved checkpoint
def load_model_from_converted_checkpoint(arch, checkpoint_path):
    model_constructor = pytorch_models[arch]
    model = model_constructor(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path))
    return model


if __name__ == "__main__":
    # Convert the checkpoints
    for arch in ckptpaths.keys():
        model = load_pretrained_robust_model(arch,)
        torch.save(model.state_dict(), join(ckpt_root, f"{arch}_pure.pt"))
        print(f"Successfully converted model '{arch}' to {join(ckpt_root, f'{arch}_pure.pt')}")

        checkpoint_path = join(ckpt_root, f"{arch}_pure.pt")
        try:
            model = load_model_from_converted_checkpoint(arch, checkpoint_path)
            print(f"Successfully loaded model '{arch}' from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load model '{arch}' from {checkpoint_path}: {e}")