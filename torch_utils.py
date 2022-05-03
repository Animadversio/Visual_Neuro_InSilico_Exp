import os.path
import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from build_montages import make_grid_T
def show_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    PILimg.show()
    return PILimg


def save_imgrid(img_tsr, path, *args, **kwargs):
    PILimg = ToPILImage()(make_grid_T(img_tsr.cpu(), *args, **kwargs))
    PILimg.save(path)
    return PILimg


def save_imgrid_by_row(img_tsr, path, n_row=5, *args, **kwargs):
    """Seperate img_tsr into rows and save them into different png files, with numbering 0-n."""
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    n_total = img_tsr.shape[0]
    row_num = np.ceil(n_total // n_row).astype(int)  # n_total // n_row + 1
    stem, ext = os.path.splitext(path)
    for i in range(row_num):
        PILimg = ToPILImage()(make_grid(img_tsr[i * n_row: (i + 1) * n_row].cpu(), n_row=5, *args, **kwargs))
        PILimg.save(stem + "_" + str(i) + ext)
    return