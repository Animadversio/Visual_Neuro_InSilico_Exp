import os
from time import time
import re
import numpy as np
import torch
from cv2 import imread, imwrite
import matplotlib.pylab as plt
import sys
import os
sys.path.append("D:\Github\pytorch-caffe")
from caffenet import *

print(torch.cuda.current_device())
# print(torch.cuda.device(0))
if torch.cuda.is_available():
    print(torch.cuda.device_count(), " GPU is available:", torch.cuda.get_device_name(0))
#%%
