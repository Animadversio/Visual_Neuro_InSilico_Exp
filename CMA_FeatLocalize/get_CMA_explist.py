import os
from scipy.io import loadmat
import pandas as pd
from os.path import join
rootdir = "E:\OneDrive - Harvard University\CMA_localize"
matpath = "E:\OneDrive - Harvard University\CMA_localize\preMeta.mat"
#%%
premeta = loadmat(matpath, struct_as_record=False)["preMeta"][0]
#%%
dict_col = []
for i in range(len(premeta)):
    ephysFN = premeta[i].ephysFN[0]
    expControlFN = premeta[i].expControlFN[0]
    stimuli = premeta[i].stimuli[0]
    if premeta[i].comments.size == 0:
        comments = ""
    else:
        comments = premeta[i].comments[0]

    dict_col.append({"ephysFN":ephysFN, "expControlFN":expControlFN,
                     "stimuli":stimuli, "comments":comments,})

metatab = pd.DataFrame(dict_col)
metatab.to_csv(join(rootdir, "metatab.csv"))

#%%




