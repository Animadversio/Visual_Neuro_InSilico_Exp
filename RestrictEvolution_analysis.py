basedir = r"D:\Generator_DB_Windows\data\with_CNN"
unit_arr = [('caffe-net', 'conv3', 5, 10, 10),
            #('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1)]

import numpy as np
from os.path import join
#%%
layer_names = [unit[1] for unit in unit_arr]
for unit in unit_arr[:1]:
    best_scores = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
best_scores.mean(axis=1)
#np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
#%%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=best_scores.mean(axis=1),
    #y=[1, 2, 3, 4, 5],
))

# fig.add_trace(go.Scatter(
#     x=[1, 2, 3, 4, 5],
#     y=[5, 4, 3, 2, 1],
# ))

fig.show()