from NN_PC_visualize.NN_PC_lib import *
#%% Robust ResNet50
"""Record feature distribution and compute their SVD"""
# Load network
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
netname = "resnet50_linf8"

#%% Process images and record feature vectors
dataset = create_imagenet_valid_dataset()
reclayers = [".layer2.Bottleneck2", ".layer3.Bottleneck2", ".layer4.Bottleneck2"]
feattsrs = record_dataset(model, reclayers, dataset, return_input=False,
                   batch_size=125, num_workers=8)
torch.save(feattsrs, join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))

#%% SVD of image tensor
tsr_svds = feattsr_svd(feattsrs, device="cuda")
torch.save(tsr_svds, join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))


#%% VGG16
_, model = load_featnet("vgg16")
model.eval().cuda()
netname = "vgg16"
#%% Process images and record feature vectors
dataset = create_imagenet_valid_dataset()
reclayers = [".features.ReLU8", ".features.ReLU15", ".features.ReLU29",
             ".classifier.ReLU1", ".classifier.ReLU4"]
feattsrs = record_dataset(model, reclayers, dataset, return_input=False,
                   batch_size=50, num_workers=8)
torch.save(feattsrs, join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))  # 1 iter/sec 1000 iters totoal
#%% SVD of image tensor
tsr_svds = feattsr_svd(feattsrs, device="cpu")
torch.save(tsr_svds, join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))


#%% DenseNet
from robustCNN_utils import load_pretrained_robust_model
model = load_pretrained_robust_model("densenet")
model.eval().cuda()
netname = "densenet_robust"
dataset = create_imagenet_valid_dataset()
#%% Process images and record feature vectors
reclayers = [".features._DenseBlockdenseblock1", #".features.transition1.AvgPool2dpool",
             ".features._DenseBlockdenseblock2", #".features.transition2.AvgPool2dpool",
             ".features._DenseBlockdenseblock3", #".features.transition3.AvgPool2dpool",
             ".features._DenseBlockdenseblock4",
             ]
feattsrs = record_dataset(model, reclayers, dataset, return_input=False,
                   batch_size=40, num_workers=8)
torch.save(feattsrs, join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))  # 1 iter/sec 1000 iters totoal
#%  SVD of image tensor
tsr_svds = feattsr_svd(feattsrs, device="cpu")
torch.save(tsr_svds, join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
#%%
import torchvision
torchvision.models.densenet121()



#%% Dev zone, working pipeline, process dataset to get
# conv2 - conv3 - conv5 fc6, fc7
# reclayers = [".features.ReLU8", ".features.ReLU15", ".features.ReLU29",
#              ".classifier.ReLU1", ".classifier.ReLU4"]
# return_input = False
# fetcher = featureFetcher(model, device="cuda")
# for layer in reclayers:
#     fetcher.record(layer, return_input=return_input, ingraph=False)
# fetcher.record(".Linearfc", return_input=True, ingraph=False)
# loader = DataLoader(dataset, batch_size=125, shuffle=False, drop_last=False, num_workers=8)
# feat_col = defaultdict(list)
# feattsrs = {}
# for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
#     with torch.no_grad():
#         model(imgtsr.cuda())
#
#     for layer in reclayers:
#         if return_input:
#             feats_full = fetcher[layer][0].cpu()
#         else:
#             feats_full = fetcher[layer].cpu()
#         feats = slice_center_col(feats_full, ingraph=False)
#         feat_col[layer].append(feats)
#
# for layer in reclayers:
#     feattsrs[layer] = torch.cat(tuple(feat_col[layer]), dim=0)