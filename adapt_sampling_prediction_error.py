from insilico_Exp_torch import TorchScorer
from GAN_utils import upconvGAN
from ZO_HessAware_Optimizers import CholeskyCMAES
G = upconvGAN("fc6") 
Neuron_model = TorchScorer
scorer = TorchScorer("vgg16")
scorer.select_unit(("vgg16", "fc2", 10, 10, 10))
# Given `Model` `Sampler` `Real_system`
# `Sampler` is a local optimizer + some random migrated samples. 
z_resv = []
r_resv = []
z = Z_INIT
while not CONVERGE:
    r_pred = Model.predict(z)
    r_obsv = eval(Real_system(r_pred))
    Model.update(z, r_obsv)
    score = abs(r_obsv - r_pred)
    z_new = Sampler.propose(z, score, max=True)
    z = z_new
    z_resv.extend(z_new)
    r_resv.extend(r_obsv)