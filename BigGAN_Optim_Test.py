from insilico_Exp import *
from ZO_HessAware_Optimizers import HessAware_Gauss_Cylind_DC, HessAware_Gauss_DC
from BigGAN_Evolution import one_hot_from_names, model, BigGAN_render, BigGAN_embed_render
import torch

#%%
unit = ('caffe-net', 'fc8', 1)
savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Generator_Testing\BigGAN_Evol"
# savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
expdir = join(savedir, "%s_%s_%d_Gauss_DC_Cylind" % unit)
os.makedirs(expdir, exist_ok=True)
# lr = 3; mu = 0.002;
# lr_norm = 5; mu_norm = 1
lr = 1; mu = 0.05 # 0.1
Lambda=1; trial_i=0
classname = "table"
class_vector = one_hot_from_names([classname],  batch_size=1)
with torch.no_grad():
    ebd_class = model.embeddings(torch.from_numpy(class_vector).cuda()).cpu().numpy()
ebd_class_long = np.concatenate((np.zeros_like(ebd_class), ebd_class), axis=1)
# f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
# sys.stdout = f
# optim = HessAware_Gauss_Cylind_DC(128, population_size=40, lr_norm=lr_norm, mu_norm=mu_norm, lr_sph=lr_sph, mu_sph=mu_sph, Lambda=Lambda, Hupdate_freq=201,
#             rankweight=True, nat_grad=True, maximize=True, max_norm=10)
optim = HessAware_Gauss_DC(256, lr=lr, mu=mu, Lambda=1, Hupdate_freq=201,
            maximize=True, max_norm=300, rankweight=True, nat_grad=True)
optim_name = str(optim.__class__).split(".")[1].split("'")[0]
fn_str = "%s-256init-%s_lr%.1f_mu%.4f_Lambda%.2f_unbound_tr%d" % (optim_name, classname, lr, mu, Lambda, trial_i)

experiment = ExperimentEvolve_DC(unit, max_step=50, optimizer=optim, GAN="BigGAN")
experiment.run(init_code=ebd_class_long)
param_str = "%s, init%s, lr=%.1f, mu=%.2f, Lambda=%.2f." % (optim_name, classname, lr, mu, optim.Lambda)
fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
fig2 = experiment.visualize_best(show=False, title_str="")
fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
fig3 = experiment.visualize_exp(show=False, title_str=param_str)
fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
plt.show(block=False)
sleep(5)
plt.close('all')
# sys.stdout = orig_stdout