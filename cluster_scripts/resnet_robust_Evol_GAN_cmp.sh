
python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G BigGAN --net resnet50_linf_8 --layer .layer3.Bottleneck4 --optim HessCMA CholCMA --rep 3 # --RFresize True
python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G fc6 --net resnet50_linf_8 --layer .layer3.Bottleneck4 --optim HessCMA500  --rep 3 # --RFresize True

python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G BigGAN --net resnet50_linf_8 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 3 # --RFresize True
python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G fc6 --net resnet50_linf_8 --layer .layer4.Bottleneck2 --optim HessCMA500  --rep 3 # --RFresize True


python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G BigGAN --net resnet50_linf_8 --layer .Linearfc --optim HessCMA CholCMA --rep 3 # --RFresize True
python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G fc6 --net resnet50_linf_8 --layer .Linearfc --optim HessCMA500  --rep 3 # --RFresize True

python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G BigGAN --net resnet50_linf_8 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 3 # --RFresize True
python BigGAN_Evol_cmp_RIS_cluster.py --chans 0 50 --G fc6 --net resnet50_linf_8 --layer .layer2.Bottleneck3 --optim HessCMA500  --rep 3 # --RFresize True

