call activate caffe36
cd D:\Github\Visual_Neuro_InSilico_Exp
python BigGAN_Evol_cluster.py --layer conv5 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv4 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv3 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv2 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv1 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: HessCMA800 HessCMA500_1 