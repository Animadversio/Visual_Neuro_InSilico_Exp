call activate caffe36
ECHO Start Measure %Time%
ECHO Start Measure %Time% >> timer.txt
cd D:\Github\Visual_Neuro_InSilico_Exp
:: python BigGAN_Evol_cluster.py --layer conv5 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv4 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv3 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv2 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv1 --chans 25 40 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv5 --chans 40 50 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv4 --chans 40 50 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv3 --chans 40 50 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv2 --chans 40 50 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv1 --chans 40 50 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv5 --chans 10 25 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv4 --chans 10 25 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv3 --chans 10 25 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv2 --chans 10 25 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv1 --chans 10 25 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv5 --chans 50 70 --optim HessCMA CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv4 --chans 50 70 --optim HessCMA CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv3 --chans 50 70 --optim HessCMA CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv2 --chans 50 70 --optim HessCMA CholCMA --steps 100 --reps 5 --RFresize True
:: python BigGAN_Evol_cluster.py --layer conv1 --chans 50 64 --optim HessCMA CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv5 --chans 50 70 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv4 --chans 50 70 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv3 --chans 50 70 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv2 --chans 50 70 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
python BigGAN_Evol_cluster.py --layer conv1 --chans 50 64 --G fc6 --optim CholCMA --steps 100 --reps 5 --RFresize True
:: HessCMA800 HessCMA500_1
ECHO Stop Measure %Time%
ECHO Stop Measure %Time% >> timer.txt