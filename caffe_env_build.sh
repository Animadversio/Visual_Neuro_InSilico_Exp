mamba create -n caffe
mamba activate caffe
# mamba install -c anaconda caffe-gpu
mamba install -c anaconda caffe
mamba install caffe pytorch torchvision -c pytorch -c nvidia -c anaconda
# mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
# mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install pytorch torchvision -c pytorch -c nvidia