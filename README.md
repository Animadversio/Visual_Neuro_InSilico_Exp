# Visual_Neuro_InSilico_Exp
 
This project serves the purpose of developping and testing computational tools *in silico* for future visual experiments *in vivo*. It has mulitple components, structure of project is the following: 

* NN Infrastructures 
    - Loading networks in Caffe backend `net_utils`
    - Loading Generator in `Generator`
    - Loading networks in Torch backend `torch_net_utils`. (Autograd for Hessian computation requires torch.) 
* A class of Experimental Objects defined in `insilico_Exp`
* Many types of Zeroth Order Optimizers defined in 
    - The older one in `Optimizer` CMA-ES, Cholesky CMA-ES, GA, CholeskyCMAES_Sphere
    - The series of optimizer based on the Hessian Aware ZO Optimization paper `ZO_HessAware_Optimizers`
    - The optimizers based on Powell's method `PowellOptimizers`
* Some objects target at estimating and analyzing Hessian for Black Box and non linear functions. 
    - `pytorch_CNN_hessian` 
    - `pytorch_GAN_similarity_hessian` 
    - A collection of functions to help analyze hessian matrix and spectrum `hessian_analysis`
    - Analysis code for many experiments `hessian_analysis_batch` 
    - A non-gradient Hessian estimation method inspired by the HessAware ZOO paper `ZO_Hessian_Estim`
* `Hessian` subfolder contains the experimental codes and infrastructures for the Hessian project. For a more structured code base, see [this repo](https://github.com/Animadversio/GAN-Geometry). 
* `movie_timescale` subfolder contains the code for measuring the response timescale for units across neural network, CNN or Transformers. 


## Usage 
This Repo can potentially be deployed onto cluster to run at large scale, based on torch (preferred) or Caffe or TF backend.


## Dependency

* [Pytorch Receptive Field](https://github.com/Fangyh09/pytorch-receptive-field) is used in some of the *in silico* experiments
* [pytorch hessian eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings) is used in some scripts


