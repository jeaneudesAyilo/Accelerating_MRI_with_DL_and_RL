# Accelerating_MRI_with_DL_and_RL

In the context of Magnetic Resonance Imaging, it is important to accelerate the acquisition time. Different
reconstruction model have already been proposed for this purpose. The current study aims to
propose a model that jointly learns undersampling pattern to select most important part in the data and
reconstructed the undersampled data. We exploit the SPIRiT (Iterative Self-consistent Parallel Imaging
Reconstruction From Arbitrary k-Space, [Lustig and Pauly, 2010](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.22428) )
reconstruction model and extend it to include and sampler model.
The sampler could perform sequentially (using reinforcement learning (RL)) or not, but at the current step
our results are limited to the non sequential sampling with local and global sampling mask. We apply
the model to MNIST dataset and a simulated brain data set. For MNIST, we found that the global mask
performed better than the local and other baseline masks. Conversely for the MRI data, the Caipirinha
baseline mask outperforms the learned masks and the learned local mask perform better than the learned
global mask. Future works, will consist in implementation of the sampler with reinforcement learning based on [Pineda et al 2020](https://arxiv.org/abs/2007.10469) and [their repository](https://github.com/facebookresearch/active-mri-acquisition). 


## Installation

* Clone this repository : `git clone https://github.com/jeaneudesAyilo/Accelerating_MRI_with_DL_and_RL.git`
* Create environment, example : `conda create --name pytorch-mri python=3.9.7` 
* install the required packages : `pip install -r myrequirements.txt` or install manually as described in `installed_packages.txt`.

## Some notebooks versions 

* Check [here](./Pytorch-MRI-ML-recon-V1.1/notebook_current_working_version_fft_norm_loss_norm/network_with_normalize_2D_mask_mnist.ipynb), for 2D learned/fixed mask and spirit reconstruction on mnist. This notebook has been repeatedly used with slight variations.
* Check this [notebook](./Pytorch-MRI-ML-recon-V1.1/notebook_current_working_version_fft_norm_loss_norm/network_with_normalize_2D_mask_mri.ipynb) for 2D learned/fixed mask and spirit reconstruction on mri data set. This notebook has been repeatedly used with slight variations.


## To launch the grid search (non RL) on cluster 

### For mnist experiment
* `cd "./Pytorch-MRI-ML-recon-V1.1/experiment on mnist"`

For learned mask : 
* `cd ./mask_general_learned_normalize_code_new`
* run the command `source grid_search_general.sh -e learn_general -c config_general.json`. This will create different json files each of them correspond to a given configurations. All the configurations are defined in grid_search_general.sh
* then run : `sbatch run_job_array.sh`. This will create  a job array and take a given number of models as jobs and run then in parallel.

For fixed mask : 
* `cd ./mask_fixed_normalize_code_new`
* generate a dataframe for list of masks and particularly the number of random masks to be used. Example : 
`python generate_list_of_mask_df.py  ./list_of_mask_df.csv 30 123 `
* create configs : 
`source grid_search_fixedmask.sh -e learn_fixedmask -c config_fixedmask.json -m list_of_mask_df.csv`
* run models : `sbatch run_job_array.sh`

### For mri experiment
#### Data generation
* `cd "./Pytorch-MRI-ML-recon-V1.1`

* simulate train brain volumes `python generate_mr_data_for_mask_learning.py --saving_path ./data/data_for_mask_learning/output/train --contrast_seed 123 --nb_volumes 24 --noise_seed 123`

* simulate test brain volumes `python generate_mr_data_for_mask_learning.py --saving_path ./data/data_for_mask_learning/output/test --contrast_seed 123 --nb_volumes 24 --noise_seed 123`

Or simply `python generate_mr_data_for_mask_learning.py --saving_path ./data/data_for_mask_learning/output/train --contrast_seed 123 --nb_volumes 30 --noise_seed 123`  and then move the last 6 folders into test folder.

* Create numpy array data 
 `cd ./experiment on mri/mask_learned_normalize_mri_2`
 `python save_data.py` 

 Then run learned masks models (folder mask_learned_normalize_mri_2) and fixed masks models (mask_fixed_normalize_mri) as in mnist case: 
#### For learned mask
* `cd ./experiment on mri/mask_learned_normalize_mri_2`
* `source grid_search_general.sh -e learn_general -c config_general.json`.
* `sbatch run_job_array.sh`.

#### For fixed mask
* `cd ./experiment on mri/mask_fixed_normalize_code_new`
* `python generate_list_of_mask_df.py  ./list_of_mask_df.csv 20 123 `
* `source grid_search_fixedmask.sh -e learn_fixedmask -c config_fixedmask.json -m list_of_mask_df.csv`
* `sbatch run_job_array.sh`

### Get results 
Once the models have finished running, get a summary table of the performance. For example, 
* `cd "./Pytorch-MRI-ML-recon-V1.1/experiment on mnist/mask_general_learned_normalize_code_new"`
* `python get_grid_search_results.py`

### Out of memory issue
When running the models on the cluster, we faced out-of-memory error which made some of them not run. Increase the memory (in run_job_array.sh file) didn't necessarily solve the problem. As a result we may need to re-run models for which there was out-of-memory error. Folders `./Pytorch-MRI-ML-recon-V1.1/mask_general_learned_normalize_code_new_out_of_memory` and `./Pytorch-MRI-ML-recon-V1.1/mask_fixed_normalize_code_new_out_of_memory` illustrate it. For example, if we run the grid search of learned mask on mnist and there is a out-of memory, we first need to identify which model to re execute by listing the models which directories are empty in the result directory (it is just a possible way to do it, there may be other solution).

* `cd ./mask_general_learned_normalize_code_new/result`
* `find . –type d -empty > out_of_memory.csv` or check https://linuxhint.com/list-empty-directories-in-linux/  
* `cd ./experiment on mnist/mask_fixed_normalize_code_new_out_of_memory`. The rest is the same as previously
* `source grid_search_general.sh -e learn_general -c config_general.json`. has ben adapted For the circumstance
* `sbatch run_job_array.sh`
