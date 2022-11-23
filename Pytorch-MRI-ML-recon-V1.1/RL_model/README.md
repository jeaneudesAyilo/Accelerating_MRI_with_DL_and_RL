### Sampling trajectory learning with double deep q network + spirit reconstruction

Here the goal is to learn a RL model that determines from available data, the next most informative data to acquire. To put it simply, we know that an MNIST image has a shape of (28,28). Suppose we want to acquire 1/4 of columns ie 7 columns, and then reconstruct the full image using only these 7 columns. We start by pre-selecting 2 columns in the mild of the image, the rest of columns are not initially acquired. Then, the RL model will choose which is the next best column to acquire in order to have a more accurate reconstruction. So 5 columns will be acquired sequentially. Note that in our implementation we work in Fourrier domaine as Spirit model was implemented on this domain also. As a result, the q values network uses complexe-value operations. The following steps can summarize the methodology:

1- take a batch of kspace, were only 2 columns are acquired , and provide their reconstruction with the spirit reconstruction model

2- determine the third column to acquire with the q-values network, and compute the reward as the gain in reconstruction metric, update the mask by indicating the new column which is acquired (the action). The mask is specific to each k-space, this is different from our non sequential model where there was one mask for the whole dataset.

3- add the reconstructed k-space, the action, the reward,... to a replay buffer

4- repeat 2 and 3 until we acquire the 5 columns to reach 7 columns selected

5- update the parameters of the q-values using the data of the replay buffer (if enough data are available in the replay buffer) 

These steps are repeated on several batchs.

Note that the reconstructor model we used, is the one learned in this [notebook](../notebook_current_working_version_fft_norm_loss_norm/network_with_normalize_1D_mask_mnist.ipynb). This reconstructor was learned on only one acceleration factor (ie 4 ; 1/4 of clumns ), but we should learn it on different acceleration factors (from 3/28 columns to 7/28 columns) to ensure that it can make good job for reconstruction on lower than 7 columns acquired. That is a drawback of this methodology based on RL, as we need to use a pretrained reconstructor. Of course, this have already been solved in studies that didn't even apply RL, and learn in an end-to-end way a reconstruction model and a sequential sampler. See for example [this article](https://arxiv.org/abs/2105.06460). More sophisticated q-values network and reconstructor can also be used as in the [main paper](https://arxiv.org/abs/2007.10469) we follow (https://facebookresearch.github.io/active-mri-acquisition/index.html). Others study didn't impose a structure of the trajectory (non cartesian trajectory, see [this](https://arxiv.org/abs/2204.02480)). All in all, there are so many possibilities.


