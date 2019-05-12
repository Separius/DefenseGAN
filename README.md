This is an implementation of [Defense-GAN](https://openreview.net/pdf?id=BkJ3ibb0-) as my deep learning assignment for Clova internship.

** How to run **
* train a classifier ``python classifier.py [--mlp] [--adv]``
* attack the trained classifier ``python attacks.py``(it also saves the generated examples for the evaluation phase)
* calculate inception statistics of MNIST ``python fid.py``
* train a GAN on MNIST ``python gan.py``
* train an auto encoder on MNIST ``python autoencoder.py``
* evaluate different defense mechanisms ``python defenses.py``

** Notes **
* it is written in Pytorch 1.1.0 and uses tb-nightly (for the new tensorboard feature in pytorch)
* the DCGAN code is based on [pytorch's example](https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py)
* the SNGAN code is based on the [official chainer implementation](https://github.com/pfnet-research/sngan_projection)
* FID calculation code is based on the [the author's officially unofficial PyTorch BigGAN implementation](https://github.com/ajbrock/BigGAN-PyTorch)
        * Note that I use 4096 samples for FID calculation instead of the typical 5000 or 50000
        * due to this sample size, FID of real data is 1.25 instead of 0
        * FID calculation was wrong before commit `fff7a61656eeea9dd89adc9d877bbe70b1a6a91c` I think it's because torch_cov is different from numpy.cov
    * I could not find a good hyper parameter for my GAN, I don't have a GPU and I used colab with the wrong FID calculator(so the founded hyper parameters are not that great)
    * I also implemented a conditional GAN(with conditional batch normalization and projection) to improve the FID(which didn't :/)
* CW2 attack is mostly borrowed from [here](https://github.com/kkew3/pytorch-cw2/)
* the MNIST classifier training loop is also from [pytorch examples](https://github.com/pytorch/examples/blob/master/mnist/main.py)
* I was able to train a good GAN, but I was unable to use it to reconstruct data points well
    * I guess that spectral normalization in the generator can help
    * `L` and `R` parameters of the paper are important
    * also because I don't have a GPU right now I couldn't set L to a larger number and instead found that setting a high R is as good!
    * so it seems that adversarial examples don't have a good representation on the generator's latent manifold and it's enough to just pick a random restart with minimum L2 distance and do not fine-tune the z
* I will upload all the trained models to my google drive and just put some pictures in the repo

** Plots and Samples **
* early samples from the generator
<p align="center"> 
	<img src="assets/gan_samples_early.png">
</p>
* final samples from the generator
<p align="center"> 
	<img src="assets/gan_samples_good.png">
</p>
* gan reconstruction example
<p align="center"> 
	<img src="assets/recon1.gif">
</p>
* autoencoder results
<p align="center"> 
	<img src="assets/autoencoder.png">
</p>
* attacks on the cnn classifier
<p align="center"> 
	<img src="assets/attacks_on_cnn.png">
</p>
** Interesting Defence Results **

| **Defence Method** | **Classifier** | **Adversarially Trained?** | **Attack Method** | **Accuracy with Defence** | **Accuracy without Defence** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| binarize image | cnn | yes | fgsm_0.15 | 0.9812 | 0.9786 |
| binarize image | cnn | yes | cw2 | 0.9843 | 0.9816 |
| binarize image | cnn | no | fgsm_0.3 | 0.9057 | 0.416 |
| binarize image | cnn | no | cw2 | 0.9767 | 0.0 |
| binarize image | mlp | no | cw2 | 0.9588 | 0.0007 |
| binarize image | mlp | no | bb_cnn | 0.928 | 0.6089 |
| binarize image | mlp | no | bb_mlp | 0.9434 | 0.6692 |
| gaussian kernel | cnn | no | rfgsm | 0.5685 | 0.583 |
| gaussian kernel | cnn | no | cw2 | 0.7161 | 0.0 |
| autoencoder | cnn | yes | none | 0.8967 | 0.9918 |
| autoencoder | cnn | yes | fgsm_0.3 | 0.5702 | 0.9491 |
| autoencoder | cnn | yes | rfgsm | 0.6517 | 0.9618 |
| autoencoder | cnn | yes | bb_cnn | 0.5418 | 0.9768 |
| autoencoder | cnn | no | fgsm_0.3 | 0.5268 | 0.416 |
| autoencoder | cnn | no | rfgsm | 0.6167 | 0.583 |
| autoencoder | cnn | no | cw2 | 0.8326 | 0.0 |
| autoencoder->binarize | cnn | yes | none | 0.8854 | 0.9918 |
| autoencoder->binarize | cnn | yes | fgsm_0.3 | 0.5613 | 0.9491 |
| autoencoder->binarize | cnn | yes | rfgsm | 0.6384 | 0.9618 |
| autoencoder->binarize | cnn | yes | bb_cnn | 0.5289 | 0.9768 |
| autoencoder->binarize | cnn | no | fgsm_0.3 | 0.5508 | 0.416 |
| autoencoder->binarize | cnn | no | rfgsm | 0.6289 | 0.583 |
| autoencoder->binarize | cnn | no | cw2 | 0.8242 | 0.0 |
| binarize->autoencoder | cnn | yes | none | 0.8895 | 0.9918 |
| binarize->autoencoder | cnn | yes | fgsm_0.3 | 0.8463 | 0.9491 |
| binarize->autoencoder | cnn | yes | rfgsm | 0.8581 | 0.9618 |
| binarize->autoencoder | cnn | yes | bb_cnn | 0.8342 | 0.9768 |
| binarize->autoencoder | cnn | no | fgsm_0.3 | 0.8438 | 0.416 |
| binarize->autoencoder | cnn | no | rfgsm | 0.8535 | 0.583 |
| binarize->autoencoder | cnn | no | cw2 | 0.8647 | 0.0 |
| gan(best initial z) | cnn | yes | none | 0.5499 | 0.9918 |
| gan(best initial z) | cnn | yes | fgsm_0.15 | 0.5275 | 0.9786 |
| gan(best initial z) | cnn | yes | fgsm_0.3 | 0.5059 | 0.9491 |
| gan(best initial z) | cnn | yes | rfgsm | 0.5097 | 0.9618 |
| gan(best initial z) | cnn | yes | cw2 | 0.5158 | 0.9816 |
| gan(best initial z) | cnn | yes | bb_cnn | 0.4723 | 0.9768 |
| gan(best initial z) | cnn | yes | bb_mlp | 0.4905 | 0.9829 |
| gan(best initial z) | cnn | no | none | 0.5548 | 0.989 |
| gan(best initial z) | cnn | no | fgsm_0.15 | 0.5315 | 0.865 |
| gan(best initial z) | cnn | no | fgsm_0.3 | 0.4984 | 0.416 |
| gan(best initial z) | cnn | no | rfgsm | 0.5078 | 0.583 |
| gan(best initial z) | cnn | no | cw2 | 0.5094 | 0.0 |
| gan(best initial z) | cnn | no | bb_cnn | 0.4696 | 0.9415 |
| gan(best initial z) | cnn | no | bb_mlp | 0.488 | 0.9679 |
| gan(best initial z) | mlp | yes | none | 0.5549 | 0.9762 |
| gan(best initial z) | mlp | yes | fgsm_0.15 | 0.5377 | 0.954 |
| gan(best initial z) | mlp | yes | fgsm_0.3 | 0.513 | 0.9085 |
| gan(best initial z) | mlp | yes | rfgsm | 0.5166 | 0.9279 |
| gan(best initial z) | mlp | yes | cw2 | 0.5466 | 0.9725 |
| gan(best initial z) | mlp | yes | bb_cnn | 0.4921 | 0.9288 |
| gan(best initial z) | mlp | yes | bb_mlp | 0.4936 | 0.9456 |
