This is an implementation of [Defense-GAN](https://openreview.net/pdf?id=BkJ3ibb0-) as my deep learning assignment for Clova internship.

** Notes **
* it is written in Pytorch 1.1.0
* GAN
    * the DCGAN code is based on [pytorch's example](https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py)
    * the SNGAN code is based on the [official chainer implementation](https://github.com/pfnet-research/sngan_projection)
    * FID calculation code is based on the [the author's officially unofficial PyTorch BigGAN implementation](https://github.com/ajbrock/BigGAN-PyTorch)
        * Note that I use 4096 samples for FID calculation instead of the typical 5000 or 50000
        * due to this sample size, FID of real data is 1.25 instead of 0
        * FID calculation was wrong before commit `fff7a61656eeea9dd89adc9d877bbe70b1a6a91c` I think it's because torch_cov is different from numpy.cov
    * I could not find a good hyper parameter for my GAN, I don't have a GPU and I used colab with the wrong FID calculator(so the founded hyper parameters are not that great)
    * I also implemented a conditional GAN(with conditional batch normalization and projection) to improve the FID 
* CW2 attack is mostly borrowed from [here](https://github.com/kkew3/pytorch-cw2/)
* the MNIST classifier training loop is also from [pytorch examples](https://github.com/pytorch/examples/blob/master/mnist/main.py)

** How to run **
* train a classifier ``python classifier.py [--mlp]``
* attack the trained classifier ``python attacks.py``
* calculate inception statistics of MNIST ``python fid.py``
* train a GAN on MNIST ``python gan.py``
* evaluate different defense mechanisms ``python defenses.py``