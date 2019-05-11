This is an implementation of [Defense-GAN](https://openreview.net/pdf?id=BkJ3ibb0-) as my deep learning assignment for Clova internship.

** Notes **
* it is written in Pytorch 1.1.0
* the DCGAN code is based on [pytorch's example](https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py)
* the SNGAN code is based on the [official chainer implementation](https://github.com/pfnet-research/sngan_projection)
* FID calculation code is based on the [the author's officially unofficial PyTorch BigGAN implementation](https://github.com/ajbrock/BigGAN-PyTorch)
* CW2 attack is mostly borrowed from [here](https://github.com/kkew3/pytorch-cw2/)
* the MNIST classifier training loop is also from [pytorch examples](https://github.com/pytorch/examples/blob/master/mnist/main.py)

** How to run **
* train a classifier ``python classifier.py [--mlp]``
* attack the trained classifier ``python attacks.py``
* calculate inception statistics of MNIST ``python fid.py``
* train a GAN on MNIST ``python gan.py``
* evaluate different defense mechanisms ``python defenses.py``