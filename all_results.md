| **Defence Method** | **Classifier** | **Adversarially Trained?** | **Attack Method** | **Accuracy with Defence** | **Accuracy without Defence** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| binarize image | cnn | yes | none | 0.9897 | 0.9918 |
| binarize image | cnn | yes | fgsm_0.15 | 0.9812 | 0.9786 |
| binarize image | cnn | yes | fgsm_0.3 | 0.9674 | 0.9491 |
| binarize image | cnn | yes | rfgsm | 0.9728 | 0.9618 |
| binarize image | cnn | yes | cw2 | 0.9843 | 0.9816 |
| binarize image | cnn | yes | bb_cnn | 0.98 | 0.9768 |
| binarize image | cnn | yes | bb_mlp | 0.9832 | 0.9829 |
| binarize image | cnn | no | none | 0.9866 | 0.989 |
| binarize image | cnn | no | fgsm_0.15 | 0.9626 | 0.865 |
| binarize image | cnn | no | fgsm_0.3 | 0.9057 | 0.416 |
| binarize image | cnn | no | rfgsm | 0.9293 | 0.583 |
| binarize image | cnn | no | cw2 | 0.9767 | 0.0 |
| binarize image | cnn | no | bb_cnn | 0.9705 | 0.9415 |
| binarize image | cnn | no | bb_mlp | 0.9769 | 0.9679 |
| binarize image | mlp | yes | none | 0.975 | 0.9762 |
| binarize image | mlp | yes | fgsm_0.15 | 0.9641 | 0.954 |
| binarize image | mlp | yes | fgsm_0.3 | 0.9476 | 0.9085 |
| binarize image | mlp | yes | rfgsm | 0.954 | 0.9279 |
| binarize image | mlp | yes | cw2 | 0.9721 | 0.9725 |
| binarize image | mlp | yes | bb_cnn | 0.955 | 0.9288 |
| binarize image | mlp | yes | bb_mlp | 0.9602 | 0.9456 |
| binarize image | mlp | no | none | 0.9669 | 0.9754 |
| binarize image | mlp | no | fgsm_0.15 | 0.8968 | 0.2177 |
| binarize image | mlp | no | fgsm_0.3 | 0.7353 | 0.0172 |
| binarize image | mlp | no | rfgsm | 0.7954 | 0.0399 |
| binarize image | mlp | no | cw2 | 0.9588 | 0.0007 |
| binarize image | mlp | no | bb_cnn | 0.928 | 0.6089 |
| binarize image | mlp | no | bb_mlp | 0.9434 | 0.6692 |
| gaussian kernel | cnn | yes | none | 0.978 | 0.9918 |
| gaussian kernel | cnn | yes | fgsm_0.15 | 0.9489 | 0.9786 |
| gaussian kernel | cnn | yes | fgsm_0.3 | 0.8811 | 0.9491 |
| gaussian kernel | cnn | yes | rfgsm | 0.9115 | 0.9618 |
| gaussian kernel | cnn | yes | cw2 | 0.9369 | 0.9816 |
| gaussian kernel | cnn | yes | bb_cnn | 0.9122 | 0.9768 |
| gaussian kernel | cnn | yes | bb_mlp | 0.9505 | 0.9829 |
| gaussian kernel | cnn | no | none | 0.9619 | 0.989 |
| gaussian kernel | cnn | no | fgsm_0.15 | 0.7717 | 0.865 |
| gaussian kernel | cnn | no | fgsm_0.3 | 0.4486 | 0.416 |
| gaussian kernel | cnn | no | rfgsm | 0.5685 | 0.583 |
| gaussian kernel | cnn | no | cw2 | 0.7161 | 0.0 |
| gaussian kernel | cnn | no | bb_cnn | 0.7851 | 0.9415 |
| gaussian kernel | cnn | no | bb_mlp | 0.902 | 0.9679 |
| gaussian kernel | mlp | yes | none | 0.9649 | 0.9762 |
| gaussian kernel | mlp | yes | fgsm_0.15 | 0.9301 | 0.954 |
| gaussian kernel | mlp | yes | fgsm_0.3 | 0.8526 | 0.9085 |
| gaussian kernel | mlp | yes | rfgsm | 0.8841 | 0.9279 |
| gaussian kernel | mlp | yes | cw2 | 0.9567 | 0.9725 |
| gaussian kernel | mlp | yes | bb_cnn | 0.8628 | 0.9288 |
| gaussian kernel | mlp | yes | bb_mlp | 0.9111 | 0.9456 |
| gaussian kernel | mlp | no | none | 0.9151 | 0.9754 |
| gaussian kernel | mlp | no | fgsm_0.15 | 0.5225 | 0.2177 |
| gaussian kernel | mlp | no | fgsm_0.3 | 0.2294 | 0.0172 |
| gaussian kernel | mlp | no | rfgsm | 0.2998 | 0.0399 |
| gaussian kernel | mlp | no | cw2 | 0.8 | 0.0007 |
| gaussian kernel | mlp | no | bb_cnn | 0.3635 | 0.6089 |
| gaussian kernel | mlp | no | bb_mlp | 0.3647 | 0.6692 |
| autoencoder | cnn | yes | none | 0.8967 | 0.9918 |
| autoencoder | cnn | yes | fgsm_0.15 | 0.7952 | 0.9786 |
| autoencoder | cnn | yes | fgsm_0.3 | 0.5702 | 0.9491 |
| autoencoder | cnn | yes | rfgsm | 0.6517 | 0.9618 |
| autoencoder | cnn | yes | cw2 | 0.839 | 0.9816 |
| autoencoder | cnn | yes | bb_cnn | 0.5418 | 0.9768 |
| autoencoder | cnn | yes | bb_mlp | 0.6092 | 0.9829 |
| autoencoder | cnn | no | none | 0.8923 | 0.989 |
| autoencoder | cnn | no | fgsm_0.15 | 0.7826 | 0.865 |
| autoencoder | cnn | no | fgsm_0.3 | 0.5268 | 0.416 |
| autoencoder | cnn | no | rfgsm | 0.6167 | 0.583 |
| autoencoder | cnn | no | cw2 | 0.8326 | 0.0 |
| autoencoder | cnn | no | bb_cnn | 0.5045 | 0.9415 |
| autoencoder | cnn | no | bb_mlp | 0.5625 | 0.9679 |
| autoencoder | mlp | yes | none | 0.897 | 0.9762 |
| autoencoder | mlp | yes | fgsm_0.15 | 0.7636 | 0.954 |
| autoencoder | mlp | yes | fgsm_0.3 | 0.5268 | 0.9085 |
| autoencoder | mlp | yes | rfgsm | 0.6092 | 0.9279 |
| autoencoder | mlp | yes | cw2 | 0.8642 | 0.9725 |
| autoencoder | mlp | yes | bb_cnn | 0.5829 | 0.9288 |
| autoencoder | mlp | yes | bb_mlp | 0.632 | 0.9456 |
| autoencoder | mlp | no | none | 0.8929 | 0.9754 |
| autoencoder | mlp | no | fgsm_0.15 | 0.7405 | 0.2177 |
| autoencoder | mlp | no | fgsm_0.3 | 0.4628 | 0.0172 |
| autoencoder | mlp | no | rfgsm | 0.5531 | 0.0399 |
| autoencoder | mlp | no | cw2 | 0.8555 | 0.0007 |
| autoencoder | mlp | no | bb_cnn | 0.5474 | 0.6089 |
| autoencoder | mlp | no | bb_mlp | 0.6131 | 0.6692 |
| autoencoder->binarize | cnn | yes | none | 0.8854 | 0.9918 |
| autoencoder->binarize | cnn | yes | fgsm_0.15 | 0.7768 | 0.9786 |
| autoencoder->binarize | cnn | yes | fgsm_0.3 | 0.5613 | 0.9491 |
| autoencoder->binarize | cnn | yes | rfgsm | 0.6384 | 0.9618 |
| autoencoder->binarize | cnn | yes | cw2 | 0.8294 | 0.9816 |
| autoencoder->binarize | cnn | yes | bb_cnn | 0.5289 | 0.9768 |
| autoencoder->binarize | cnn | yes | bb_mlp | 0.6005 | 0.9829 |
| autoencoder->binarize | cnn | no | none | 0.8808 | 0.989 |
| autoencoder->binarize | cnn | no | fgsm_0.15 | 0.7703 | 0.865 |
| autoencoder->binarize | cnn | no | fgsm_0.3 | 0.5508 | 0.416 |
| autoencoder->binarize | cnn | no | rfgsm | 0.6289 | 0.583 |
| autoencoder->binarize | cnn | no | cw2 | 0.8242 | 0.0 |
| autoencoder->binarize | cnn | no | bb_cnn | 0.5213 | 0.9415 |
| autoencoder->binarize | cnn | no | bb_mlp | 0.587 | 0.9679 |
| autoencoder->binarize | mlp | yes | none | 0.8862 | 0.9762 |
| autoencoder->binarize | mlp | yes | fgsm_0.15 | 0.7535 | 0.954 |
| autoencoder->binarize | mlp | yes | fgsm_0.3 | 0.5275 | 0.9085 |
| autoencoder->binarize | mlp | yes | rfgsm | 0.6063 | 0.9279 |
| autoencoder->binarize | mlp | yes | cw2 | 0.852 | 0.9725 |
| autoencoder->binarize | mlp | yes | bb_cnn | 0.58 | 0.9288 |
| autoencoder->binarize | mlp | yes | bb_mlp | 0.6211 | 0.9456 |
| autoencoder->binarize | mlp | no | none | 0.8824 | 0.9754 |
| autoencoder->binarize | mlp | no | fgsm_0.15 | 0.7456 | 0.2177 |
| autoencoder->binarize | mlp | no | fgsm_0.3 | 0.524 | 0.0172 |
| autoencoder->binarize | mlp | no | rfgsm | 0.6007 | 0.0399 |
| autoencoder->binarize | mlp | no | cw2 | 0.8494 | 0.0007 |
| autoencoder->binarize | mlp | no | bb_cnn | 0.5708 | 0.6089 |
| autoencoder->binarize | mlp | no | bb_mlp | 0.6132 | 0.6692 |
| binarize->autoencoder | cnn | yes | none | 0.8895 | 0.9918 |
| binarize->autoencoder | cnn | yes | fgsm_0.15 | 0.8757 | 0.9786 |
| binarize->autoencoder | cnn | yes | fgsm_0.3 | 0.8463 | 0.9491 |
| binarize->autoencoder | cnn | yes | rfgsm | 0.8581 | 0.9618 |
| binarize->autoencoder | cnn | yes | cw2 | 0.868 | 0.9816 |
| binarize->autoencoder | cnn | yes | bb_cnn | 0.8342 | 0.9768 |
| binarize->autoencoder | cnn | yes | bb_mlp | 0.8614 | 0.9829 |
| binarize->autoencoder | cnn | no | none | 0.8872 | 0.989 |
| binarize->autoencoder | cnn | no | fgsm_0.15 | 0.8712 | 0.865 |
| binarize->autoencoder | cnn | no | fgsm_0.3 | 0.8438 | 0.416 |
| binarize->autoencoder | cnn | no | rfgsm | 0.8535 | 0.583 |
| binarize->autoencoder | cnn | no | cw2 | 0.8647 | 0.0 |
| binarize->autoencoder | cnn | no | bb_cnn | 0.8298 | 0.9415 |
| binarize->autoencoder | cnn | no | bb_mlp | 0.8545 | 0.9679 |
| binarize->autoencoder | mlp | yes | none | 0.8904 | 0.9762 |
| binarize->autoencoder | mlp | yes | fgsm_0.15 | 0.8762 | 0.954 |
| binarize->autoencoder | mlp | yes | fgsm_0.3 | 0.8507 | 0.9085 |
| binarize->autoencoder | mlp | yes | rfgsm | 0.86 | 0.9279 |
| binarize->autoencoder | mlp | yes | cw2 | 0.8825 | 0.9725 |
| binarize->autoencoder | mlp | yes | bb_cnn | 0.8493 | 0.9288 |
| binarize->autoencoder | mlp | yes | bb_mlp | 0.8579 | 0.9456 |
| binarize->autoencoder | mlp | no | none | 0.8875 | 0.9754 |
| binarize->autoencoder | mlp | no | fgsm_0.15 | 0.8722 | 0.2177 |
| binarize->autoencoder | mlp | no | fgsm_0.3 | 0.8435 | 0.0172 |
| binarize->autoencoder | mlp | no | rfgsm | 0.8557 | 0.0399 |
| binarize->autoencoder | mlp | no | cw2 | 0.8804 | 0.0007 |
| binarize->autoencoder | mlp | no | bb_cnn | 0.845 | 0.6089 |
| binarize->autoencoder | mlp | no | bb_mlp | 0.8544 | 0.6692 |
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
