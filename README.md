# Subtype-DCGCN: an unsupervised approach for cancer subtype diagnosis based on multi-omics data
This repository contains the code for the Subtype-DCGCN method, which is an unsupervised approach for cancer subtype diagnosis based on multi-omics data. Our approach can be divided into three steps. Firstly, the dual contrastive learning module guides the graph convolutional neural network to extract the effective low-dimensional representation of each omics. Secondly, we apply the mean fusion strategy and the decoders to obtain a final representation of each sample. Finally, K-means is applied to the final representation to identify cancer subtypes.

## Quick start
Subtype-DCGCN is based on the Python program language. The network's implementation was based on the open-source library Pytorch 1.7.0.
We used the NVIDIA RTX 2080 Ti for the model training. It is recommended to use the conda command to configure the environment:
```
# create an environment for running
conda env create -f dcgcn.yml

# activate environment
conda activate dcgcn

# We can use the following command to finish the subtyping process: 
python main.py -t UVM -i ./input/UVM.list -m training

# We can use the following command to cluster the extracted low-dimensional features: 
python main.py -t UVM -i ./input/UVM.list -m clustering

```
## Data
The datasets used in this study are available at: **https://github.com/haiyang1986/Subtype-GAN**

## Note
You can choose whether to conduct weakly paired datasets training in the main.py. Additionally, you can set the missing rate for weakly paired datasets in the train.py.

## Update
In the latest update, we added many comments to aid in reading. 
