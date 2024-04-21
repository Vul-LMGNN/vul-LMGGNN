# Vul-LMGGNN
Code for the paper - Source Code Vulnerability Detection: Combining Code Language Models and Code Property Graph
## Introduction

In this work, we propose Vul-LMGNN, a unified model that combines pre-trained code language models with code property graphs for code vulnerability detection. Vul-LMGNN constructs a code property graph, thereafter leveraging pre-trained code model to extract local semantic features as node embeddings in the code property graph. Furthermore, we introduce a gated code Graph Neural Network (GNN). By jointly training the code language model and the gated code GNN modules in Vul-LMGNN, our proposed method efficiently leverages the strengths of both mechanisms. Finally, we use a pre-trained CodeBERT as an auxiliary classifier. The proposed method demonstrated superior performance compared to six state-of-the-art approaches.

## Getting Started 

Create environment and install required packages for LMGGNN

### Install packages

- [Joern](https://joern.io/docs/)
- [Python (=>3.8)](https://www.python.org/)

- [Pandas (>=1.0.1)](https://pandas.pydata.org/)
- [scikit-learn (>=0.22.2)](https://scikit-learn.org/stable/)
- [PyTorch (=2.2.0)](https://pytorch.org/)
- [PyTorch Geometric (>=1.4.2)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Gensim (>=3.8.1)](https://radimrehurek.com/gensim/)
- [cpgclientlib (>=0.11.111)](https://pypi.org/project/cpgclientlib/)
- transformer(=3.3.1)

The experiments were executed on single NVIDIA A100 80GB GPU. The system specifications comprised NVIDIA driver version 525.85.12 and CUDA version 11.8.

## Dataset

We evaluated the performance of our model using four publicly available datasets. The composition of the datasets is as follows, and you can click on the dataset names to download them. Please note that you need to modify the code in the `CPG_generation` function in `run.py` to adapt to different dataset formats.

| *Dataset*                                                    | *#Vulnerable* | *#Non-Vulnerable* | *Source*       |
| ------------------------------------------------------------ | ------------- | ----------------- | -------------- |
| [DiverseVul](https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view?usp=sharing) | 18,945        | 330,492           | Snyk,Bugzilla  |
| [Devign](https://sites.google.com/view/devign)               | 11,888        | 14,149            | Github         |
| [VDSIC](https://osf.io/d45bw/)                               | 82,411        | 119,1955          | Github, Debian |
| [ReVeal](https://github.com/VulDetProject/ReVeal)            | 1664          | 16,505            | Chrome, Debian |

## Usage

##### Some tips:

- **Modifications** to the `configs.json` structure **should be updated** in the `configs.py` script.
- Joern processing may be slow or potentially freeze your OS, depending on your system’s specs. To prevent this, **reduce the chunk size** processed during the **CPG_generation** process by adjusting the `"slice_size"` value in the `"create"` section of the `configs.json` file.
- Within the `"slice_size"` parameter, nodes exceeding the configured size limit **will be filtered out and discarded**.
- Follow the instructions on [Joern's documentation page](https://joern.io/docs/) and install Joern's command line tools under `'project'\joern\joern-cli\ `.

##### **Preparing the CPG :**

```
python run.py -cpg -embed -mode train -path /your/model/path
```

`-cpg` and `-embed` respectively represent using `joern` to extract the code's `CPG` and generating corresponding embeddings. `-path` is used to specify the path for saving the model.

##### Training and Testing:

```
python run.py -mode test -path /your/model/saved/path
```

`-mode` is used to specify whether only the training process is executed or both the training and testing processes are performed.  `-path` is used to specify the path for saving the model.

##### Fine-tuning process:

This command is used to fine-tune CodeBERT on a specific dataset and then generate embeddings for subsequent nodes. Pre-trained CodeBERT weights need to be downloaded from [here](https://huggingface.co/microsoft/codebert-base).

```
python fine-tune.py
```

## Main Results

Here only the accuracy results are displayed; for other metrics, please refer to the paper.

| *Model*                                             | *DiverseVul* | *VDSIC*   | *Devign*  | *ReVeal*  |
| --------------------------------------------------- | ------------ | --------- | --------- | --------- |
| *[BERT](https://arxiv.org/abs/1810.04805)*          | 91.99        | 79.41     | 60.58     | 86.88     |
| *[CodeBERT](https://arxiv.org/abs/2002.08155)*      | 92.40        | 83.13     | 64.80     | 88.64     |
| *[GraphCodeBERT](https://arxiv.org/abs/2009.08366)* | 92.96        | 83.98     | 64.80     | 89.25     |
| *TextCNN*                                           | 92.16        | 66.54     | 60.38     | 85.43     |
| *[TextGCN](https://arxiv.org/abs/1809.05679)*       | 91.50        | 67.55     | 60.47     | 87.25     |
| *[Devign](https://arxiv.org/abs/1909.03496)*        | 70.21        | 59.30     | 57.66     | 65.47     |
| ***Our***                                           | **93.06**    | **84.38** | **65.70** | **90.80** |

## Acknowledgement

Parts of the code for data preprocessing and graph construction using `Joern` are adapted from *[Devign](https://arxiv.org/abs/1909.03496)*. We appreciate their excellent work!

