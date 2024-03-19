# pepMTL

![pepMTL](https://github.com/GuoLab-YunLi/pepMTL/assets/156044382/d36f38c7-7784-4dbf-b9a4-3874c673a922)

Unleashing Unmatched Predictive Capabilities: pepMTL, an unprecedented multi-property predictor, harnesses the synergistic might of a Multi-Task Architecture and a Pre-Trained Protein Language Model to empower concurrent and precise prediction of CCS, RT, and MS/MS data for peptides.

This tutorial will guide you through the process of training the pepMTL model, a multi-task learning model for the prediction of RT, CCS, and MS/MS of peptides. The model is built using the pre-trained ESM-2 language model as the backbone and is trained on a dataset of peptide sequences and their corresponding RT, CCS, and MS/MS data.

# Prerequisites
Before running the code, make sure you have the following prerequisites installed:
1.	Python (version 3.11.0)
2.	NVIDIA GPU (the code is designed to run on a system with an NVIDIA GPU)
3.	Miniconda (or Anaconda)
4.	Required Python packages: PyTorch, Transformers, Datasets, NumPy, Pandas, SciPy, Scikit-learn, Tqdm, and ms_entropy

# Setup
Specific version of the python environment and toolbox listed in requirements.txt has been verified to be available.
1.	Install Miniconda: If you haven’t already, install Miniconda on your system.
2.	Create a Python environment: Open the Anaconda Prompt and create a new Python environment named "pepMTL" with Python version 3.11.0 using the following command:
```
conda create -n pepMTL python==3.11.0
```
3.	Activate the environment: Activate the "pepMTL" environment using the following command:
```
conda activate pepMTL
```
4.	Download the pepMTL package and extract it to your desktop. Navigate to the extracted folder using the following command:
```
cd C:/Users%USERNAME%/Desktop/pepMTL
```
5.	Install PyTorch: Install the required version of PyTorch with CUDA support using the following command:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
6.	Install dependencies: Install the remaining dependencies using the provided requirements.txt file:
c
pip install -r requirements.txt
```
7.	Open Spyder: Open the Spyder IDE by running the following command:
```
spyder
```












