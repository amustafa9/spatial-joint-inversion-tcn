The repository contains codes for our paper titled "Joint Learning for Spatial Context-based Seismic Inversion of Multiple Datasets for Improved Generalizability and Robustness" accepted to Geophysics. The preprint for the work may be obtained [here](https://arxiv.org/abs/2104.02750). 

The complete bibtex citation for the work is as follows:

```
@article{MustafaJointLearningwithSpatialContext,
author = {Ahmad Mustafa and Motaz Alfarraj and Ghassan AlRegib},
title = {Joint Learning for Spatial Context-based Seismic Inversion of Multiple Datasets for Improved Generalizability and Robustness},
journal = {Geophysics},
volume = {86},
issue = {4},
pages = {},
year = {2021},
doi = {},

URL = {},
eprint = {}
,
    abstract = {}
}

```

The repository contains all the data used to run the codes. Clone the github repo to an appropriate directory on your machine. You may use a dedicated IDE like Pycharm or Spyder to run the python files for each of the various learning methods listed in Table 2 of the paper. Running the training file for each model will extract the data, perform training, and then use the trained models to infer on the SEAM and Marmousi 2 sections to print out the estimated 2-D profiles as well as various regression metrics between the estimated and ground-truth impedances for both datasets. Alternatively, you may use the command line to to run codes and print results, as shown in the example for 1-D TCN below:

```
cd <project root directory>
python train-1D-TCN.py --no_wells 12 --epochs 900 --data_flag <marmousi or seam>
```
For the transfer learning based methods, namely joint learning, combined learning, and pretraining-finetuning, there are additional parameters to specify the number of wells to be sampled from each dataset. An example for running the joint learning method for the default settings described in the paper is shown below:

```
cd <project root directory>
python train-joint-learning.py --no_wells_seam 12 --no_wells_marmousi 50 --epochs 900 --gamma 0.0001
```

In case you face problems with running the codes, please contact Ahmad Mustafa at amustafa9@gatech.edu.


