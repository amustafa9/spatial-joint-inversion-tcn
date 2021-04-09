The repository contains codes for our work titled "Spatiotemporal Modeling of Seismic Images for Acoustic Impedance Estimation" accepted and presented at the SEG 2020 Annual Meeting and published to SEG 2020 Expanded Abstracts. The preprint for the work may be obtained [here](https://arxiv.org/pdf/2006.15472.pdf). 

The complete bibtex citation for the work is as follows:

```
@inbook{doi:10.1190/segam2020-3428298.1,
author = {Ahmad Mustafa and Motaz Alfarraj and Ghassan AlRegib},
title = {Spatiotemporal modeling of seismic images for acoustic impedance estimation},
booktitle = {SEG Technical Program Expanded Abstracts 2020},
chapter = {},
pages = {1735-1739},
year = {2020},
doi = {10.1190/segam2020-3428298.1},
URL = {https://library.seg.org/doi/abs/10.1190/segam2020-3428298.1},
}

```

The repository contains all the data used to run the codes. Clone the github repo to an appropriate directory on your machine. You may use a dedicated IDE like Pycharm or Spyder to run the python files for each machine learning model used in the paper, namely the 2-D TCN, 1-D TCN, and one-layer LSTM. Running the training file for each model will extract the data, perform training, and then use the trained models to infer on the SEAM section to print out the estimated 2-D profiles as well as various regression metrics between the estimated and ground-truth impedance. Alternatively, you may use the command line to to run codes and print results, as shown in the example for 1-D TCN below:

```
cd <project root directory>
python train-1D-TCN.py --no_wells 12 --epochs 900 
```

In case you face problems with running the codes, please contact Ahmad Mustafa at amustafa9@gatech.edu.


