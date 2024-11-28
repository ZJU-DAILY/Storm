# Storm
This is a PyTorch implementation of the paper: Towards Online Spatio-Temporal Data Prediction: A Knowledge Distillation Driven Continual Learning Approach (Storm)

![](C:\Users\Nights\Desktop\素材\framework_00(1).png)


## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Preparation

1. PEMS-BAY: A network representation of 325 traffic sensors in the Bay Area, collected by the California Department of Transportation (CalTrans) Measurement System (PeMS), displaying traffic flow data at 5-minute intervals from January 2017 to May 2017.
2. PEMSD4：The traffic flow datasets are collected by the Caltrans Performance Measurement System ([PeMS](http://pems.dot.ca.gov/)) ([Chen et al., 2001](https://trrjournalonline.trb.org/doi/10.3141/1748-12)) in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from the raw data. 

```

# Create data directories, For example
mkdir -p data/{PEMSD4}

Download PEMSD4 dataset then unzip in PEMSD4

```

## Model Training

* STGCN

```
cd STGCN 

python STGCN_Main.py

```

* AGCRN

```
cd AGCRN 

python AGCRN_Main.py

```

* MTGNN

```
cd MTGNN 

python MTGNN_Main.py

```

* STAEformer

```
cd STAEformer 

python STGCN_Main.py

cd model/

python train.py -d <dataset> -g <gpu_id>
```

Our research mainly refers to the following works:

[1] STG4Traffic：https://github.com/trainingl/STG4Traffic

[2] STAEformer https://github.com/XDZhelheim/STAEformer

