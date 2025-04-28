# AreaDisasterRecognition

A deep learning pipeline for aerial disaster image classification using PyTorch, React, and Azure.

# Installation

## Clone the repo
```
git clone https://github.com/mahmed50/AreaDisasterRecognition.git
cd ader
```

## Download and split the datasets
```
sh get_datasets.sh
```

rename the downloaded test folder to "val", respectively
```
cd data/aider
mv test val
```

## Set up virtual environment
Chances are, you won't be able to run python commands unless you set up a virtual environment.
```
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
```
## Install dependencies
```
pip install torch torchvision tqdm
```
## Single-Node Training
```
python src/train_model.py
```
Outputs:

- Model weights: models/aider_resnet18.pt

- Metrics: metrics/metrics_single.json
  
## Distributed (DDP) Training (Multi-VM)

1. SSH into both VMs.

2. Set environment variables for DDP:
```
export MASTER_ADDR=<IP_or_hostname_of_rank0>
export MASTER_PORT=29500
```

##Launch training:

On VM1 (rank 0):
```
python src/train_ddp.py --rank 0 --world_size 2
```
On VM2 (rank 2):
```
python src/train_ddp_unified.py --rank 1 --world_size 2
```
Outputs:

- Model weights: models/ader_ddp_cpu.pt

- Metrics: models/metrics_ddp.json

##Test a saved model
You can test using any image downloaded from online. Just point the model to the image using the --image flag.

You can test the smart model or the dumb model. The only difference is the sample sizes they were trained on (full sample vs 5,000 images).

```
python src/demo/smart_model.py --image demo/flood.jpg
```

