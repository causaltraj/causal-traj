# CausalTraj
**This is a temporary repo for AAAI workshop submission. Official repo coming soon!**

CausalTraj is a causal likelihood model for coherent multi-agent trajectory generation.

Environment: Python 3.12, CUDA 12.8 \
Hardware used: Single A100 GPU

Command too train model on NBA dataset:
```
HYDRA_FULL_ERROR=1 python train.py --config-path configs/nba --config-name cpointnet_default.yaml
```