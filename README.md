# Reinforcement Learning - Examples and Tools

This repository contains the following examples:
* Simple DQN (Deep Q Network) from [pytorch official page](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* Stock price forecasting from https://github.com/notadamking/Stock-Trading-Environment
* Inventory control from [https://github.com/ikatsov/tensor-house/](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/supply-chain-reinforcement-learning.ipynb)


## Requirements
* Linux >= 16.04
* conda or docker

## Setup
#### Conda

Note, virtual display does not work with nvidia driver preinstalled on Azure DSVM. To use Azure DSVM with `conda`, one should uninstall the nvidia driver and reinstall with `--no-opengl-files` option.

```
# 1. ssh with x11 and jupyter-notebook port tunneling
ssh your-id@your-vm -X -L 8888:localhost:8888 

# 2. Install render requirements on the VM
sudo apt-get update
sudo apt-get install xvfb python-opengl ffmpeg

# 3. Setup env
conda env create -f environment.yml
conda activate relearn
python -m ipykernel install --user --name relearn

# 4. Start jupyter notebook
xvfb-run -s "-screen 0 640x480x24" jupyter notebook 
```

#### Docker
```
# pull the image
sudo docker pull loomlike/relearn:gpu

# start the container
sudo docker run -p 8888:8888 loomlike/relearn:gpu

# open http://localhost:8888 from your browser
```
To use GPU, install [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker) if haven't already done, and use `--gpus all` flag.
