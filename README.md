# Reinforcement Learning - Examples and Tools

This repository contains the following examples:
* DQN (Deep Q Network)

## Requirements
* conda env w/ python 3.6
* pytorch 1.4 (cuda 10.1) and torchvision
* openai gym

## Setup
```
conda env create -f environment.yml
conda activate relearn
python -m ipykernel install --user --name relearn
```

## Run from a remote VM

> Note, virtual display does not work with nvidia driver preinstalled on Azure DSVM. One should uninstall the driver and reinstall with `--no-opengl-files` option.

```
ssh your-vm -X 

# Install render requirements
sudo apt install xvfb
#sudo apt install python-opengl
#sudo apt install ffmpeg

xvfb-run -s "-screen 0 640x480x24" jupyter notebook 
```





