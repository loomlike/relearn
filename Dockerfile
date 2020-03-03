ARG HOME="/root"

FROM nvidia/cuda:10.1-base-ubuntu18.04

#ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
        apt-utils curl vim git wget build-essential software-properties-common \
        xvfb ffmpeg python-opengl python-pyglet freeglut3-dev mesa-utils

ARG HOME
ENV HOME="${HOME}"
WORKDIR ${HOME}

# Install Anaconda
ARG ANACONDA="https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh"
RUN curl ${ANACONDA} -o anaconda.sh && \
    /bin/bash anaconda.sh -b -p conda && \
    rm anaconda.sh && \
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH="${HOME}/conda/bin:${PATH}"

# Setup Jupyter notebook configuration
ENV NOTEBOOK_CONFIG="${HOME}/.jupyter/jupyter_notebook_config.py"
RUN mkdir ${HOME}/.jupyter && \
    echo "c.NotebookApp.token = ''" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.allow_root = True" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >> ${NOTEBOOK_CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >> ${NOTEBOOK_CONFIG}

# Clone repo
RUN git clone --depth 1 --single-branch -b master https://github.com/loomlike/relearn

# Install environment
RUN conda update -n base -c defaults conda && \
    conda env update -f relearn/environment.yml -n base && \
    conda clean -fay && \
    python -m ipykernel install --user --name 'python3' --display-name 'python3'

# Nvidia thing...
RUN ldconfig

WORKDIR ${HOME}/relearn

# Jupyter Notebook
EXPOSE 8888

CMD xvfb-run -s "-screen 0 1400x900x24" jupyter notebook
