FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev \
    libav-tools \
    libfontconfig1-dev && \
    rm -rf /var/lib/apt/lists/*
 
RUN conda install numpy pyyaml scipy ipython mkl mkl-include&& conda clean -ya
RUN pip install tensorflow opencv-python

WORKDIR /pytorch_car_caring

COPY . .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python test.py