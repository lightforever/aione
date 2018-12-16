FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev \
    libav-tools \
    libfontconfig1-dev \
    libopencv-dev python-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy tensorflow opencv-python gym PyVirtualDisplay box2d-py

WORKDIR /pytorch_car_caring

COPY . .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python test.py