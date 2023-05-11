FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update && apt install -y &&\
    apt install -y --no-install-recommends gcc vim htop python3-pip libpython3.10-dev

COPY requirements.txt requirements.txt
RUN pip3 install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt ipython

ENV NCCL_SOCKET_IFNAME=lo \
    NCCL_MIN_NCHANNELS=32 \
    NCCL_SOCKET_NTHREADS=4 \
    NCCL_NSOCKS_PERTHREAD=16

WORKDIR /didi

ENTRYPOINT ["bash"]
