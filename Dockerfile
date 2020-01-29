FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout b66ffc1 && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY ./ ./

CMD ["./train.sh"]

