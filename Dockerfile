FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
    
RUN git -C /tmp clone https://github.com/NVIDIA/apex && \
    cd /tmp/apex && \
    git checkout b66ffc1 && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    rm -rf /tmp/apex

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

ENTRYPOINT ["sh", "-c", "./train.sh"]

