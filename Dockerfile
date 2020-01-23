FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip --no-cache-dir install \
        'numpy>=1' \
        'pynvml>=8' \
        'tensorboard>=2' \
        'torch>=1' \
        'tqdm>=4'

RUN git clone "https://github.com/huggingface/transformers.git" && \
    cd transformers && \
    git checkout 90b7df4 && \
    pip install .

WORKDIR ./transformers/examples

COPY train.txt entrypoint.sh ./

CMD ["./entrypoint.sh"]


