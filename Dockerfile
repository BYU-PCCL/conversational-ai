FROM tensorflow/tensorflow:1.15.2-gpu-py3    

COPY requirements.txt ./

# pip & tensorflow are stupid
RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall tensorflow tensorflow-gpu -y && \
    pip install --no-cache-dir 'tensorflow-gpu>=1.15.0,<2.0.0' && \
    pip install --no-cache-dir sanic>=19.0.0

COPY ["*.tsv", "*.txt", "./"]

COPY ["*.sh", "*.py", "./"]

EXPOSE 6006
EXPOSE 8080

CMD ["sh", "-c", "tensorboard --logdir=/models & python3 train.py"]
