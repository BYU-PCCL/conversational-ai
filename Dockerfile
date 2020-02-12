FROM tensorflow/tensorflow:1.15.2-gpu-py3    

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sanic>=19.0.0

COPY ./ ./

EXPOSE 6006
EXPOSE 8080

CMD ["sh", "-c", "tensorboard --logdir=/checkpoint & ./train.py"]

