FROM tensorflow/tensorflow:1.15.2-gpu-py3    

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

EXPOSE 6006

CMD ["sh", "-c", "tensorboard --logdir=/checkpoint --bind_all & ./train.py"]

