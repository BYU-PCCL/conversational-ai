FROM tensorflow/tensorflow:1.15.2-gpu-py3    

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

CMD ["./train.py"]

