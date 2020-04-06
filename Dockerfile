FROM tensorflow/tensorflow:2.1.0-gpu-py3

WORKDIR /workspace/

COPY requirements.txt ./

RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY ["*.tsv", "*.txt", "./"]

COPY ["*.sh", "*.gin", "*.py", "./"]

EXPOSE 8080

CMD ["python3", "t5_model.py"]
