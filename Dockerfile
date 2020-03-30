FROM tensorflow/tensorflow:1.15.2-gpu-py3    

WORKDIR /workspace/

COPY requirements.txt ./

# pip & tensorflow are stupid
RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y tensorflow tensorflow-gpu && \
    pip install --no-cache-dir 'tensorflow-gpu>=1.15.0,<2.0.0' && \
    pip install --no-cache-dir 'sanic==19.*'

COPY ["*.tsv", "*.txt", "./"]

COPY ["*.sh", "*.gin", "*.py", "./"]

EXPOSE 8080

CMD ["python3", "models.py"]
