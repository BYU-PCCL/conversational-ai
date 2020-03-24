FROM tensorflow/tensorflow:1.15.2-gpu-py3    

WORKDIR /workspace/

COPY requirements.txt ./

# pip & tensorflow are stupid
RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y tensorflow tensorflow-gpu && \
    pip install --no-cache-dir "$(grep tensorflow-gpu requirements.txt)" && \
    pip install --no-cache-dir 'sanic==19.*'

COPY ["*.tsv", "*.txt", "./"]

COPY ["*.sh", "*.py", "./"]

EXPOSE 6006
EXPOSE 8080

CMD tensorboard --logdir=${CONVERSATIONAL_AI_MODEL_DIR:-/models/} -v -2 & python3 model.py
