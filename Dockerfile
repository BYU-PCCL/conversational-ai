FROM tensorflow/tensorflow:2.1.0-gpu-py3

WORKDIR /workspace/

COPY requirements.txt ./

RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY conversational_ai/ ./conversational_ai/

EXPOSE 8080

ENTRYPOINT ["python3", "-m"]
CMD ["conversational_ai.t5_model"]
