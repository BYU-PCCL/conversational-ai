FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh run_lm_finetuning.py train.txt ./

CMD ["./entrypoint.sh"]

