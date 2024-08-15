FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime


COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /src

WORKDIR /src

CMD ["python", "ml/server.py"]
