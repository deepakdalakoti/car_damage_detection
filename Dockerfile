FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y python3.10 \
    && apt-get install -y python3-pip \
    && apt-get install -y wget \
    && apt-get install -y unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt


RUN apt-get update && apt-get install -y git

RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN export PYTHONPATH=/app:/app/dependencies

CMD tail -f /dev/null

