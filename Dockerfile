FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y python3.10 \
    && apt-get install -y python3-pip \
    && apt-get install -y wget \
    && apt-get install -y unzip \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

ARG APPUSER

ENV APPUSER=$APPUSER

RUN useradd -m $APPUSER && echo "$APPUSER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/$APPUSER/app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

ENV PYTHONPATH=/home/$APPUSER/app:/home/$APPUSER/app/dependencies

USER $APPUSER

CMD ["bash"]

