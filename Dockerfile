FROM ubuntu:focal
LABEL Maintainer = "lequockhang"

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

WORKDIR  /workdir/

COPY . /exec

RUN pip3 install -r /exec/requirements.txt

ENTRYPOINT [ "python3", "/exec/Clustering.py" ]

