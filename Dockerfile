FROM ubuntu:focal
LABEL Maintainer="lequockhang"

RUN apt-get update -y \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip

WORKDIR /workdir/

RUN mkdir exec/
COPY requirements.txt /workdir/
COPY . /workdir/exec/

RUN pip3 install -r requirements.txt

EXPOSE 5001

ENTRYPOINT ["python3", "/workdir/exec/RNA3Dclust.py"]