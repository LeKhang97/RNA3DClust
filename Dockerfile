FROM ubuntu:focal
LABEL Maintainer="lequockhang"

RUN apt-get update -y \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip

WORKDIR /workdir/

#RUN mkdir exec/

COPY requirements.txt /workdir/
#COPY . /workdir/exec/
COPY . /workdir/

RUN pip3 install -r requirements.txt

EXPOSE 5001

#ENTRYPOINT ["python3", "exec/RNA3Dclust.py"]

ENTRYPOINT ["python3", "RNA3Dclust.py"]