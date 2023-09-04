FROM ubuntu:focal
LABEL Maintainer = "lequockhang"

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

WORKDIR /workdir

COPY . ./

RUN pip3 install -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python3", "Clustering.py" ]
