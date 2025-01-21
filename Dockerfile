# Use a lightweight Python base image
FROM python:3.9-slim
LABEL Maintainer="quoc-khang.le@universite-paris-saclay.fr"

# Set the working directory in the container
WORKDIR /workdir/

RUN useradd -ms /bin/bash dockeruser

# Switch to the non-root user
USER dockeruser

# Install dependencies
COPY requirements.txt /workdir/

COPY . /workdir/

RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to run the program as a module
ENTRYPOINT ["python3", "RNA3Dclust.py"]
