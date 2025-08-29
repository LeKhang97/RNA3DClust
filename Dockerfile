# Use a lightweight Python base image
FROM python:3.9-slim
LABEL Maintainer="quoc-khang.le@universite-paris-saclay.fr"

# Set working directory
WORKDIR /workdir/

RUN useradd -ms /bin/bash dockeruser

# Switch to the non-root user
USER dockeruser

# Install dependencies
COPY requirements.txt /workdir/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workdir/

# Set entry point
ENTRYPOINT ["python3","RNA3Dclust.py"]
