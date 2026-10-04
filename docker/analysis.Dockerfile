FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/opt/coinjoin-analysis/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        fontconfig \
        fonts-dejavu-core \
        git \
        graphviz \
        libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/coinjoin-analysis

COPY requirements.txt setup.py ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e . \
    && chmod +x /opt/coinjoin-analysis/docker/analysis-entrypoint.sh

ENTRYPOINT ["/opt/coinjoin-analysis/docker/analysis-entrypoint.sh"]
CMD ["help"]
