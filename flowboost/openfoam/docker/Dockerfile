FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        software-properties-common \
        ca-certificates && \
    wget -q -O /etc/apt/trusted.gpg.d/openfoam.asc https://dl.openfoam.org/gpg.key && \
    add-apt-repository -y http://dl.openfoam.org/ubuntu && \
    apt-get update && \
    apt-get install -y --no-install-recommends openfoam13 file && \
    apt-get purge -y wget software-properties-common && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Source OpenFOAM environment by default
RUN echo "source /opt/openfoam13/etc/bashrc" >> /etc/bash.bashrc
ENV BASH_ENV=/opt/openfoam13/etc/bashrc

WORKDIR /work
