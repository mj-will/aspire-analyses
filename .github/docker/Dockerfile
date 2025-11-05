FROM continuumio/miniconda3:latest

ARG CONDA_ENV=aspire
ENV CONDA_ENV=${CONDA_ENV}
ENV PATH="/opt/conda/envs/${CONDA_ENV}/bin:${PATH}"

# Install TeX Live and other system packages required for docs builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-xetex \
        dvipng \
        cm-super \
        curl \
        unzip \
        make \
        git \
        git-lfs \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and create the conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm -f /tmp/environment.yml

WORKDIR /workspace

# Copy the project in so we can install it into the environment
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src

# setuptools_scm needs a version when building without the git metadata
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+docker
RUN conda run -n ${CONDA_ENV} pip install --no-deps --editable .
ENV SETUPTOOLS_SCM_PRETEND_VERSION=
