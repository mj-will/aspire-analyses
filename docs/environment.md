# Environment

The repository contains environment file that can be used with `conda` (or `mamba`)
to create the necessary environment:

```bash
conda env create -f environment.yml
conda activate aspire
pip install .    # Install aspire-analysis-tools
```

We also provide a [docker image](https://github.com/mj-will/aspire-analyses/pkgs/container/aspire-analyses%2Fdocs-container) used for building the documentation.
