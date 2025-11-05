# aspire-analyses

Accelerated Sequential Posterior Inference via Reuse (aspire) analyses and supporting tools for gravitational-wave inference.

## Overview

- Reproducible workflows, notebooks, and configuration files used in the *Accelerated Sequential Posterior Inference via Reuse for Gravitational-Wave Analyses* study.
- Python utilities in `aspire_analysis_tools` for comparing and post-processing `bilby`/`aspire` results.
- Companion Jupyter Book documentation covering data releases, figures, and toy examples.

## Repository layout
- `src/aspire_analysis_tools`: Python package and CLI helpers (installs `aat_plot_comparison`).
- `gw/`: `bilby` and `aspire` configuration files, injection studies, and plotting assets for the paper.
- `docs/`: Jupyter Book; `make -C docs build` regenerates the site and syncs notebooks.
- `toy_examples/`: Toy examples with analytic/toy likelihoods.
- `data_releases/`: Makefiles and support scripts for packaging published artefacts.

## Getting started
- Requires Python 3.11+, `conda` (or `mamba`), and the dependencies listed in `environment.yml`.
- Create the recommended environment and install the utilities in editable mode:

```bash
conda env create -f environment.yml
conda activate aspire
pip install -e .
```

## Reproducing documentation and figures
- Build the documentation site (copies curated notebooks and runs Jupyter Book):

```bash
make -C docs build
```

- Open `docs/_build/html/index.html` in a browser for guidance on data releases, analysis recipes, and figures from the paper.
- Generated figures and statistics notebooks for the first paper live under `gw/plots/first_paper/`.

## Command-line utilities
- `aat_plot_comparison` overlays corner plots from one or more `bilby` result files:

```bash
aat_plot_comparison \
  --results result_dynesty.json result_aspire.json \
  --labels "Dynesty" "Aspire" \
  --filename comparison.png
```

## Development

- Run `pytest` from the repository root to execute the available unit tests.
- Use `make -C docs clean` before rebuilding documentation if you need to clear cached outputs.

## Citation

- Cite aspire via [Zenodo DOI 10.5281/zenodo.15658747](https://doi.org/10.5281/zenodo.15658747).
- DOIs for code and result releases are listed in the documentation.

## Acknowledgements

- Numerical computations were carried out on the SCIAMA High Performance Compute (HPC) cluster, supported by the ICG and the University of Portsmouth.
