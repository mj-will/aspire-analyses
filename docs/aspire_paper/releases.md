# Data releases

The paper is accompanied by three data releases:

- Code release (DOI):
- Results release (DOI): https://doi.org/10.5281/zenodo.17514969

Each of these can be downloaded from the corresponding DOI or using
code provided in the code release (or [repository](https://github.com/mj-will/aspire-analyses))

```{note}
The paper used v1.0 of the code repository.
```

## Code release

The code release mirrors the GitHub Repository structure:

```bash
├── data_releases
│   ├── config.mk
│   ├── fetch_results_from_yaml.py
│   ├── first_paper_data_release
│   ├── first_paper_data_release.zip
│   ├── Makefile
│   ├── README.md
│   └── results_first_paper.yaml
├── docs
│   ├── aspire_paper
│   ├── _build
│   ├── _config.yml
│   ├── intro.md
│   ├── Makefile
│   ├── _toc.yml
│   └── toy_examples
├── environment.yml
├── gw
│   ├── data
│   ├── injections
│   ├── plots
│   ├── pp_tests
│   └── real_data
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   ├── aspire_analysis_tools
│   └── aspire_analysis_tools.egg-info
└── toy_examples
    ├── biased_initial_samples.ipynb
    └── outdir
```

All code for running analyses described to this paper is contained in
the `gw` directory, including plotting scripts.


## Results release

The results release includes three files:

- `core_results.zip`: core results for reproducing the figures
- `data.zip`: simulated data files for performing injection analyses
- `additional_results.zip`: additional result files, including the full set of P-P test result files

### Core results


Core results contains the following files:

```bash
core_results
├── pp_test_results
│   └── pp_test_credible_levels.hdf5
└── single_event_analyses
    ├── eccentric_TaylorF2_dynesty_result.hdf5
    ├── eccentric_TaylorF2Ecc_aspire_result.hdf5
    ├── eccentric_TaylorF2Ecc_dynesty_result.hdf5
    ├── GW150914_IMRPhenomXO4a_aspire_result.hdf5
    ├── GW150914_IMRPhenomXO4a_dynesty_result.hdf5
    ├── GW150914_IMRPhenomXPHM_dynesty_result.hdf5
    ├── GW150914_like_IMRPhenomD_dynesty_result.hdf5
    ├── GW150914_like_IMRPhenomPv2_aspire from IMRPhenomD_result.hdf5
    ├── GW150914_like_IMRPhenomPv2_dynesty_result.hdf5
    ├── GW150914_like_IMRPhenomXO4a_aspire from IMRPhenomXPHM_result.hdf5
    ├── GW150914_like_IMRPhenomXO4a_dynesty_result.hdf5
    ├── GW150914_like_IMRPhenomXPHM_dynesty_result.hdf5
    ├── q4_IMRPhenomXO4a_aspire from IMRPhenomXPHM_result.hdf5
    ├── q4_IMRPhenomXO4a_dynesty_result.hdf5
    └── q4_IMRPhenomXPHM_dynesty_result.hdf5
```

`single_event_analyses` contain the main results labelled based on the analysis, waveform and sampler used.
`pp_test_results` contains the confidence intervals need to recreate the P-P plot. The full results are in
the additional results file.


## Additional results

Additional results has the following structure:

```bash
additional_results
└── pp_test_results
    ├── aspire_pp_test_data0_0_analysis_H1L1V1_result.hdf5
    ├── aspire_pp_test_data10_0_analysis_H1L1V1_result.hdf5
    ├── aspire_pp_test_data1_0_analysis_H1L1V1_result.hdf5
    ├── aspire_pp_test_data11_0_analysis_H1L1V1_result.hdf5
    ├── aspire_pp_test_data12_0_analysis_H1L1V1_result.hdf5
    ...
    ├── aspire_pp_test_data98_0_analysis_H1L1V1_result.hdf5
    └── aspire_pp_test_data99_0_analysis_H1L1V1_result.hdf
```

## Data

Data contains the injection parameters and corresponding frame files

```bash
data
├── frame_files
│   ├── GW150914_XPHM_data
│   │   ├── injection_0_H1_1364342474_1364346570.hdf5
│   │   ├── injection_0_L1_1364342474_1364346570.hdf5
│   │   └── injection_0_V1_1364342474_1364346570.hdf5
│   └── HM_injection_q4_IMRPhenomXPHM_data
│       ├── injection_0_H1_1364342474_1364346570.hdf5
│       ├── injection_0_L1_1364342474_1364346570.hdf5
│       └── injection_0_V1_1364342474_1364346570.hdf5
└── injection_parameters
    ├── GW150914_parameters.hdf5
    └── HM_injection_q4_parameters.hdf5
```

```{note}
Due to constraints in TaylorF2(Ecc) analysis, this was running directly using and
frames were not produced. See the code release for more details.
```