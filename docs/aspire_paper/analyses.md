# Running analyses

Code and ini files to produce the analyses in the paper can be found in the accompanying [GitHub repository]().
For the specific versions used in this paper, checkout [*version*]() or download it from this [DOI]().

## Injection studies

### Data

The exact data used in the paper is included in an additional data release. Alternatively, one
can regenerate the data using the `Makefile` in `gw/data/`.

### Running analyses

Analyses are configured and run using `bilby_pipe`.

Analyses using `aspire` require that the corresponding `dynesty` result already exist.

#### Eccentric analyses

Eccentric analyses are run directly using `bilby` via the Python script in `gw/injections/eccentric_TF2_example/`.

## P-P tests

The P-P tests were performed using built-in functionality in `bilby_pipe`. The necessary files are included in
`gw/pp_tests`, including the injection parameters and priors.

The tests are configured from `precessing_spins_pp_test.ini`. Before submitting the jobs, you should update
the ini file to use appropriate settings for the cluster you are using. See [`bilby_pipe` documentation](https://lscsoft.docs.ligo.org/bilby_pipe/master/index.html)
for details.

Once the ini file has been updated, submit the P-P tests using:

```bash
bilby_pipe precessing_spins_pp_test.ini --submit
```

## Real data

These analyses use data from GWOSC which can be downloaded automatically using `bilby_pipe`.