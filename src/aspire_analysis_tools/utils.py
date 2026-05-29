from aspire.samples import MCMCSamples


def read_make_config(filename: str) -> dict:
    """Read a simple key=value config file, ignoring comments and blank lines."""
    config = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


def compute_rhat(samples: MCMCSamples) -> float:
    """Compute the R-hat convergence diagnostic for MCMC samples.

    Returns
    -------
    rhat : np.ndarray
        The R-hat statistic for the MCMC samples per parameter
    """
    import arviz as az

    arviz_data = az.from_dict({"posterior": samples.chain.transpose(1, 0, 2)})
    rhat = az.rhat(arviz_data)
    return rhat["posterior"].values
