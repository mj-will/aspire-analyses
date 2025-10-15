from bilby.gw.conversion import generate_all_bbh_parameters as bilby_generate_all_bbh_parameters
import numpy as np


def generate_all_bbh_parameters(sample, likelihood=None, priors=None, npool=1):
    """Wrapper around bilby's generate_all_bbh_parameters to allow for npool argument."""
    output_sample = bilby_generate_all_bbh_parameters(sample, likelihood, priors, npool=npool)
    if "delta_phase" not in output_sample:
        output_sample["delta_phase"] = np.mod(
            output_sample["phase"] + np.sign(np.cos(output_sample["theta_jn"]) * output_sample["psi"]),
            2 * np.pi,
        )
    return output_sample
