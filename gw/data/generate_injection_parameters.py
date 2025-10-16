"""Generate injection parameters from a prior file.

Computes the optimal SNR in each detector and the network SNR for each injection.

Based on code originally written by Colm Talbot.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import trange

import bilby


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior-file",
        type=Path,
        help="Path to the prior file.",
    )
    parser.add_argument(
        "--injection-file",
        type=Path,
        help="Path to the injection file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--waveform",
        type=str,
        default="IMRPhenomXPHM",
        help="Duration of the data.",
    )
    parser.add_argument(
        "--psd-path",
        type=Path,
        default="psds",
        help="Path to the PSD files.",
    )
    parser.add_argument(
        "--reference-frequency",
        type=float,
        default=20,
        help="Reference frequency for the waveform.",
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=5,
        help="Minimum frequency for the waveform.",
    )

    return parser


def main(args):
    prior_file = args.prior_file
    injection_file = args.injection_file

    injection_dir = injection_file.parent
    injection_dir.mkdir(parents=True, exist_ok=True)

    start_time = 1364342418
    injection_time = start_time + 256

    asd_files = {
        "H1": args.psd_path / "aligo_O3actual_H1.txt",
        "L1": args.psd_path / "aligo_O3actual_L1.txt",
        "V1": args.psd_path / "avirgo_O3actual.txt",
    }
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifos.set_strain_data_from_zero_noise(duration=8, sampling_frequency=4096)
    for ifo in ifos:
        ifo.minimum_frequency = 20
        ifo.maximum_frequency = 2048
        ifo.power_spectral_density = bilby.gw.detector.psd.PowerSpectralDensity(
            asd_file=asd_files[ifo.name]
        )

    if args.waveform == "SEOBNRv5PHM":
        source_function = bilby.gw.source.gwsignal_binary_black_hole
    elif args.waveform == "SEOBNRv5EHM":
        source_function = bilby.gw.source.gwsignal_eccentric_binary_black_hole
    elif args.waveform == "TaylorF2Ecc":
        from aspire_analysis_tools.gw.source import (
            eccentric_binary_black_hole_aligned_spins,
        )

        source_function = eccentric_binary_black_hole_aligned_spins
    else:
        source_function = bilby.gw.source.lal_binary_black_hole

    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=8,
        sampling_frequency=4096,
        frequency_domain_source_model=source_function,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant=args.waveform,
            reference_frequency=args.reference_frequency,
            minimum_frequency=args.minimum_frequency,
        ),
    )

    prior = bilby.gw.prior.CBCPriorDict(filename=prior_file)
    prior.pop("geocent_time", None)
    n_samples = args.n_samples

    if not prior.non_fixed_keys:
        if n_samples > 1:
            raise ValueError("Cannot sample from a prior with no non-fixed keys.")
        print("Prior appears to be a single set of parameters")
        # Pass each line x=1 as a dictionary
        with open(prior_file, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split("=") for line in lines]
        lines = {line[0]: float(line[1]) for line in lines}
        injection_params = deepcopy(lines)
        samples = pd.DataFrame([injection_params])
    else:
        samples = pd.DataFrame(prior.sample(n_samples))
    samples["injection_time"] = injection_time
    samples["geocent_time"] = injection_time + np.random.uniform(
        -0.1, 0.1, size=n_samples
    )
    print(samples)
    snrs = {ifo.name: np.zeros(n_samples) for ifo in ifos}

    for ii in trange(n_samples):
        params = dict(samples.iloc[ii])
        wf = wfg.frequency_domain_strain(params)
        for ifo in ifos:
            signal = ifo.get_detector_response(
                waveform_polarizations=wf, parameters=params
            )
            snrs[ifo.name][ii] = np.real(ifo.optimal_snr_squared(signal)) ** 0.5
    snrs["network"] = np.linalg.norm([snrs[ifo.name] for ifo in ifos], axis=0)

    print(
        np.percentile(snrs["network"], 10),
        np.percentile(snrs["network"], 50),
        np.percentile(snrs["network"], 90),
    )
    for key in snrs:
        samples[f"{key}_snr"] = snrs[key]
    samples["approximant"] = args.waveform
    samples.to_hdf(injection_file, key="injections")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
