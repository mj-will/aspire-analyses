#!/usr/bin/env python
"""
Script to collate result files in nested directories and run a probability-
probability test.

Searches recursively for directories called `final_result` that contain
files with the specified extension.
"""

import argparse
from itertools import product
from collections import namedtuple
import re
from pesummary.gw.plots.latex_labels import GWlatex_labels
import os
import numpy as np
import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats
from natsort import natsorted
import json

from bilby.core.result import read_in_result


def get_injection_credible_level(result, parameter, injection_parameters, weights=None):
    if weights is None:
        weights = np.ones(len(result.posterior[parameter]))
    if parameter not in injection_parameters:
        raise ValueError(f"Parameter {parameter} not found in injections")
    credible_level = np.sum(
        np.array(result.posterior[parameter] < injection_parameters[parameter])
        * weights
    ) / np.sum(weights)
    return credible_level


def get_all_credible_levels(result, injection_parameters, keys, weights=None):
    return {
        key: get_injection_credible_level(
            result, key, injection_parameters, weights=weights
        )
        for key in keys
    }

def pp_plot_from_credible_levels(
    credible_levels,
    confidence_interval=[0.68, 0.95, 0.997],
    lines=None,
    legend_fontsize="x-small",
    title=True,
    confidence_interval_alpha=0.1,
    **kwargs,
):
    """
    Make a P-P plot from a dataframe of credible levels.
    """
    if lines is None:
        colors = ["C{}".format(i) for i in range(8)]
        linestyles = ["-", "--", ":"]
        lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    if len(lines) < len(credible_levels.keys()):
        raise ValueError("Larger number of parameters than unique linestyles")

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)

    figsize = plt.rcParams["figure.figsize"].copy()
    # figsize[1] = 1.5 * figsize[1]
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(
            confidence_interval
        )
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval"
        )

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1.0 - ci) / 2.0
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color="k")

    pvalues = []
    print("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array(
            [
                sum(credible_levels[key].values < xx) / len(credible_levels)
                for xx in x_values
            ]
        )
        pvalue = scipy.stats.kstest(credible_levels[key], "uniform").pvalue
        pvalues.append(pvalue)
        print(f"{key}: {pvalue}")

        name = GWlatex_labels.get(key, key)
        name = re.sub(r"\[.*?\]", "", name)
        label = "{} ({:2.3f})".format(name, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label, **kwargs)

    Pvals = namedtuple("pvals", ["combined_pvalue", "pvalues", "names"])
    pvals = Pvals(
        combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
        pvalues=pvalues,
        names=list(credible_levels.keys()),
    )
    print("Combined p-value: {}".format(pvals.combined_pvalue))

    if title:
        ax.set_title(
            "N={}, $p$-value={:2.4f}".format(
                len(credible_levels), pvals.combined_pvalue
            ),
            fontsize=10,
        )
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.legend(
        handlelength=2,
        labelspacing=0.25,
        fontsize=legend_fontsize,
        loc="center",
        bbox_to_anchor=(0.55, -0.07),
        ncol=4,
    )
    fig.tight_layout()
    return fig, pvals


def get_parser():
    parser = argparse.ArgumentParser(
        description="Collate result files in nested directories and run a probability-probability test."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="hdf5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="png",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="pp_test",
    )
    parser.add_argument(
        "--credible-levels-file",
        type=str,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the credible levels file if it exists.",
    )
    return parser


def discover_result_files(result_dir, extension):
    result_files = {}
    for dirpath, _, filenames in os.walk(result_dir):
        print(filenames)
        for filename in filenames:
            if filename.endswith(extension):
                # Get the injection ID from the path with formation `<label>_data<id>_0_<something>_result.hdf5`
                inj_id = filename.split("_data")[-1].split("_")[0]
                result_files[inj_id] = os.path.join(dirpath, filename)

    # Sort the result files by injection ID
    result_files = dict(natsorted(result_files.items()))
    return result_files


def main(args):

    plt.style.use("plots.style")
    plt.rcParams["text.usetex"] = False

    with open(args.injection_file, "r") as f:
        injection_parameters = json.load(f)["injections"]["content"]
        # Convert from dict of lists per parameter to list of dicts per injection
        injection_parameters = [
            {key: injection_parameters[key][i] for key in injection_parameters}
            for i in range(len(injection_parameters["chirp_mass"]))
        ]

    # injection_parameters = pd.read_hdf(args.injection_file, key="injections")
    # injection_parameters = injection_parameters.to_dict(orient="records")

    if args.outdir is None:
        outdir = args.result_dir
    else:
        outdir = args.outdir
        outdir.mkdir(exist_ok=True, parents=True)

    keys = [
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "luminosity_distance",
        "dec",
        "ra",
        "theta_jn",
        "psi",
        "geocent_time",
        "phase",
    ]

    credible_levels_filename = args.credible_levels_file

    if credible_levels_filename is not None:
        credible_levels_filename = Path(credible_levels_filename)
        credible_levels_filename.parent.mkdir(exist_ok=True, parents=True)

    if credible_levels_filename and credible_levels_filename.exists() and not args.overwrite:
        print(f"Loading credible levels from {credible_levels_filename}")
        # Load credible levels from a file if provided
        credible_levels = pd.read_hdf(credible_levels_filename, key="credible_levels")
        if not set(keys).issubset(set(credible_levels.columns)):
            raise ValueError("Credible levels file does not contain all keys.")
    else:
        result_files = discover_result_files(args.result_dir, args.extension)

        print(f"Found {len(result_files)} result files")
        print("Reading in results")
        results = []
        for rf in tqdm.tqdm(result_files.values()):
            results.append(read_in_result(rf))
        credible_levels = list()
        for i, result in enumerate(results):
            credible_levels.append(
                get_all_credible_levels(
                    result=result,
                    injection_parameters=injection_parameters[i],
                    keys=keys,
                )
            )
        credible_levels = pd.DataFrame(credible_levels)
        if credible_levels_filename is not None:
            print(f"Saving credible levels to {credible_levels_filename}")
            credible_levels.to_hdf(
                credible_levels_filename,
                key="credible_levels",
                mode="w",
                format="table",
                data_columns=True,
            )

    print("Producing P-P plot")
    fig, p_values = pp_plot_from_credible_levels(
        credible_levels=credible_levels,
    )
    filename = outdir / f"{args.filename}.{args.figure_format}"
    fig.savefig(filename)

    with open(outdir / "p_values.json", "w") as f:
        json.dump(p_values._asdict(), f)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)