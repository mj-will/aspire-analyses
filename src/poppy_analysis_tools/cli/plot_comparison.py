import bilby
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description="Plot corner plots from results files")
    parser.add_argument(
        "-r", "--results", nargs="+", help="List of results files to use."
    )
    parser.add_argument("-f", "--filename", default=None, help="Output file name.")
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        default=None,
        help="List of labels to use for each result.",
    )
    parser.add_argument(
        "-p", "--parameters", nargs="+", default=None, help="List of parameters."
    )
    parser.add_argument("-i", "--injection-parameters-file", default=None, type=str)
    parser.add_argument("-id", "--injection-id", default=None, type=int)
    return parser


def load_injection_parameters(injection_file, injection_id):
    """
    Load injection parameters from a file.
    """
    injection_parameters = pd.read_hdf(injection_file, key="injections").iloc[
        injection_id
    ]
    return injection_parameters.to_dict()


def main():
    args = get_parser().parse_args()

    results = [bilby.core.result.read_in_result(result) for result in args.results]

    if args.labels is None:
        labels = [f"Result {i}" for i in range(len(results))]
    else:
        labels = args.labels

    if args.parameters is None:
        common_parameters = set(results[0].search_parameter_keys)
        for result in results[1:]:
            common_parameters = common_parameters.intersection(
                set(result.search_parameter_keys)
            )
        parameters = list(common_parameters)
    else:
        parameters = args.parameters

    if args.injection_parameters_file is not None:
        injection_parameters = load_injection_parameters(
            args.injection_parameters_file,
            args.injection_id,
        )
        injection_parameters = bilby.gw.conversion.generate_all_bbh_parameters(
            injection_parameters
        )
        print(injection_parameters)
        parameters = {p: injection_parameters.get(p, np.nan) for p in parameters}
        print(f"Injection parameters: {parameters}")

    colours = [f"C{i}" for i in range(len(results))]
    lines = [
        Line2D([0], [0], color=colours[i], label=labels[i]) for i in range(len(results))
    ]

    print("Producing corner plot")
    fig = None
    for i, result in enumerate(results):
        fig = result.plot_corner(
            parameters=parameters,
            fig=fig,
            color=colours[i],
            truth_color="k",
        )
    axes = fig.get_axes()

    # Scale axes
    for i, ax in enumerate(axes):
        ax.autoscale()
    plt.draw()

    # Add legend
    ndim = int(np.sqrt(len(axes)))
    axes[ndim - 1].legend(handles=lines)
    # Save
    fig.savefig(args.filename, bbox_inches="tight")
