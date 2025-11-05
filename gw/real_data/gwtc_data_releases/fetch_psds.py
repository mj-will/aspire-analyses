#!/usr/bin/env
"""
Script to fetch the PSD from GWTC-2.1/3 data releases and save them in
<SID>/psds/<key>-psd.dat.
"""

import argparse
import pathlib
import h5py
import numpy as np

def find_gwtc_results(
    data_release_path,
    data_releases,
    event,
    cosmo,
):
    for release in data_releases:
        if cosmo:
            suffix = "cosmo"
        else:
            suffix = "nocosmo"
        release_path = pathlib.Path(f"{data_release_path}/{release}/")
        if not release_path.exists():
            raise RuntimeError(f"Release path {release_path} does not exist")
        matches = list(release_path.glob(f"*-{event}_*{suffix}.h5"))
        if len(matches) > 1:
            raise RuntimeError("Found more than one file")
        elif len(matches) == 0:
            continue
        else:
            filepath = matches[0]
            break
    else:
        raise RuntimeError("No file found")
    return filepath, release


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SID", type=str)
    parser.add_argument(
        "--data-releases", nargs="+", type=str, default=["GWTC-2.1", "GWTC-3"]
    )
    parser.add_argument("--data-release-path", type=str, default="data_releases")
    parser.add_argument("--cosmo", action="store_true")
    parser.add_argument("--analysis", type=str, default="C01:IMRPhenomXPHM")
    parser.add_argument("--outdir", type=pathlib.Path)
    return parser


def main(args):
    filepath, release = find_gwtc_results(
        args.data_release_path, args.data_releases, args.SID, args.cosmo
    )

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "psds").mkdir(parents=True, exist_ok=True)
    (outdir / "cal").mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "r") as f:
        print(f[f"{args.analysis}/priors"].keys())
        print(f[f"{args.analysis}/priors/analytic/chirp_mass"][()])
        for key, psd in f[f"{args.analysis}/psds"].items():
            np.savetxt(outdir / "psds" /f"{key}-psd.dat", psd)
        for key, cal in f[f"{args.analysis}/calibration_envelope"].items():
            np.savetxt(outdir / "cal"/ f"{key}-cal.dat", cal)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)