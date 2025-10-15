#!/usr/bin/env python
"""Script to fetch results specified in the results.yaml file, rename them,
and store them in a specified output directory.

Results are symbolically linked to avoid unnecessary duplication of data.
"""
import argparse
from pathlib import Path
import yaml


def get_parser():
    parser = argparse.ArgumentParser(
        description="Fetch and rename result files from a YAML file"
    )
    parser.add_argument(
        "yaml",
        type=Path,
        help="Path to the YAML file containing the results",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Output directory for the renamed files",
        required=True,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        help="Path to save the updated YAML file with new file paths",
        default=None,
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.yaml, "r") as f:
        yaml_content = yaml.safe_load(f)

    updated_yaml_content = {}

    if not yaml_content:
        parser.error(f"No output directories found in '{args.yaml}'.")

    # Yaml file structure:
    # label:
    #   sampler: path to results file
    for label in yaml_content:
        updated_yaml_content[label] = {}
        for waveform, entries in yaml_content[label].items():
            updated_yaml_content[label][waveform] = {}
            for analysis, result_file in entries.items():
                src_file = Path(result_file)
                if not src_file.exists():
                    raise FileNotFoundError(f"Source file '{src_file}' does not exist.")
                out_filename = f"{label}_{waveform}_{analysis}_result.hdf5"
                out_path = outdir / out_filename
                # Replace - with _ in the filename
                out_path = out_path.with_name(out_path.name.replace("-", "_"))
                if args.verbose:
                    print(f"Linking '{src_file}' to '{out_path}'")
                if out_path.exists() and not args.overwrite:
                    if args.verbose:
                        print(f"Output file '{out_path}' already exists. Skipping.")
                    continue
                out_path.symlink_to(src_file)
                updated_yaml_content[label][waveform][analysis] = str(out_path)

    if args.output_yaml:
        with open(args.output_yaml, "w") as f:
            yaml.dump(updated_yaml_content, f)
        if args.verbose:
            print(f"Updated YAML file saved to '{args.output_yaml}'")


if __name__ == "__main__":
    main()