#!/usr/bin/env python3
"""Proposal-moment efficiency study with MiniPCN-SMC.

This script studies SMC efficiency while directly changing proposal moments
(mean and/or variance), instead of targeting a KL value.

Supported sweep modes:
- mean:     vary proposal mean, keep sigma fixed at posterior sigma
- variance: vary proposal sigma/variance, keep mean fixed at posterior mean
- both:     vary both mean and sigma on a 2D grid
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussian_kl(mu_q: float, sigma_q: float, mu_p: float, sigma_p: float) -> float:
    var_q = sigma_q**2
    var_p = sigma_p**2
    return 0.5 * (
        math.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
    )


def posterior_parameters(
    mu_prior: float,
    sigma_prior: float,
    mu_likelihood: float,
    sigma_likelihood: float,
) -> tuple[float, float]:
    var_prior = sigma_prior**2
    var_likelihood = sigma_likelihood**2
    var_post = 1.0 / (1.0 / var_prior + 1.0 / var_likelihood)
    mu_post = var_post * (mu_prior / var_prior + mu_likelihood / var_likelihood)
    return mu_post, math.sqrt(var_post)


def log_normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + z * z)


@dataclass
class GaussianProposalFlow:
    """Minimal flow-compatible isotropic Gaussian proposal."""

    mu: float
    sigma: float
    dims: int
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample_and_log_prob(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.rng.normal(self.mu, self.sigma, size=(n_samples, self.dims))
        return x, self.log_prob(x)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x2 = np.asarray(x)
        return log_normal_pdf(x2, self.mu, self.sigma).sum(axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dims",
        type=str,
        nargs="+",
        action="extend",
        required=True,
    )
    p.add_argument(
        "--sweep-mode",
        choices=["mean", "sigma", "both"],
        default="both",
        help="Which proposal moments to sweep.",
    )
    p.add_argument("--mu-min", type=float, default=-2.0)
    p.add_argument("--mu-max", type=float, default=2.0)
    p.add_argument("--n-mu-points", type=int, default=11)
    p.add_argument("--mu-val", type=float, default=None, help="If set, overrides mu grid with this fixed value.")

    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=5.0)
    p.add_argument("--n-sigma-points", type=int, default=11)
    p.add_argument("--sigma-val", type=float, default=None, help="If set, overrides sigma grid with this fixed value.")

    p.add_argument("--n-particles", type=int, default=256)
    p.add_argument("--n-repeats", type=int, default=8)
    p.add_argument("--mu-prior", type=float, default=0.0)
    p.add_argument("--sigma-prior", type=float, default=3.0)
    p.add_argument("--mu-likelihood", type=float, default=2.0)
    p.add_argument("--sigma-likelihood", type=float, default=1.0)
    p.add_argument("--target-efficiency", type=float, default=0.6)
    p.add_argument("--mcmc-nsteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--aspire-repo",
        type=Path,
        default=Path("/home/michael/git_repos/aspire"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/michael/git_repos/aspire-analyses/kl_criteria/out"),
    )
    return p.parse_args()


def total_kl_to_posterior(
    *,
    dims: int,
    mu_q: float,
    sigma_q: float,
    mu_post: float,
    sigma_post: float,
) -> float:
    return dims * gaussian_kl(mu_q, sigma_q, mu_post, sigma_post)


def total_kl_posterior_to_q(
    *,
    dims: int,
    mu_q: float,
    sigma_q: float,
    mu_post: float,
    sigma_post: float,
) -> float:
    return dims * gaussian_kl(mu_post, sigma_post, mu_q, sigma_q)


def total_wasserstein_distance(
    *,
    dims: int,
    mu_q: float,
    sigma_q: float,
    mu_ref: float,
    sigma_ref: float,
) -> float:
    delta_mu = mu_q - mu_ref
    delta_sigma = sigma_q - sigma_ref
    return math.sqrt(dims * (delta_mu * delta_mu + delta_sigma * delta_sigma))


def run_single_smc(
    *,
    dims: int,
    proposal_mu: float,
    proposal_sigma: float,
    run_seed: int,
    args: argparse.Namespace,
) -> tuple[int, int]:
    import array_api_compat.numpy as xp

    import sys

    sys.path.insert(0, str(args.aspire_repo / "src"))

    from aspire.samplers.smc.minipcn import MiniPCNSMC
    from aspire.samples import Samples
    from orng import ArrayRNG

    flow = GaussianProposalFlow(
        mu=proposal_mu,
        sigma=proposal_sigma,
        dims=dims,
        seed=run_seed,
    )

    def log_prior(samples: Samples):
        x = np.asarray(samples.x)
        return log_normal_pdf(x, args.mu_prior, args.sigma_prior).sum(axis=1)

    def log_likelihood(samples: Samples):
        x = np.asarray(samples.x)
        return log_normal_pdf(args.mu_likelihood, x, args.sigma_likelihood).sum(axis=1)

    sampler = MiniPCNSMC(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        prior_flow=flow,
        xp=xp,
    )

    rng = ArrayRNG(seed=int(run_seed), backend="numpy")

    _ = sampler.sample(
        n_samples=args.n_particles,
        adaptive=True,
        target_efficiency=float(args.target_efficiency),
        sampler_kwargs={
            "n_steps": int(args.mcmc_nsteps or 5 * dims),
            "target_acceptance_rate": 0.234,
            "step_fn": "tpcn",
        },
        rng=rng,
    )

    n_beta_steps = len(sampler.history.beta)
    n_like_evals = int(sampler.n_likelihood_evaluations)
    return n_beta_steps, n_like_evals


def mean_std(values: np.ndarray) -> tuple[float, float]:
    m = float(values.mean())
    s = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return m, s


def main() -> None:
    args = parse_args()
    dims_list = list(map(int, args.dims))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("aspire").setLevel(logging.WARNING)

    mu_post, sigma_post = posterior_parameters(
        args.mu_prior, args.sigma_prior, args.mu_likelihood, args.sigma_likelihood,
    )

    if args.mu_val is not None:
        args.mu_min = args.mu_max = args.mu_val
    if args.sigma_val is not None:
        args.sigma_min = args.sigma_max = args.sigma_val
    mu_grid = np.linspace(args.mu_min, args.mu_max, args.n_mu_points)
    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma_points)

    if args.sweep_mode == "mean":
        points = [(float(mu), float(sigma_post)) for mu in mu_grid]
    elif args.sweep_mode == "variance":
        points = [(float(mu_post), float(s)) for s in sigma_grid]
    else:
        points = [(float(mu), float(s)) for mu in mu_grid for s in sigma_grid]

    base_rng = np.random.default_rng(args.seed)
    per_dim_results = {}

    for dims in dims_list:
        logging.info(
            "Running dims=%s sweep=%s with %s points",
            dims,
            args.sweep_mode,
            len(points),
        )

        n_points = len(points)
        mu_values = np.zeros(n_points)
        sigma_values = np.zeros(n_points)

        beta_steps_init = np.zeros((n_points, args.n_repeats), dtype=float)
        like_evals_init = np.zeros((n_points, args.n_repeats), dtype=float)

        beta_steps_prior = np.zeros(args.n_repeats, dtype=float)
        like_evals_prior = np.zeros(args.n_repeats, dtype=float)

        logging.info(f"Running inference from the prior")
        for j in range(args.n_repeats):
            run_seed = int(base_rng.integers(1, 2**31 - 1))
            b, nll = run_single_smc(
                dims=dims,
                proposal_mu=args.mu_prior,
                proposal_sigma=args.sigma_prior,
                run_seed=run_seed,
                args=args,
            )
            beta_steps_prior[j] = b
            like_evals_prior[j] = nll

        logging.info(f"Running inference from the initial proposal")
        for i, (mu_init, sigma_init) in enumerate(points):
            mu_values[i] = mu_init
            sigma_values[i] = sigma_init
            for j in range(args.n_repeats):
                run_seed = int(base_rng.integers(1, 2**31 - 1))
                b, nll = run_single_smc(
                    dims=dims,
                    proposal_mu=mu_init,
                    proposal_sigma=sigma_init,
                    run_seed=run_seed,
                    args=args,
                )
                beta_steps_init[i, j] = b
                like_evals_init[i, j] = nll

        per_dim_results[dims] = dict(
            mu_values=mu_values,
            sigma_values=sigma_values,
            beta_init_mean=beta_steps_init.mean(axis=1),
            beta_init_std=(
                beta_steps_init.std(axis=1, ddof=1)
                if args.n_repeats > 1
                else np.zeros(n_points)
            ),
            like_init_mean=like_evals_init.mean(axis=1),
            like_init_std=(
                like_evals_init.std(axis=1, ddof=1)
                if args.n_repeats > 1
                else np.zeros(n_points)
            ),
            beta_prior_mean=mean_std(beta_steps_prior)[0],
            beta_prior_std=mean_std(beta_steps_prior)[1],
            like_prior_mean=mean_std(like_evals_prior)[0],
            like_prior_std=mean_std(like_evals_prior)[1],
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "dims,sweep_mode,mu_init,sigma_init,"
        "beta_steps_mean,beta_steps_std,like_evals_mean,like_evals_std,"
        "beta_steps_prior_mean,beta_steps_prior_std,"
        "like_evals_prior_mean,like_evals_prior_std\n"
    )

    per_dim_csv_paths = {}
    for dims in dims_list:
        res = per_dim_results[dims]
        dim_csv_path = args.output_dir / f"proposal_moments_efficiency_dims{dims}.csv"
        per_dim_csv_paths[dims] = dim_csv_path
        with dim_csv_path.open("w", encoding="utf-8") as f:
            f.write(header)
            for i in range(len(res["mu_values"])):
                f.write(
                    f"{dims},{args.sweep_mode},"
                    f"{res['mu_values'][i]:.8g},{res['sigma_values'][i]:.8g},"
                    f"{res['beta_init_mean'][i]:.8g},{res['beta_init_std'][i]:.8g},"
                    f"{res['like_init_mean'][i]:.8g},{res['like_init_std'][i]:.8g},"
                    f"{res['beta_prior_mean']:.8g},{res['beta_prior_std']:.8g},"
                    f"{res['like_prior_mean']:.8g},{res['like_prior_std']:.8g}\n"
                )

    csv_path = args.output_dir / "proposal_moments_efficiency.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write(header)
        for dims in dims_list:
            lines = per_dim_csv_paths[dims].read_text(encoding="utf-8").splitlines()
            for line in lines[1:]:
                if line.strip():
                    fout.write(line + "\n")

if __name__ == "__main__":
    main()
