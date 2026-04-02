"""Example of using aspire with a high-dimensional Gaussian target distribution."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aspire import Aspire
from aspire.plot import plot_comparison
from aspire.samples import Samples
from aspire.utils import AspireFile, configure_logger
# import multiprocessing as mp
import multiprocess as mp

import scipy.stats


def log_normalizer_product_of_gaussians(mu, cov, prior_mu, prior_cov):
    """
    Computes log Z where
        Z = ∫ N(x; mu, cov) N(x; prior_mu, prior_cov) dx
          = N(mu; prior_mu, cov + prior_cov)

    Works for full SPD covariance matrices.
    """
    mu = np.asarray(mu)
    prior_mu = np.asarray(prior_mu)
    cov = np.asarray(cov)
    prior_cov = np.asarray(prior_cov)

    d = mu.shape[0]
    S = cov + prior_cov
    delta = mu - prior_mu

    # Use slogdet for numerical stability
    sign, logdetS = np.linalg.slogdet(S)
    if sign <= 0:
        raise ValueError("cov + prior_cov must be positive definite (det > 0).")

    # Quadratic form delta^T S^{-1} delta without explicit inverse
    quad = delta @ np.linalg.solve(S, delta)

    logZ = -0.5 * (d * np.log(2.0 * np.pi) + logdetS + quad)
    return logZ


def main():

    # RNG for generating initial samples
    rng = np.random.default_rng(42)

    dims = 15

    # Output directory
    outdir = Path("outdir") / "gaussian_comparison_fixed" / f"{dims}d"
    outdir.mkdir(parents=True, exist_ok=True)

    # Configure logger to show INFO level messages
    configure_logger(additional_loggers=["aspire_ptemcee"])

    # Mean and covariance of the Gaussian target distribution
    mu = 2 * np.ones(dims)
    cov = np.eye(dims)

    prior_mu = 1 * np.ones(dims)
    prior_cov = 2 * np.eye(dims)

    proposal_mu = 1.5 * np.ones(dims)
    proposal_cov = 1 * np.eye(dims)

    cov_inv = np.linalg.inv(cov)
    prior_cov_inv = np.linalg.inv(prior_cov)

    posterior_cov = np.linalg.inv(cov_inv + prior_cov_inv)
    posterior_mu = posterior_cov @ (cov_inv @ mu + prior_cov_inv @ prior_mu)

    print("Posterior mean:", posterior_mu)
    print("Posterior covariance:", posterior_cov)

    log_z = log_normalizer_product_of_gaussians(mu, cov, prior_mu, prior_cov)

    print("Log normalizer (log Z):", log_z)

    true_posterior_samples = rng.multivariate_normal(posterior_mu, posterior_cov, size=5000)
    true_samples = Samples(true_posterior_samples)
    
    logl_dist = scipy.stats.multivariate_normal(mean=mu, cov=cov)
    logp_dist = scipy.stats.multivariate_normal(mean=prior_mu, cov=prior_cov)

    def log_likelihood(samples):
        """Log-likelihood of a mixture of two Gaussians"""
        x = samples.x
        return logl_dist.logpdf(x)

    def log_prior(samples):
        """Standard normal prior with std = 3 to make it less informative"""
        return logp_dist.logpdf(samples.x)

    # Generate prior samples for comparison, these are not used in SMC
    prior_samples = Samples(rng.multivariate_normal(prior_mu, prior_cov, size=5000))

    # Generate initial samples that are slightly better than the prior
    initial_samples = Samples(
        rng.multivariate_normal(proposal_mu, proposal_cov, size=5000)
    )

    # True posterior samples

    # Initialize Aspire with the log-likelihood and log-prior
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
    )

    # Fit the normalizing flow to the initial samples
    fit_history = aspire.fit(initial_samples, n_epochs=30)
    # Plot loss
    fit_history.plot_loss().savefig(outdir / "loss.png")

    samples, history = aspire.sample_posterior(
        sampler="smc",  # Sequential Monte Carlo, this uses the default minipcn sampler
        n_samples=5000,  # Number of particles in SMC
        sampler_kwargs=dict(  # Keyword arguments for the specific sampler
            n_steps=50, 
        ),
        target_efficiency=0.9,
        return_history=True,  # To return the SMC history (e.g., ESS, betas)
        rng=rng,
        n_final_samples=10_000,  # Number of final posterior samples to return after resampling
    )
    
    smc_likelihood_evals = aspire.sampler.n_likelihood_evaluations

    # Plot SMC diagnostics
    history.plot().savefig(outdir / "smc_diagnostics.png")
    # Plot SMC sample history (e.g., log-likelihood of samples over iterations)
    history.plot_sample_history(x_axis="log_likelihood").savefig(
        outdir / "smc_sample_history.png"
    )
    history.plot_sample_history(x_axis="log_p_t").savefig(
        outdir / "smc_sample_history_log_p_t.png"
    )

    ptmcmc_samples = aspire.sample_posterior(
        sampler="ptemcee",
        ntemps=5,
        nwalkers=250,
        nsteps=2000,
        rng=rng,
        vectorize=True,  # Use vectorized likelihood and prior evaluations for speed
    )
    
    ptmcmc_evals = aspire.sampler.n_likelihood_evaluations

    burn_in = 100
    colors = plt.cm.plasma(np.linspace(0, 1, ptmcmc_samples.chain.shape[0]))
    for beta_index in range(ptmcmc_samples.chain.shape[0]):
        ptmcmc_samples.plot_chain(
            beta_index,
            n_walkers=10,
            color=colors[beta_index],
            burn_in=burn_in,
        ).savefig(
            outdir / f"ptmcmc_chain_beta_{beta_index}.png"
        )
        plt.close()

    acor = ptmcmc_samples.autocorrelation_time
    print("Autocorrelation times for PTMCMC samples:", acor)
    mean_acor = int(max(1, np.mean(acor)))
    print("Mean autocorrelation time for PTMCMC samples:", mean_acor)
    ptmcmc_samples_pp = ptmcmc_samples.post_process(
        burn_in=burn_in,
        thin=int(mean_acor)
    )
    ptmcmc_posterior_samples = ptmcmc_samples_pp.cold_chain()
    print(f"Number of PTMCMC posterior samples after burn-in and thinning: {len(ptmcmc_posterior_samples)}")
    
    # Run ptemcee with a different temperature schedule to see how it affects the results
    ptmcmc_samples_alt = aspire.sample_posterior(
        sampler="ptemcee",
        ntemps=16,
        nwalkers=250,
        nsteps=2000,
        rng=rng,
        proposal="flow",  # Use the flow-based proposal for better mixing
    )
    
    ptmcmc_alt_evals = aspire.sampler.n_likelihood_evaluations
    
    burn_in_alt = 100
    acor_alt = ptmcmc_samples_alt.autocorrelation_time
    print("Autocorrelation times for PTMCMC samples with alternative temperature schedule:", acor_alt)
    mean_acor_alt = int(max(1, np.mean(acor_alt)))
    print("Mean autocorrelation time for PTMCMC samples with alternative temperature schedule:", mean_acor_alt)
    
    colors = plt.cm.viridis(np.linspace(0, 1, ptmcmc_samples_alt.chain.shape[0]))
    for beta_index in range(ptmcmc_samples_alt.chain.shape[0]):
        ptmcmc_samples_alt.plot_chain(
            beta_index,
            n_walkers=10,
            color=colors[beta_index],
            burn_in=burn_in_alt,
        ).savefig(
            outdir / f"ptmcmc_alt_chain_beta_{beta_index}.png"
        )
        plt.close()
    
    ptmcmc_samples_alt_pp = ptmcmc_samples_alt.post_process(
        burn_in=burn_in_alt,
        thin=int(mean_acor_alt)
    )
    
    ptmcmc_posterior_samples_alt = ptmcmc_samples_alt_pp.cold_chain()
    print(f"Number of PTMCMC posterior samples with alternative temperature schedule after burn-in and thinning: {len(ptmcmc_posterior_samples_alt)}")
    
    ptmcmc_samples_flow = aspire.sample_posterior(
        sampler="ptemcee",
        ntemps=16,
        nwalkers=250,
        nsteps=2000,
        rng=rng,
        proposal="flow",  # Use the flow-based proposal for better mixing
    )
    
    ptmcmc_flow_evals = aspire.sampler.n_likelihood_evaluations
    
    burn_in_flow = 200
    acor_flow = ptmcmc_samples_flow.autocorrelation_time
    print("Autocorrelation times for PTMCMC samples with flow proposal:", acor_flow)
    mean_acor_flow = int(max(1, np.mean(acor_flow)))
    print("Mean autocorrelation time for PTMCMC samples with flow proposal:", mean_acor_flow)   
    
    colors = plt.cm.inferno(np.linspace(0, 1, ptmcmc_samples_flow.chain.shape[0]))
    for beta_index in range(ptmcmc_samples_flow.chain.shape[0]):
        ptmcmc_samples_flow.plot_chain(
            beta_index,
            n_walkers=10,
            color=colors[beta_index],
            burn_in=burn_in_flow,
        ).savefig(
            outdir / f"ptmcmc_flow_chain_beta_{beta_index}.png"
        )
        plt.close()
    
    ptmcmc_samples_flow_pp = ptmcmc_samples_flow.post_process(
        burn_in=burn_in_flow,
        thin=int(mean_acor_flow)
    )
    
    ptmcmc_posterior_samples_flow = ptmcmc_samples_flow_pp.cold_chain()
    print(f"Number of PTMCMC posterior samples with flow proposal after burn-in and thinning: {len(ptmcmc_posterior_samples_flow)}")
    
    # Save the results to a file
    # AspireFile is a small wrapper around h5py.File that automatically includes
    # additional metadata
    with AspireFile(outdir / "aspire_smc_results.h5", "w") as f:
        aspire.save_config(f, "aspire_config")
        samples.save(f, "posterior_samples")
        ptmcmc_samples.save(f, "ptmcmc_samples")
        ptmcmc_posterior_samples_alt.save(f, "ptmcmc_posterior_samples_alt")
        ptmcmc_posterior_samples_flow.save(f, "ptmcmc_posterior_samples_flow")
        history.save(f, "smc_history")
        aspire.save_flow(f, "flow")
        fit_history.save(f, "fit_history")

    # Plot corner plot of the samples
    # Include initial samples and prior samples for comparison
    plot_comparison(
        initial_samples,
        true_samples,
        samples,
        ptmcmc_posterior_samples,
        ptmcmc_posterior_samples_alt,
        labels=["Initial Samples", "True Samples", "SMC Samples", "PTMCMC Samples (Alt)"],
    ).savefig(outdir / "posterior.png")

    print("Number of likelihood evaluations:")
    print(f"SMC: {smc_likelihood_evals}")
    print(f"PTMCMC: {ptmcmc_evals}")
    print(f"PTMCMC (alt): {ptmcmc_alt_evals}")
    print(f"PTMCMC (flow proposal): {ptmcmc_flow_evals}")

    # Print evidence values
    print("Log evidence (log Z):", log_z)
    print(f"SMC log evidence: {samples.log_evidence:.4f} ± {samples.log_evidence_error:.4f}")

    log_z_ptmcmc_ti = ptmcmc_samples_pp.log_evidence_thermodynamic_integration(0)
    log_z_ptmcmc_ti_coarse = ptmcmc_samples_pp.log_evidence_thermodynamic_integration(0, method="coarse")
    print(f"PTMCMC log evidence (thermodynamic integration): {log_z_ptmcmc_ti[0]:.4f} ± {log_z_ptmcmc_ti[1]:.4f}")
    print(f"PTMCMC log evidence (coarse thermodynamic integration): {log_z_ptmcmc_ti_coarse[0]:.4f} ± {log_z_ptmcmc_ti_coarse[1]:.4f}")
    log_z_ptmcmc_ss = ptmcmc_samples_pp.log_evidence_stepping_stone(0)
    print(f"PTMCMC log evidence (stepping stone): {log_z_ptmcmc_ss[0]:.4f} ± {log_z_ptmcmc_ss[1]:.4f}")
    
    log_z_ptmcmc_ti_alt = ptmcmc_samples_alt_pp.log_evidence_thermodynamic_integration(0)
    log_z_ptmcmc_ti_coarse_alt = ptmcmc_samples_alt_pp.log_evidence_thermodynamic_integration(0, method="coarse")
    print(f"PTMCMC (alt) log evidence (thermodynamic integration): {log_z_ptmcmc_ti_alt[0]:.4f} ± {log_z_ptmcmc_ti_alt[1]:.4f}")
    print(f"PTMCMC (alt) log evidence (coarse thermodynamic integration): {log_z_ptmcmc_ti_coarse_alt[0]:.4f} ± {log_z_ptmcmc_ti_coarse_alt[1]:.4f}")
    log_z_ptmcmc_ss_alt = ptmcmc_samples_alt_pp.log_evidence_stepping_stone(0)
    print(f"PTMCMC (alt) log evidence (stepping stone): {log_z_ptmcmc_ss_alt[0]:.4f} ± {log_z_ptmcmc_ss_alt[1]:.4f}")
    
    log_z_ptmcmc_ti_flow = ptmcmc_samples_flow_pp.log_evidence_thermodynamic_integration(0)
    log_z_ptmcmc_ti_coarse_flow = ptmcmc_samples_flow_pp.log_evidence_thermodynamic_integration(0, method="coarse")
    print(f"PTMCMC (flow) log evidence (thermodynamic integration): {log_z_ptmcmc_ti_flow[0]:.4f} ± {log_z_ptmcmc_ti_flow[1]:.4f}")
    print(f"PTMCMC (flow) log evidence (coarse thermodynamic integration): {log_z_ptmcmc_ti_coarse_flow[0]:.4f} ± {log_z_ptmcmc_ti_coarse_flow[1]:.4f}")
    log_z_ptmcmc_ss_flow = ptmcmc_samples_flow_pp.log_evidence_stepping_stone(0)
    print(f"PTMCMC (flow) log evidence (stepping stone): {log_z_ptmcmc_ss_flow[0]:.4f} ± {log_z_ptmcmc_ss_flow[1]:.4f}")
    
    # Save all evidence so they can be written to a table later
    evidence_results = {
        "log_z": log_z,
        "smc_log_z": (samples.log_evidence, samples.log_evidence_error),
        "ptmcmc_log_z_ti": log_z_ptmcmc_ti,
        "ptmcmc_log_z_ti_coarse": log_z_ptmcmc_ti_coarse,
        "ptmcmc_log_z_ss": log_z_ptmcmc_ss,
        "ptmcmc_alt_log_z_ti": log_z_ptmcmc_ti_alt,
        "ptmcmc_alt_log_z_ti_coarse": log_z_ptmcmc_ti_coarse_alt,
        "ptmcmc_alt_log_z_ss": log_z_ptmcmc_ss_alt,
        "ptmcmc_flow_log_z_ti": log_z_ptmcmc_ti_flow,
        "ptmcmc_flow_log_z_ti_coarse": log_z_ptmcmc_ti_coarse_flow,
        "ptmcmc_flow_log_z_ss": log_z_ptmcmc_ss_flow,
        "smc_likelihood_evals": smc_likelihood_evals,
        "ptmcmc_likelihood_evals": ptmcmc_evals,
        "ptmcmc_alt_likelihood_evals": ptmcmc_alt_evals,
        "ptmcmc_flow_likelihood_evals": ptmcmc_flow_evals,
        "smc_n_samples": len(samples),
        "ptmcmc_n_samples": len(ptmcmc_posterior_samples),
        "ptmcmc_alt_n_samples": len(ptmcmc_posterior_samples_alt),
        "ptmcmc_flow_n_samples": len(ptmcmc_posterior_samples_flow),
    }
    with open(outdir / "evidence_results.json", "w") as f:
        json.dump(evidence_results, f, indent=4)


if __name__ == "__main__":
    main()
