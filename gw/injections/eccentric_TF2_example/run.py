#!/usr/bin/env python
"""
This script runs a bilby analysis of an eccentric binary black hole signal
injected into simulated LIGO data. It uses `bilby` to set up
the waveform generator, priors, likelihood, and samplers. The script
performs the following steps:
1. Sets up the environment and imports necessary libraries.
2. Defines the waveform model for eccentric binary black holes with aligned spins.
3. Sets up the injection parameters for the binary black hole.
4. Configures the waveform generator and interferometers.
5. Defines the priors for the parameters of the binary black hole.
6. Sets up the likelihood for the gravitational wave signal.
7. Performs initial inference with and without eccentricity using the `nessai` sampler.
8. Runs a second inference using the `poppy` sampler to post-process the results
   from the aligned spin model.
"""

import copy
import bilby
import numpy as np
from bilby.core.utils.random import seed
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
import json


def lal_eccentric_binary_black_hole(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    eccentricity,
    **kwargs,
):
    """Define a custom source function for eccentric binary black holes.

    This is needed because bilby checks the signature of the source function
    and the standard function does not accept the `eccentricity` parameter.
    """
    waveform_kwargs = dict(
        waveform_approximant="TaylorF2Ecc",
        reference_frequency=10.0,
        minimum_frequency=10.0,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        pn_spin_order=-1,
        pn_tidal_order=-1,
        pn_phase_order=-1,
        pn_amplitude_order=0,
    )
    waveform_kwargs.update(kwargs)
    return bilby.gw.source._base_lal_cbc_fd_waveform(
        frequency_array=frequency_array,
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        theta_jn=theta_jn,
        phase=phase,
        a_1=a_1,
        a_2=a_2,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        phi_jl=phi_jl,
        eccentricity=eccentricity,
        **waveform_kwargs,
    )


def calculate_fisco_from_solar_masses(
    m1_solar_mass: float, m2_solar_mass: float
) -> float:
    """
    Calculates an approximate ISCO (Innermost Stable Circular Orbit) orbital frequency
    for a binary system, based on the total mass and a non-spinning Schwarzschild
    black hole approximation.

    This approximation is often used as a termination frequency for inspiral
    waveforms like TaylorF2Ecc.

    Parameters
    ----------
    m1_solar_mass : float
        Mass of the primary compact object in solar masses.
    m2_solar_mass : float
        Mass of the secondary compact object in solar masses.

    Returns
    -------
    float
        The approximate ISCO orbital frequency in Hz.
    """
    from lal import MTSUN_SI as LAL_MTSUN_SI
    from lal import PI as LAL_PI

    # Total mass in solar masses
    m = m1_solar_mass + m2_solar_mass

    # Total mass in seconds (GM_total/c^3)
    # This directly uses the LAL_MTSUN_SI constant which is GM_sun/c^3
    m_sec = m * LAL_MTSUN_SI

    # This is the orbital velocity at the ISCO for a Schwarzschild black hole
    vISCO = 1.0 / np.sqrt(6.0)

    # Gravitational wave frequency at ISCO for a quasi-circular binary is 2 * orbital frequency
    # Orbital frequency f_orb = v^3 / (pi * M_total_seconds)
    # The C code snippet calculates the orbital frequency at ISCO.
    fISCO_orbital = vISCO * vISCO * vISCO / (LAL_PI * m_sec)

    return fISCO_orbital


def main(n_pool=1):
    # Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
    seed(1234)
    # Set OMP_NUM_THREADS=1
    os.environ["OMP_NUM_THREADS"] = "1"

    duration = 16
    sampling_frequency = 512

    outdir = Path("outdir_injection_12_seed")
    label = "eccentric_injection_12_full_priors_dynesty"
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    injection_parameters = dict(
        mass_1=35.0,
        mass_2=30.0,
        eccentricity=0.25,
        luminosity_distance=500.0,
        theta_jn=0.3,
        psi=0.1,
        phase=1.2,
        geocent_time=0.2,
        ra=np.pi / 4,
        dec=0.15,
        chi_1=0.05,
        chi_2=0.01,
    )

    isco_frequency = calculate_fisco_from_solar_masses(
        injection_parameters["mass_1"], injection_parameters["mass_2"]
    )
    print(f"Calculated ISCO frequency: {isco_frequency:.2f} Hz")
    # Ensure the ISCO frequency is within the waveform generator's frequency range
    minimum_frequency = 20.0
    maximum_frequency = 64
    # maximum_frequency = 128

    with open(outdir / "injection_parameters.json", "w") as f:
        json.dump(injection_parameters, f)

    waveform_arguments = dict(
        waveform_approximant="TaylorF2Ecc",
        reference_frequency=minimum_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    aligned_waveform_arguments = dict(
        waveform_approximant="TaylorF2",
        reference_frequency=minimum_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    # Create the waveform_generator using the LAL eccentric black hole no spins
    # source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=lal_eccentric_binary_black_hole,
        parameters=injection_parameters,
        waveform_arguments=waveform_arguments,
        # parameter_conversion=conversion_function
    )

    aligned_waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=injection_parameters,
        waveform_arguments=aligned_waveform_arguments,
        # parameter_conversion=conversion_function,
    )

    wf = waveform_generator.frequency_domain_strain(
        parameters=injection_parameters,
    )

    plt.figure(figsize=(10, 6))
    plt.loglog(
        waveform_generator.frequency_array,
        np.abs(wf["plus"]),
        label="Eccentric TaylorF2",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Strain")
    plt.axvline(
        maximum_frequency, color="red", linestyle="--", label="Maximum Frequency"
    )
    plt.title("Eccentric Binary Black Hole Waveform")
    plt.legend()
    plt.grid()
    plt.savefig(outdir / "waveform_plot_updated.png")

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = maximum_frequency
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] + 2 - duration,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    # Now we set up the priors on each of the binary parameters.
    priors = bilby.core.prior.PriorDict()
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        name="chirp_mass",
        minimum=20,
        maximum=40,
    )
    priors["mass_ratio"] = bilby.core.prior.Uniform(
        name="mass_ratio",
        minimum=0.2,
        maximum=1,
    )
    priors["eccentricity"] = bilby.core.prior.Uniform(
        name="eccentricity", latex_label="$e$", minimum=0.0, maximum=0.4
    )

    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        name="luminosity_distance", minimum=1e2, maximum=2e3
    )
    priors["dec"] = bilby.core.prior.Cosine(name="dec")
    priors["ra"] = bilby.core.prior.Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    priors["theta_jn"] = bilby.core.prior.Sine(name="theta_jn")
    priors["psi"] = bilby.core.prior.Uniform(
        name="psi", minimum=0, maximum=np.pi, boundary="periodic"
    )
    priors["phase"] = bilby.core.prior.Uniform(
        name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    priors["geocent_time"] = bilby.core.prior.Uniform(
        injection_parameters["geocent_time"] - 0.1,
        injection_parameters["geocent_time"] + 0.1,
        name="geocent_time",
        unit="s",
    )
    priors["chi_1"] = bilby.core.prior.Uniform(
        minimum=-0.99, maximum=0.99, name="chi_1", latex_label="$\chi_1$"
    )
    priors["chi_2"] = bilby.core.prior.Uniform(
        minimum=-0.99, maximum=0.99, name="chi_2", latex_label="$\chi_2$"
    )

    aligned_priors = copy.deepcopy(priors)
    aligned_priors.pop("eccentricity")
    poppy_priors = copy.deepcopy(priors)

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        time_marginalization=False,
        distance_marginalization=True,
        phase_marginalization=True,
    )

    aligned_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=aligned_waveform_generator,
        priors=aligned_priors,
        time_marginalization=False,
        distance_marginalization=True,
        phase_marginalization=True,
    )

    poppy_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=copy.deepcopy(waveform_generator),
        priors=poppy_priors,
        time_marginalization=False,
        distance_marginalization=True,
        phase_marginalization=True,
    )

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        # sampler="nessai",
        # nlive=1000,
        # reset_flow=8,
        # volume_fraction=0.98,
        sampler="dynesty",
        nlive=1000,
        naccept=60,
        sample="acceptance-walk",
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        result_class=bilby.gw.result.CBCResult,
        n_pool=n_pool,
    )

    result.plot_corner()

    result_aligned = bilby.run_sampler(
        likelihood=aligned_likelihood,
        priors=aligned_priors,
        # sampler="nessai",
        # nlive=1000,
        # reset_flow=8,
        # volume_fraction=0.98,
        sampler="dynesty",
        nlive=1000,
        naccept=60,
        sample="acceptance-walk",
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label + "_aligned",
        result_class=bilby.gw.result.CBCResult,
        waveform_generator=aligned_waveform_generator,
        n_pool=n_pool,
    )

    # And finally we make some plots of the output posteriors.
    result_aligned.plot_corner()

    # Run poppy

    result_poppy = bilby.run_sampler(
        likelihood=poppy_likelihood,
        priors=poppy_priors,
        sampler="poppy",
        injection_parameters=injection_parameters,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        outdir=str(outdir),
        label=label + "_poppy_Exp3",
        result_class=bilby.gw.result.CBCResult,
        n_pool=n_pool,
        initial_result_file=str(outdir / f"{label}_aligned_result.json"),
        sample_from_prior=[
            "chirp_mass",
            "mass_ratio",
            "chi_1",
            "chi_2",
            "geocent_time",
        ],
        n_samples=1000,
        plot=True,
        n_final_samples=5000,
        sample_kwargs=dict(
            sampler="minipcn_smc",
            adaptive=True,
            target_efficiency=0.66,
            sampler_kwargs=dict(n_steps=500),
        ),
    )

    result_poppy.plot_corner()


if __name__ == "__main__":
    n_pool = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(n_pool=n_pool)
