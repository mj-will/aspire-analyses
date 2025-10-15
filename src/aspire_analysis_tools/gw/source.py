import bilby

def eccentric_binary_black_hole_aligned_spins(
    frequency_array,
    mass_1,
    mass_2,
    a_1,
    a_2,
    eccentricity,
    luminosity_distance,
    theta_jn,
    phase,
    **kwargs
):
    """Eccentric binary black hole source with aligned spins.
    
    Defaults to TaylorF2.
    """
    waveform_kwargs = dict(
        waveform_approximant='TaylorF2', reference_frequency=10.0,
        minimum_frequency=10.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    return bilby.gw.source._base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2, a_1=a_1, a_2=a_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        eccentricity=eccentricity, **waveform_kwargs)
