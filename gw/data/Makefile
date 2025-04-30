

injection_parameters/GW150914_parameters.hdf5:
	python generate_injection_parameter.py \
		--n-samples 1 \
		--prior-file GW150914-injection.prior \
		--injection-file injection_parameters/GW150914_parameters.hdf5 \
		--waveform IMRPhenomXPHM

injection_parameters/eccentric_injection_q3_parameters.hdf5:
	python generate_injection_parameter.py \
		--n-samples 1 \
		--prior-file q3-eccentric-injection.prior \
		--injection-file injection_parameters/eccentric_injection_q3_parameters.hdf5 \
		--reference-frequency 20 \
		--minimum-frequency 20 \
		--waveform SEOBNRv5EHM


GW150914_XPHM_data: injection_parameters/GW150914_parameters.hdf5
	python generate_data.py \
		--start-time 1364342474 \
		--frame-duration 4096 \
		--inject \
		--injection-file $< \
		--waveform-approximant IMRPhenomXPHM \
		--channel-name PoppyMockData \
		--n-injections 1 \
		--outdir frame_files/GW150914_XPHM_data

GW150914_XHM_data: injection_parameters/GW150914_parameters.hdf5
	python generate_data.py \
		--start-time 1364342474 \
		--frame-duration 4096 \
		--inject \
		--injection-file $< \
		--waveform-approximant IMRPhenomXHM \
		--channel-name PoppyMockData \
		--n-injections 1 \
		--outdir frame_files/GW150914_XHM_data \
		--aligned-spins

injection_q3_SEOBNRv5EHM_data: injection_parameters/eccentric_injection_q3_parameters.hdf5
	python generate_data.py \
		--start-time 1364342474 \
		--frame-duration 4096 \
		--inject \
		--injection-file $< \
		--waveform-approximant SEOBNRv5EHM \
		--channel-name PoppyMockData \
		--n-injections 1 \
		--reference-frequency 20 \
		--minimum-frequency 20 \
		--outdir frame_files/eccentric_injection_q3_SEOBNRv5EHM_data
