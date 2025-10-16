include config.mk

$(INJECTION_PARAMS_PATH)/GW150914_parameters.hdf5:
	python generate_injection_parameters.py \
		--n-samples 1 \
		--prior-file GW150914-injection.prior \
		--injection-file $(INJECTION_PARAMS_PATH)/GW150914_parameters.hdf5 \
		--waveform IMRPhenomXPHM

$(INJECTION_PARAMS_PATH)/HM_injection_q4_parameters.hdf5:
	python generate_injection_parameters.py \
		--n-samples 1 \
		--prior-file q4-HM-injection.prior \
		--injection-file $(INJECTION_PARAMS_PATH)/HM_injection_q4_parameters.hdf5 \
		--waveform IMRPhenomXPHM


GW150914_XPHM_data: $(INJECTION_PARAMS_PATH)/GW150914_parameters.hdf5
	python generate_data.py \
		--start-time 1364342474 \
		--frame-duration 4096 \
		--inject \
		--injection-file $< \
		--waveform-approximant IMRPhenomXPHM \
		--channel-name MockData \
		--n-injections 1 \
		--outdir $(FRAME_FILES_PATH)/GW150914_XPHM_data


injection_q4_HM_IMRPhenomXPHM_data: $(INJECTION_PARAMS_PATH)/HM_injection_q4_parameters.hdf5
	python generate_data.py \
		--start-time 1364342474 \
		--frame-duration 4096 \
		--inject \
		--injection-file $< \
		--waveform-approximant IMRPhenomXPHM \
		--channel-name MockData \
		--n-injections 1 \
		--outdir $(FRAME_FILES_PATH)/HM_injection_q4_IMRPhenomXPHM_data

.PHONY: first_paper_data
first_paper_data: GW150914_XPHM_data HM_injection_q4_IMRPhenomXPHM_data
