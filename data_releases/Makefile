# File for making data releases
include config.mk

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GWDATA_DIR := $(abspath $(MAKEFILE_DIR)/../gw/data)
GWDATA_config := $(GWDATA_DIR)/config.mk

# Include the GW data config to get INJECTION_PARAMS_PATH and FRAME_FILES_PATH
include $(GWDATA_config)


.PHONY: first_paper_data_release
first_paper_data_release: $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results $(FIRST_PAPER_DATA_RELEASE_DIR)/data
	@echo "Data release for first paper complete"

.PHONY: first_paper_data_release_core_results
first_paper_data_release_core_results: $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results

$(FIRST_PAPER_DATA_RELEASE_DIR)/core_results:
	@echo "Making data release for first paper core results"
	mkdir -p $@
	python fetch_results_from_yaml.py $(FIRST_PAPER_LOCAL_RESULTS) --outdir $@ --output-yaml results_first_paper.yaml --overwrite --verbose

.PHONY: first_paper_data_release_data
first_paper_data_release_data: $(FIRST_PAPER_DATA_RELEASE_DIR)/data

$(FIRST_PAPER_DATA_RELEASE_DIR)/data:
	@echo "Making data release for first paper data"
	mkdir -p $@
	cp -r $(GWDATA_DIR)/$(INJECTION_PARAMS_PATH) $@/injection_parameters
	cp -r $(GWDATA_DIR)/$(FRAME_FILES_PATH) $@/frame_files

# Zip up the data release
first_paper_data_release_core_results.zip : $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results
	@echo "Zipping data release for first paper"
	zip -r $@ $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results

first_paper_data_release_data.zip : $(FIRST_PAPER_DATA_RELEASE_DIR)/data
	@echo "Zipping data release for first paper data"
	zip -r $@ $(FIRST_PAPER_DATA_RELEASE_DIR)/data


.PHONY: clean
clean:
	rm -rf $(FIRST_PAPER_DATA_RELEASE_DIR)
	rm -f first_paper_data_release.zip