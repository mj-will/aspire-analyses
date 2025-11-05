# File for making data releases
include config.mk

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GWDATA_DIR := $(abspath $(MAKEFILE_DIR)/../gw/data)
GWDATA_config := $(GWDATA_DIR)/config.mk
PP_TEST_DIR := $(abspath $(MAKEFILE_DIR)/../gw/pp_tests)
FIRST_PAPER_PP_TEST_DIR := $(PP_TEST_DIR)/$(FIRST_PAPER_PP_TEST_OUTDIR)

ZIP_COMMAND := zip -r


# Include the GW data config to get INJECTION_PARAMS_PATH and FRAME_FILES_PATH
include $(GWDATA_config)


.PHONY: first_paper_data_release
first_paper_data_release: $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results $(FIRST_PAPER_DATA_RELEASE_DIR)/data
	@echo "Data release for first paper complete"

# Core results data release
.PHONY: first_paper_data_release_core_results
first_paper_data_release_core_results: $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results

$(FIRST_PAPER_DATA_RELEASE_DIR)/core_results:
	@echo "Making data release for first paper core results"
	mkdir -p $@
	python fetch_results_from_yaml.py $(FIRST_PAPER_LOCAL_RESULTS) --outdir $@ --output-yaml results_first_paper.yaml --overwrite --verbose

# Results data release
.PHONY: first_paper_data_release_data
first_paper_data_release_data: $(FIRST_PAPER_DATA_RELEASE_DIR)/data

$(FIRST_PAPER_DATA_RELEASE_DIR)/data:
	@echo "Making data release for first paper data"
	mkdir -p $@
	cp -r $(GWDATA_DIR)/$(INJECTION_PARAMS_PATH) $@/injection_parameters
	cp -r $(GWDATA_DIR)/$(FRAME_FILES_PATH) $@/frame_files


# Additional results release
.PHONY: first_paper_data_release_additional_results
first_paper_data_release_additional_results: $(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results
$(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results:
	@echo "Making data release for first paper additional results"
	mkdir -p $@
	@echo "Linking P-P test results from $(FIRST_PAPER_PP_TEST_DIR)"
	rm -f $@/pp_test_results
	ln -s $(FIRST_PAPER_PP_TEST_DIR)/final_result $@/pp_test_results

# Zip up the data release
.PHONY: first_paper_data_release_core_results_zip
first_paper_data_release_core_results_zip: $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results.zip

$(FIRST_PAPER_DATA_RELEASE_DIR)/core_results.zip : $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results
	@echo "Zipping data release for first paper"
	cd $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results && $(ZIP_COMMAND) $(abspath $@) .

.PHONY: first_paper_data_release_data_zip
first_paper_data_release_data_zip: $(FIRST_PAPER_DATA_RELEASE_DIR)/data.zip

$(FIRST_PAPER_DATA_RELEASE_DIR)/data.zip : $(FIRST_PAPER_DATA_RELEASE_DIR)/data
	@echo "Zipping data release for first paper data"
	cd $(FIRST_PAPER_DATA_RELEASE_DIR)/data && $(ZIP_COMMAND) $(abspath $@) .

.PHONY: first_paper_data_release_additional_results_zip
first_paper_data_release_additional_results_zip: $(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results.zip

$(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results.zip : $(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results
	@echo "Zipping additional results data release for first paper"
	cd $(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results && $(ZIP_COMMAND) $(abspath $@) .

.PHONY: first_paper_data_release_zip
first_paper_data_release_zip: first_paper_data_release_core_results_zip first_paper_data_release_data_zip first_paper_data_release_additional_results_zip
	@echo "Creating final zipped data release for first paper"
	zip -j first_paper_data_release.zip $(FIRST_PAPER_DATA_RELEASE_DIR)/*.zip


# Safely extract results. Fails if files already exist.
extract_results:
	@echo "Extracting first paper data release zip"
	unzip -n $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results.zip -d $(FIRST_PAPER_DATA_RELEASE_DIR)/core_results
	unzip -n $(FIRST_PAPER_DATA_RELEASE_DIR)/data.zip -d $(FIRST_PAPER_DATA_RELEASE_DIR)/data
	unzip -n $(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results.zip -d $(FIRST_PAPER_DATA_RELEASE_DIR)/additional_results

.PHONY: clean
clean:
	rm -rf $(FIRST_PAPER_DATA_RELEASE_DIR)
	rm -f first_paper_data_release.zip