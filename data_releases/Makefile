# File for making data releases
include config.mk

# Data release for first paper
$(FIRST_PAPER_DATA_RELEASE_DIR): $(FIRST_PAPER_LOCAL_RESULTS)
	@echo "Making data release for first paper"
	python fetch_results_from_yaml.py $(FIRST_PAPER_LOCAL_RESULTS) --outdir $(FIRST_PAPER_DATA_RELEASE_DIR) --output-yaml results_first_paper.yaml --overwrite --verbose

# Zip up the data release
first_paper_data_release.zip : $(FIRST_PAPER_DATA_RELEASE_DIR)
	@echo "Zipping data release for first paper"
	zip -r $@ $(FIRST_PAPER_DATA_RELEASE_DIR)
