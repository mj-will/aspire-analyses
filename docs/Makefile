MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TOY_EXAMPLES_SRC := $(abspath $(MAKEFILE_DIR)/../toy_examples)
TOY_EXAMPLES_DST := $(abspath $(MAKEFILE_DIR)/toy_examples)

FIRST_PAPER_SRC := $(abspath $(MAKEFILE_DIR)/../gw/plots/first_paper)
FIRST_PAPER_DST := $(abspath $(MAKEFILE_DIR)/aspire_paper)

.PHONY: all
all: build

.PHONY: sync_toy_examples
sync_toy_examples:
	mkdir -p "$(TOY_EXAMPLES_DST)"
	rm -f "$(TOY_EXAMPLES_DST)"/*.ipynb
	for nb in "$(TOY_EXAMPLES_SRC)"/*.ipynb; do \
		[ -e "$$nb" ] || continue; \
		cp "$$nb" "$(TOY_EXAMPLES_DST)/"; \
	done

.PHONY: sync_first_paper_plots
sync_first_paper_plots:
	mkdir -p "$(FIRST_PAPER_DST)"
	rm -f "$(FIRST_PAPER_DST)"/*.ipynb
	cp "$(FIRST_PAPER_SRC)/plots.style" "$(FIRST_PAPER_DST)/"
	cp "$(FIRST_PAPER_SRC)/config.mk" "$(FIRST_PAPER_DST)/"
	for nb in "$(FIRST_PAPER_SRC)"/*.ipynb; do \
		[ -e "$$nb" ] || continue; \
		cp "$$nb" "$(FIRST_PAPER_DST)/"; \
	done
	

.PHONY: build
build: sync_toy_examples sync_first_paper_plots
	jupyter-book build .

.PHONY: clean
clean:
	rm -rf $(MAKEFILE_DIR)/_build/html
	rm -rf $(MAKEFILE_DIR)/_build/latex
	rm -f $(TOY_EXAMPLES_DST)/*.ipynb
	rm -f $(FIRST_PAPER_DST)/*.ipynb
