include config.mk

.PHONY: pp-plot
pp-plot:
	python pp_plot.py --result-dir $(PP_TEST_OUTDIR)/final_result/ --outdir pp_plot/ --injection-file $(PP_TEST_INJECTION_FILE) --credible-levels-file $(PP_TEST_CACHE_DIR)/pp_test_credible_levels.hdf5 --figure-format pdf