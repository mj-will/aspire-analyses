## BBH P-P tests

### Injections

Aligned spin injections were generated using 

```
bilby_pipe_create_injection_file -f aligned_spin_injections.json aligned_spin.prior --extension json -n 100 -s 1234
```


### Analyses

In these analyses, `poppy` is used as standard sampler rather than a post-processing tool.
