# CFG_Rescale_webui
Adds a CFG rescale option to A1111 webui for vpred models. implements features described in https://arxiv.org/pdf/2305.08891.pdf

When installed, you will see a CFG slider below appear below the seed selection.

# Extra Features
This extension also implements trailing schedule for DDIM sampler as described in the paper. When enabled, the DDIM sampler will be overriden with a trailing schedule instead of a linear one.

TODO: Rescale betas from the paper. Trailing sampler for other samplers.
