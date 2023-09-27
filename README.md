# CFG_Rescale_webui
Adds a CFG rescale option to A1111 webui for vpred models. Implements features described in https://arxiv.org/pdf/2305.08891.pdf

When installed, you will see a CFG slider below appear below the seed selection.

The recommended settings from the paper is 0.7 CFG rescale. The paper recommends using it at 7.5 CFG, but I find that the default 7.0 CFG works just fine as well.
If you notice your images are dull or muddy, try setting CFG Rescale to 0.5 and/or turning on color fix.

# Extra Features
Auto color fix: CFG rescale can result in the color palette of the image becoming smaller, resulting in muddy or brown images. Turning this on attempts to return the image colors to full range again, making it appear more vibrant and natural.
