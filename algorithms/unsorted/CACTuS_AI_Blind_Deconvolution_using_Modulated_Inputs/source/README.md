# Blind Deconvolution using Modulated Inputs

This page provides software to generate the figures and the experiments in the paper [Blind Deconvolution using Modulated Inputs](https://arxiv.org/pdf/1811.08453.pdf). We also provide software created by other groups which is necessary to run our own code. 

## Required Toolboxes

The following toolboxes are required to run the MATLAB scripts below. The paths to the associated directories need to be provided in the script.

- Noiselet Toolbox

## Matlab scripts

We provide the Matlab scripts that generate the figures, as well as a test file that demonstrates large scale blind deconvolution using modulated inputs.

- Script to generate Figure 4 (phase transition M vs K): Figure4a.m
  
  - Requires Noiselet Toolbox
  - Requires: adjcA1d.m
  - Requires: concat1d.m
  
- Script to generate Figure 4 (phase transition N vs K): Figure4f.m
  
  - Requires Noiselet Toolbox
  - Requires: adjcA1d.m
  - Requires: concat1d.m
  
- Script to generate Figures5 (image deblurring): Figure5.m
  - Requires: Noiselet Toolbox
  - Requires: adjcA2d.m
  - Image data: concat2d.m

- Script to generate Figure 6  (recovery in the presence of noise): Figure6_left.m
  - Requires: Noiselet Toolbox
  - Requires: adjcA1d.m
  - Requires: concat1d.m

- Script to generate Figure 6  (oversampling): Figure6_right.m

  - Requires: Noiselet Toolbox
  - Requires: adjcA1d.m

  - Requires: concat1d.m

    

## References

The Noiselet toolbox is by Professor Justin Romberg, if you use either of these files in your personal work, please remember to cite this reference.
