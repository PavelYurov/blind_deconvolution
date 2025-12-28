function [J, PSF] = kanugaba_deconvwnr(I, LEN, THETA, noise_var, nsr_override)
%KANUGABA_DECONVWNR Wiener deconvolution with motion PSF.
%   I must be a 2D grayscale image in [0,1] (double).
%   If nsr_override is NaN, NSR is estimated as noise_var/var(I(:)).

I = im2double(I);

PSF = fspecial('motion', LEN, THETA);

if isnan(nsr_override)
    estimated_nsr = noise_var / var(I(:));
else
    estimated_nsr = nsr_override;
end

J = deconvwnr(I, PSF, estimated_nsr);
end

