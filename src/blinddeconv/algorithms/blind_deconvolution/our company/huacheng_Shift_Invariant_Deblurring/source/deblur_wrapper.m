function [I_latent, k] = deblur_wrapper(I, params)
% A functional wrapper around deblur.m to make it callable from MATLAB Engine.

I = im2double(I);
if ndims(I) == 3
    I = I(:,:,1);
end

lambda_coarse = params.lambda_coarse;
ganma = params.gamma;
final_lambda = params.final_lambda;
ratio = params.ratio;
ks = params.ks;
sigma = params.gaussian_sigma;
edgetaper_iters = params.edgetaper_iters;
perona_iter = params.perona_iter;
shock_iter = params.shock_iter;
dt = params.shock_dt;
h = params.shock_h;

if numel(ratio) ~= numel(ks)
    error('params.ratio and params.ks must have the same length.');
end

Im = imresize(I, 1/ratio(1));
I_latent = Im;
tau_r = 0;
tau_s = 0;

for i = 1:numel(ratio)
    Im = imresize(I, 1/ratio(i));
    [xim,yim] = size(Im);
    [xil,yil] = size(I_latent);
    minxi = min([xim,xil]);
    minyi = min([yim,yil]);
    Im = Im(1:minxi,1:minyi);
    I_latent = I_latent(1:minxi,1:minyi);

    ksize = 2*ks(i)+1;
    kernel_gaus = fspecial('gaussian', ksize, sigma);

    Im = padarray(Im, [1 1]*ks(i), 'replicate', 'both');
    I_latent = padarray(I_latent, [1 1]*ks(i), 'replicate', 'both');
    for j = 1:edgetaper_iters
        Im = edgetaper(Im, kernel_gaus);
        I_latent = edgetaper(I_latent, kernel_gaus);
    end

    I_sh = perona_malik(I_latent, perona_iter);
    I_sh = shock_filter(I_sh, shock_iter, dt, h);

    [xim,yim] = size(Im);
    Im_x = [Im(1:xim-1,:)-Im(2:xim,:);Im(xim,:)-Im(1,:)];
    Im_y = [Im(:,1:yim-1)-Im(:,2:yim),Im(:,yim)-Im(:,1)];
    Ish_x = [I_sh(1:xim-1,:)-I_sh(2:xim,:);I_sh(xim,:)-I_sh(1,:)];
    Ish_y = [I_sh(:,1:yim-1)-I_sh(:,2:yim),I_sh(:,yim)-I_sh(:,1)];

    [M, tau_r] = M_compute(Im_x, Im_y, ks, i, tau_r);
    [H, tau_s] = H_compute(Ish_x, Ish_y, M, ks, i, tau_s);

    Is_x = Ish_x .* H;
    Is_y = Ish_y .* H;

    [k, I_latent] = coarse_kernel_est(Is_x, Is_y, Im_x, Im_y, Im, ksize, lambda_coarse, ganma);

    if i < numel(ratio)
        I_latent = I_latent(ks(i)+1:xim-ks(i), ks(i)+1:yim-ks(i));
        I_latent = imresize(I_latent, ratio(i)/ratio(i+1), 'bicubic');
    end
end

I_latent = multi_deriv_deconv(Im, k, final_lambda);
I_latent = I_latent(ks(end)+1:xim-ks(end), ks(end)+1:yim-ks(end));

end

