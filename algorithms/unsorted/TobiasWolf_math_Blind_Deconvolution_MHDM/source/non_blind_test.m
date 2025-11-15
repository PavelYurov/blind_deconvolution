% Copyright (C) 2025 Tobias Wolf <tobias.wolf@aau.at>

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.

% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%performs non-blind deconvolution MHDM with 1000 guessed kernels for the
%images from data_comp.mat; evaluates and saves the results;
%Attention! Long runtime! Large files are saved!


close all
clear
clc

%load test images and results from blind-deconvolution
dir = pwd;
load(dir+"\data_comp.mat");

%%
u_MHDM_guess_psnr_arr = {};
psnr_guess = zeros(num_tests,1000);

u_MHDM_guess_ssim_arr = {};
ssim_guess = zeros(num_tests,1000);

psnr_guess_end = zeros(1,num_tests);
ssim_guess_end = zeros(1,num_tests);

for l = 1:num_tests
    tic;
    f = cell2mat(f_arr(l));
    u_true = cell2mat(u_true_arr(l));
    lambda_0 = lambda_0_MHDM(l);
    delta = cell2mat(delta_arr(l));
    stopping  = tau*delta;
    [m,n] = size(f);
    sigmas = linspace(1,12,1000);


    psnr_guess_end_temp = 0;
    psnr_try = -1;

    ssim_guess_end_temp = 0;
    ssim_try = -1;

    for ll = 1:length(sigmas)
        kernel_guess = fspecial('gaussian',[m,n],sigmas(ll));
        k_four = psf2otf(kernel_guess,[m,n]);
        [u_MHDM_guess,u_four_MHDM_guess,its] = non_blind_MHDM(f,k_four,lambda_0,r,stopping);


        %find best performing image with repect to PSNR
        psnr_try = psnr(u_MHDM_guess,u_true);
        if psnr_try >= psnr_guess_end_temp
            psnr_guess_end_temp = psnr_try;
            u_MHDM_guess_psnr_end = u_MHDM_guess;
        end



        %find best performing image with repect to SSIM
        ssim_try = ssim(u_MHDM_guess,u_true);
        if ssim_try >= ssim_guess_end_temp
            ssim_guess_end_temp = ssim_try;
            u_MHDM_guess_ssim_end = u_MHDM_guess;
        end

    end
    u_MHDM_guess_psnr_arr{l} = u_MHDM_guess_psnr_end;
    u_MHDM_guess_ssim_arr{l} = u_MHDM_guess_ssim_end;
    psnr_guess_end(l) = psnr_guess_end_temp;
    ssim_guess_end(l) = ssim_guess_end_temp;
    display([l,toc]);
end

%%
save(dir+"data_non_blind.mat");