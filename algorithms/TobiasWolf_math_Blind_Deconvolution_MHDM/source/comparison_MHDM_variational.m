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

%creates 144 noisy and blurry test images and performs blind deconvolution
%MHDM and grid search on single-step variational blind deconvolution; results are evaluated and saved;
%Attention! Long runtime! Large files are saved!




close all
clear
clc
rng('default')
l2= @(u) sqrt(sum(abs(u).^2,'all'));
%%
%create test images

u_true_arr = {};
f_blur_arr = {};
f_arr = {};
noise_arr = {};
delta_arr = {};
ker_arr = {};

num_Ims = 16;
for k = 1:num_Ims

    %load images
    u_true= im2double((imread(strcat('test_images/im',num2str(k),'.jpg'))));
    if (size(u_true,3) > 1)
        u_true = rgb2gray(u_true);
    end
    [m,n] = size(u_true);


    %blur images
    kernels = [];
    kernels(:,:,1) = fspecial('gaussian',[m,n],8);
    kernels(:,:,2) = (5*fspecial('gaussian',[m,n],4)+ fspecial('gaussian',[m,n],5)+fspecial('gaussian',[m,n],1)+fspecial('gaussian',[m,n],3))/8;
    kernels(:,:,3) = fspecial('gaussian',[m,n],2);

    noise_levels = [4e-4,4e-5,3e-3];
    num_noises = length(noise_levels);
    num_kernels = length(kernels(1,1,:));


    for l = 1:length(kernels(1,1,:))
        f_blur = real(ifft2(psf2otf(kernels(:,:,l), [m n]).*fft2(u_true)));
        for ll = 1:num_noises
            f =  imnoise(f_blur,'gaussian',0,noise_levels(ll));
            noise=f-f_blur;
            delta = l2(noise);

            u_true_arr = [u_true_arr,u_true];
            ker_arr = [ker_arr, kernels(:,:,l)];
            f_blur_arr = [f_blur_arr,f_blur];
            f_arr = [f_arr,f];
            noise_arr = [noise_arr,noise];
            delta_arr = [delta_arr,delta];
        end
    end

end

clear ll k l f f_blur kernels noise m n u_true delta

%%
num_tests = length(f_arr);

tol = 1e-10;
r = 1;
s =1e-1;
tau = 1.001;
maxits = 30;
minits = 10;

u_end_MHDM_arr = {};
k_end_MHDM_arr = {};
u_four_MHDM_arr = {};
k_four_MHDM_arr = {};

u_end_single_arr = {};
k_end_single_arr = {};
u_four_single_arr = {};
k_four_single_arr = {};

for l = 1:num_tests

    u_true = cell2mat(u_true_arr(l));
    kernel = cell2mat(ker_arr(l));
    delta = cell2mat(delta_arr(l));
    f = cell2mat(f_arr(l));
    [m,n] = size(f);
    f_four = fft2(f);
    stopping = tau*delta;
    [rr,c] = ismember(f_four,conj(f_four));
    c =reshape(c,m*n,1);
    M = [c, c(c)];
    sortedM = sort(M, 2);
    [~, uniqueIdx] = unique(sortedM, 'rows', 'stable');
    indices = M(uniqueIdx, :);

    %blind MHDM
    [lambda_0,mu_0,u_end,k_end,u_four_MHDM,k_four_MHDM,its] = find_parameters_MHDM(u_true,f,f_four,r,s,tol,stopping,maxits,minits,indices);
    psnr_u_MHDM(l) = psnr(u_end,u_true);
    ssim_u_MHDM(l) = ssim(u_end,u_true);
    err_ker_MHDM(l) = l2(k_end-kernel)./l2(kernel);

    lambda_0_MHDM(l) = lambda_0;
    mu_0_MHDM(l) = mu_0;
    its_MHDM(l) = its;

    u_end_MHDM_arr = [u_end_MHDM_arr,u_end];
    k_end_MHDM_arr = [k_end_MHDM_arr,k_end];
    u_four_MHDM_arr = [u_four_MHDM_arr,u_four_MHDM];
    k_four_MHDM_arr = [k_four_MHDM_arr,k_four_MHDM];

    %blind grid search
    [lambda_0,mu_0,u_end,k_end,u_four_single,k_four_single,its] = find_parameters_single(u_true,f,f_four,r,s,stopping,maxits,minits);

    psnr_u_single(l) = psnr(u_end,u_true);
    ssim_u_single(l) = ssim(u_end,u_true);
    err_ker_single(l) = l2(k_end-kernel)./l2(kernel);

    lambda_0_single(l) = lambda_0;
    mu_0_single(l) = mu_0;
    its_single(l) = its;

    u_end_single_arr = [u_end_single_arr,u_end];
    k_end_single_arr = [k_end_single_arr,k_end];
    u_four_single_arr = [u_four_single_arr,u_four_single];
    k_four_single_arr = [k_four_single_arr,k_four_single];



end
%%
save('data_comp.mat')

