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


function[lambda_0,mu_0,u_end,k_end,u_four_single,k_four_single,its] = find_parameters_single(u_true,f,f_four,r,s,stopping,maxits,minits)
% finds optimal initial parameters for single-step variational blind deconvolution with grid search and returns
% the parameters and reconstructions obtained

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: true image u_true, blurred and noisy image f and its Fourier
%transform f_four, powers r and s for sobolev norms in Fourier space,
%stopping threshold stopping for grid search, maximum number of grid search iterations
%maxits, minimum numbers of grid search iterations minits
%Output: initial parameters lambda_0 and mu_0, final iterates u_enda and k_end, Fourier transforms u_four and k_four of the reconstructed kernel and image



ratio_0 =1e9; %mu/lambda


%compute middle psnr
dict = [0;0];
%compute middle psnr
ratio_mid = ratio_0;
[val_mid,dict] = check_ratio_single(u_true,f,f_four,ratio_mid,r,s,stopping,maxits,minits,dict);

%compute left psnr
ratio_left = ratio_0/2;
[val_left,dict] = check_ratio_single(u_true,f,f_four,ratio_left,r,s,stopping,maxits,minits,dict);

%compute right psnr
ratio_right = ratio_0*2;
[val_right,dict] = check_ratio_single(u_true,f,f_four,ratio_right,r,s,stopping,maxits,minits,dict);

%find optimal parameters
while true
    max_val = max([val_left,val_right,val_mid]);
    if max_val == val_left
        ratio_right = ratio_mid;
        val_right = val_mid;
        ratio_mid = ratio_left;
        val_mid = val_left;

        ratio_left = ratio_left-(ratio_right-ratio_left)/2;
        [val_left,dict] = check_ratio_single(u_true,f,f_four,ratio_left,r,s,stopping,maxits,minits,dict);



    elseif max_val == val_right
        ratio_left = ratio_mid;
        val_left = val_mid;
        ratio_mid = ratio_right;
        val_mid =val_right;

        ratio_right = ratio_right+(ratio_right-ratio_left)/2;
        [val_right,dict] = check_ratio_single(u_true,f,f_four,ratio_right,r,s,stopping,maxits,minits,dict);

    else
        if val_left <= val_right
            ratio_left = (ratio_left+ratio_mid)/2;
            [val_left,dict] = check_ratio_single(u_true,f,f_four,ratio_left,r,s,stopping,maxits,minits,dict);
        else
            ratio_right= (ratio_right+ratio_mid)/2;
            [val_right,dict] = check_ratio_single(u_true,f,f_four,ratio_right,r,s,stopping,maxits,minits,dict);
        end
    end

    if abs(ratio_mid - ratio_right)/ratio_mid < 2*eps || abs(ratio_mid - ratio_left)/ratio_mid < 2*eps
        ratio_mid = (ratio_right-ratio_left)/2;
        [val_mid,dict] = check_ratio_single(u_true,f,f_four,ratio_mid,r,s,stopping,maxits,minits,dict);
    end
    if abs(ratio_left/ratio_right) > 1-0.5^9
        break;
    end
end


%compute outputs
its = 0;
factor_mid = 1;
mu_0 = sqrt(ratio_mid);
lambda_0 = mu_0/ratio_mid;
while its < minits
    mu_0 = mu_0*factor_mid;
    lambda_0 = lambda_0*factor_mid;
    [u_end,k_end,u_four_single,k_four_single,its] =blind_deconvolution_grid(f,f_four,lambda_0,mu_0,r,s,stopping,maxits);
    factor_mid = 5*factor_mid;
end











