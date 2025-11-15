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


function[val,dict] = check_ratio_MHDM(u_true,f,f_four,ratio,r,s,tol,stopping,maxits,minits,indices,dict)
% computes psnr value of the image obtained via blind deconvolution MHDM for a given  ratio of regularization parameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: true image u_true, blurred and noisy image f and its Fourier
%transform f_four, ratio of regularization parameters, powers r and s for
%sobolev norms in Fourier space, tolerance tol for small negative values due to numerical issues,
%stopping threshold stopping for MHDM, maximum number of MHDM iterations
%maxits, minimum numbers of MHDM iterations minits, indices for which
%anti-symmetry condition is enforced, dictionary dict to check if values
%have been computed befor
%Output: psnr value val, updated dictionary dict to check if values


check = find(dict(1,:)== ratio);
if isempty(check)
    its = 0;
    factor = 1;
    mu_0 = sqrt(ratio);
    lambda_0 = mu_0/ratio;
    %ensure minimum number of MHDM iterations
    while its < minits
        mu_0 = mu_0*factor;
        lambda_0 = lambda_0*factor;
        [u_end,k_end,u_four_MHDM,k_four_MHDM,its] =blind_deconvolution_MHDM(f,f_four,lambda_0,mu_0,r,s,tol,stopping,maxits,indices);
        factor = 5*factor;
    end
    val = psnr(u_end,u_true);
    dict = [dict, [ratio; val]];
else
    %use value from dict if already computed
    val = dict(2,check);
end