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

function [u_end,k_end,u_four_MHDM,k_four_MHDM,its] = blind_deconvolution_MHDM(f,f_four,lambda_0,mu_0,r,s,tol,stopping,maxits,indices)
%performs the blind deconvolution MHDM for noisy data until the discrepancy
%principle is met
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: blurred and noisy image f, initial regularization parameters
%lambda_0, mu_0, powers r and s for sobolev norms in Fourier space,
%tolerance tol for small negative values due to numerical issues, value
%stopping to decide if discrepancy principle is met, maximal number of
%iteration maxits before terminating the algorithm 
%table of indices for which anti-symmetry condition is enforced
%Output: final iterates u_end and k_end, all Fourier transforms of the
%iterates u_four_MHDM and k_four_MHDM and number of iterates

l2= @(u) sqrt(sum(abs(u).^2,'all'));
lambda = lambda_0;
mu = mu_0;


%compute initial iterates
[u,k]= MHDM_initial(f_four,lambda,mu,r,s);
u_four_MHDM(:,:,1) = u;
k_four_MHDM(:,:,1) = k;
its = 1;
residual = l2(f-real(ifft2(u_four_MHDM(:,:,end).*k_four_MHDM(:,:,end))));

%iteration until stopping criterion
while residual > stopping && its <= maxits
    mu = mu/4;
    lambda = lambda/4;
    [u,k] = MHDM_step(u_four_MHDM(:,:,end),k_four_MHDM(:,:,end),f_four,lambda,mu,r,s,tol,indices);
    its = its+1;
    u_four_MHDM(:,:,its) = u+u_four_MHDM(:,:,its-1);
    k_four_MHDM(:,:,its) = k+k_four_MHDM(:,:,its-1);
    residual = l2(f-real(ifft2(u_four_MHDM(:,:,end).*k_four_MHDM(:,:,end))));
end
u_end = real(ifft2(u_four_MHDM(:,:,end)));
k_end = real(otf2psf(k_four_MHDM(:,:,end)));

end