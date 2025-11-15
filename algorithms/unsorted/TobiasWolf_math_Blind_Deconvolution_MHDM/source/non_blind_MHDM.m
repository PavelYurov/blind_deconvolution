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

function [u_end,u_four,its] =  non_blind_MHDM(f,k_four,lambda_0,r,stopping)
%performs the MHDM for guessed kernel with Sobolev penalty

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: noise corrupted blurred image f, Fourier transform k_four of
%guessed kernel, initial regularization parameter lambda_0, power r for
%sobolev norm in Fourier space, threshhold stopping to terminate via
%discrepancy principle
%Output: Fourier transform u_four of regularized approximate solution

l2 = @(u) sqrt(sum(abs(u).^2,'all'));
[m,n] = size(f);

%Fourier weights (according to page 110 in L. Justen. Blind Deconvolution:
%"Theory, Regularization and Applications".), indexing adapted to fft2 in
%Matlab
delta =  1+repmat(2*m^2*(1-cos(2.*pi.*(0:m-1)./m))',1,n) + repmat(2*n^2*(1-cos(2.*pi.*(0:n-1)./n)),m,1);


f_four = fft2(f);
lambda = lambda_0;
u_four = (conj(k_four).*f_four)./(abs(k_four).^2+lambda*delta.^r);
its = 1;
residual = l2(f-real(ifft2(k_four.*u_four)));
while residual > stopping
    its = its+1;
    lambda = lambda/4;
    u_four = u_four + (conj(k_four).*(f_four-k_four.*u_four))./(abs(k_four).^2+lambda*delta.^r);
    residual = l2(f-real(ifft2(k_four.*u_four)));
end
u_end = real(ifft2(u_four));
end