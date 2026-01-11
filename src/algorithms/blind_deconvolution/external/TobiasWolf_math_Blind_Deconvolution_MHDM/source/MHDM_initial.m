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

function [u_four,k_four] = MHDM_initial(f_four,lambda,mu,r,s)
%computes the initial iterates of the blind deconvolution MHDM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: blurred and noisy image f, regularization parameters lambda and mu, powers r and s for sobolev norms in Fourier space 
%Output: Fourier transforms u_four and k_four of the reconstructed kernel and image 

[m,n]=size(f_four);

%Fourier weights (according to page 110 in L. Justen. Blind Deconvolution:
%"Theory, Regularization and Applications".), indexing adapted to fft2 in
%Matlab
delta =  1+repmat(2*m^2*(1-cos(2.*pi.*(0:m-1)./m))',1,n) + repmat(2*n^2*(1-cos(2.*pi.*(0:n-1)./n)),m,1);

%pointwise computation of minimizers
u_four = sqrt(max(sqrt((mu/lambda).*delta.^(s-r)).*abs(f_four)-mu.*delta.^s,0)).*sign(f_four);
k_four = sqrt(max(sqrt((lambda/mu).*delta.^(r-s)).*abs(f_four) -lambda.*delta.^r,0));
k_four(1) = 1;
u_four(1) = f_four(1);