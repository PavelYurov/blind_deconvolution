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

function[u_four,k_four] = MHDM_step(u_n,k_n,f_four,lambda,mu,r,s,tol,indices)
%computes the increments of the blind deconvolution MHDM after the initial
%step

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: previous iterates u_n and k_n, Fourier transform of blurred and noisy image f_four,
%regularization parameters lambda and mu, powers r and s for sobolev norms
%in Fourier space, tolerance tol for small negative values due to numerical issues, indices for which anti-symmetry condition is enforced
%Output: Fourier transforms u_four and k_four of the reconstructed kernel
%and image increments

[m,n]=size(f_four);






num_inds = length(indices(:,2));

%Fourier weights (according to page 110 in L. Justen. Blind Deconvolution:
%"Theory, Regularization and Applications".), indexing adapted to fft2 in
%Matlab
delta =  1+repmat(2*m^2*(1-cos(2.*pi.*(0:m-1)./m))',1,n) + repmat(2*n^2*(1-cos(2.*pi.*(0:n-1)./n)),m,1);


u_four = zeros(m,n);
k_four = u_four;

k_four_h = zeros(1,num_inds);
u_four_h = k_four_h;

%pointwise computation of minimizers
parfor k = 1:num_inds
    l = indices(k,2);
    a_n = lambda*delta(l)^r;
    b_n = mu*delta(l)^s;
    obj = @(q) a_n./((abs(q+k_n(l)).^2)+a_n).*abs(u_n(l).*(q+k_n(l))-f_four(l)).^2 + b_n*abs(q).^2;
    coeffs = real([b_n,-k_n(l)*b_n,2*a_n*b_n,a_n*f_four(l)*conj(u_n(l))-2*a_n*b_n*k_n(l),a_n^2*abs(u_n(l))^2-a_n*abs(f_four(l))^2+a_n^2*b_n,-a_n^2*(f_four(l)*conj(u_n(l))+b_n*k_n(l))]);
    zer = roots(coeffs) - k_n(l);
    length_zer = length(zer);
    k_four_h(k) = zer(1);
    for ll = 2:length_zer
        if obj(zer(ll)) <= obj(k_four_h(k)) && real(zer(ll)) >=-tol
            k_four_h(k) = zer(ll);
        end
    end
    u_four_h(k) = (a_n*u_n(l)+f_four(l)*conj(k_four_h(k) +k_n(l)))/(abs(k_four_h(k) + k_n(l))^2+a_n)-u_n(l);
end

k_four(indices(:,2)) = k_four_h;
u_four(indices(:,2)) = u_four_h;
k_four(indices(:,1)) = conj(k_four(indices(:,2)));
u_four(indices(:,1)) = conj(u_four(indices(:,2)));
k_four(1) =1-k_n(1);
u_four(1) = f_four(1) -u_n(1);

end


