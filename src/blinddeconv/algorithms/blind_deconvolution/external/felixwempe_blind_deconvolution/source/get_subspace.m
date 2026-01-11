function [ C, S, N, m ] = get_subspace( x, L, mat )
%Takes an image and computes the subspace using the wavedecomposition with
%haar wavelets.
%   Input: image x
%   Output: Subspace matrix C (each column is a unit vector)
%           Bookkeeping matrix S from wavedecomposition
%           Dimension of subspace N
%           groundtruth vector m (compare to paper for notation)
X = mat(x);

[C_haar,S] =  wavedec2(X, 1, 'haar');

% Ind = abs(C_haar) > 1.0e-1;% Das sollte besser gewählt werden.....
% N = sum(Ind);
% C = sparse(L,N);
% m = zeros(N,1);
% j=1;
% for i=1:length(C_haar)
%     if(Ind(i))
%         C(i,j) = 1;
%         m(j) = C_haar(i);
%         j=j+1;
%     end
% end

N = sum(boolean(C_haar));
C = sparse(L,N);
m = zeros(N,1);
j=1;
for i=1:length(C_haar)
    if(C_haar(i))
        C(i,j) = 1;
        m(j) = C_haar(i);
        j=j+1;
    end
end

end

