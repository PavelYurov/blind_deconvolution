function [ x, C ] = generate_image( C_full, m_full )
%Generates an image in the subspace of C_full.
%   Takes in a sparse vector m_full to generate the image X. Also returns
%   the subspace of C_full in which X lies.
%   C_full*m_full must be a vector!!! 
    
    % Compute X
    x = C_full*m_full;
    L = length(x);
    
    % Compute the subspace C of m_full
    N = sum(m_full~=zeros(length(m_full),1)); %Counts how many indices of m_full are non-zero.
    C = zeros(L,N);
    j=1;
    for i=1:length(m_full)
        if(m_full(i))%only consider entries non-zero
            C(:,j) = C_full(:,i);
            j = j+1;
        end
    end
  %  C = sparse(C);
end

