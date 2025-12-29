function [ circ_C ] = circular( C )
%Returns the matrix realising the circular convolution as a matrix product
%   Input C is a vector.
if size(C,1)~=1 && size(C,2)~=1
    error('Input not a vector.')
elseif size(C,1)~=1
    C = C';
end

l = length(C);
circ_C = zeros(l);
C_rev = wrev(C);
for i=1:l
    circ_C(i,:) = [C_rev(l-i+1:l) C_rev(1:l-i)];
end

end

