function XX = adjcA1d(y,B,C,N)
vec = @(x) x(:);
XX = cell(N,1);
    for n = 1:N 
        XX{n} = (B)'*(diag(vec(y{n}))*conj((C{n})));
    end
end