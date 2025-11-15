function X = concat1d(XX,N,K)
    X = zeros(size(XX{1},1),N*K);
    for n = 1:N
       X(:,(N-1)*K+1:N*K) = XX{n};
    end
end