function [ h_opt, m_opt ] = recov_m_h( X_opt, m_1, h_gt )
%recovers m and h from X.
%   

% h_opt = h_gt;
% A = zeros(length(X_opt(:))-length(h_opt), size(X_opt,2)-1);
% for i=1:size(X_opt,2)-1
%     A((i-1)*length(h_opt)+1:i*length(h_opt),i) = h_opt;
% end
% X_opt = X_opt(:);
% X = X_opt(length(h_opt)+1:end);
% m_opt = A\X;
% m_opt = vertcat(m_1, m_opt);

h_opt = h_gt;
A = zeros(length(X_opt(:)), size(X_opt,2));
for i=1:size(X_opt,2)
    A((i-1)*length(h_opt)+1:i*length(h_opt),i) = h_opt;
end
X = X_opt(:);
m_opt = A\X;


end

