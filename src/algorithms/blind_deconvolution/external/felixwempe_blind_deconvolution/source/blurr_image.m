function [ y, B, w, h ] = blurr_image( x, mat )
%Takes in image x and blurres it using a blurring kernel w.
%Furthermore takes a functionhandle to resize the image.
%   Returns the blurred image y and the subspace B in which the blurring
%   kernel w lies. Also returns blurring kernel w and groundtruth h if 
%   needed.
    
    L = length(x); %For notation compare to the paper.
    % Resize the image
    X = mat(x);
    [s1,s2] = size(X);
    
    % Generate blurring kernel w
    blur_kern = fspecial('motion', 5,15);
    [sb1, sb2] = size(blur_kern);
    w = zeros(size(X)); % w needs same size as x
    w(s1/2-(sb1+1)/2+1:s1/2+(sb1+1)/2-1, s2/2-(sb2+1)/2+1:s2/2+(sb2+1)/2-1) = blur_kern; %Change if any of the values is odd.
    
    % Convolute the image. 
    Y = ifft2(fft2(X).*fft2(w));
    Y = fftshift(Y);
    y = Y(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maybe problem with the convolution
    
    % Compute the subspace of w
    w = w(:);
    K = sum(zeros(length(w),1) ~= w); %Count how many indices of w are unequal zero.
    B = sparse(L,K);
    h = zeros(K,1);
    j=1;
    for i=1:L
        if(w(i))
            B(i,j)=1;
            h(j) = w(i);
            j=j+1;
        end
    end
    
    
end

