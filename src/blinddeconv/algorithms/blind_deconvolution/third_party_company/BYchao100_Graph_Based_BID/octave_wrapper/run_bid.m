function [I_FHLP, k_estimate] = run_bid(Y_b, I_blur, k_estimate_size, show_intermediate)
    root = fileparts(mfilename('fullpath'));
    pkg load image
    addpath(genpath(fullfile(root, '..', 'source')));
    addpath(genpath(fullfile(root, '..', 'source', 'Graph_Based_BID', 'Graph_Based_BID_v1.1')));
    show_intermediate = logical(show_intermediate);

    [k_estimate, ~] = bid_rgtv_c2f_cg(Y_b, k_estimate_size, show_intermediate);

    if ndims(I_blur) == 3 && size(I_blur, 3) == 3
        I_FHLP = I_blur;
        for c = 1:3
            I_FHLP(:,:,c) = Deconvolution_FHLP(I_blur(:,:,c), k_estimate);
        end
    else
        I_FHLP = Deconvolution_FHLP(I_blur, k_estimate);
    end

end
