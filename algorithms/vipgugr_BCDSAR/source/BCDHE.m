function [CT, M, alpha, beta, gamma ] = BCDHE(I, RM, term, nitermax)
%BCDHE Bayesian Color Deconvolution of Hematoxilyn, Eosin slides
%@ARTICLE{,
% author = {N. Hidalgo-Gavira and J. Mateos and M. Vega and R. Molina and A.K. Katsaggelos},
% title = {Variational Bayesian Blind Color Deconvolution of Histopathological Images},
% journal = {IEEE Transactions on Image Processing},
% year = {2020},
% volume = {29},
% number = {1},
% pages = {2026-2036},
% month = {},
% url = { https://decsai.ugr.es/vip/files/journals/2019_TIP_BCD.pdf },
% annote = {Most whole-slide histological images are stained with two or more chemical dyes. Slide stain separation or color deconvolution is a crucial step within the digital pathology workflow. In this paper, the blind color deconvolution problem is formulated within the Bayesian framework. Starting from a multi-stained histological image, our model takes into account both spatial relations among the concentration image pixels and similarity between a given reference color-vector matrix and the estimated one. Using Variational Bayes inference, three efficient new blind color deconvolution methods are proposed which provide automated procedures to estimate all the model parameters in the problem. A comparison with classical and current state-of-the-art color deconvolution algorithms using real images has been carried out demonstrating the superiority of the proposed approach. }
% }
% Inputs: 
%   I: RGB observed image
%   RM: Reference color vector matrix of size 3xns
%   term: Noise parameter
%   nitermax: number of max iter
% Outputs:
%   CT: Stain concentration matrix of size nsxn_pixels
%   M: Estimated color vector matrix
%   alpha, beta, gamma: model parameters
    [m,n,nc] = size(I);
    tamm = m*n;
    ns = size(RM,2); %number of stains

    if (nc ~= 3 )
        error('Input image does not have 3 channels');
    end

    y2d = intensities2OD( I ); % dimension (m,n,nc)
    YT=reshape(y2d,m*n,nc)';

    clear I y2d 

    % Some stuff
    Fn=[ 0   -.25 0;
        -.25  1  -.25;
        0   -.25 0];
    FTFn=conv2(Fn,Fn);
    FTF=psf2otf(FTFn, [m, n]);

    clear Fn FTFn

    % Initial values
    CT = RM \ YT;
    CT(CT < eps) = eps;
    M = RM;

    for s=1:ns
        SigmaC = zeros(m*n,ns);
        SigmaM(s) = 0 ;
    end

    CT0 = CT;
    iter = 1; convH = term +1.0; convE = term +1.0;
    %Iterations
    while (((convH > term) || (convE > term)) && (iter <= nitermax))

        % Parameters update
        % beta
        beta = beta_update(YT,CT,SigmaC,M,SigmaM);

        % alpha
        alpha = alpha_update(CT,FTF,SigmaC);

        % gamma
        gamma = gamma_update(M,SigmaM,RM);

        fprintf('iter: %3d\t beta: %f\n',iter,beta)
        fprintf('H\t alpha: %f\t gamma: %f\n',alpha(1),gamma(1))
        fprintf('E\t alpha: %f\t gamma: %f\n',alpha(2),gamma(2))

        % Color vector update
        [M, SigmaM] = color_vector_update(YT,CT,SigmaC,M,RM,beta,gamma);

        % Concentration update
        [CT,SigmaC] = conc_update(YT,CT,M,SigmaM,FTF,beta,alpha);

        convH = sum((CT(1,:)- CT0(1,:)).*(CT(1,:)- CT0(1,:))) / sum(CT0(1,:).*CT0(1,:));
        convE = sum((CT(2,:)- CT0(2,:)).*(CT(2,:)- CT0(2,:))) / sum(CT0(2,:).*CT0(2,:));
        CT0 = CT;

        fprintf('H\t conv: %e\n',convH)
        fprintf('E\t conv: %e\n',convE)
        M


        iter = iter +1;

    end

    CT(CT < eps) = eps;

end


