% This example shows how to use the method proposed in:
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
%
%
% 
%% Load image and reference vectors
clc,clear all
I = imread('histWB.jpg');
load 'MLandini' RM;
[m,n,nc] = size(I);
subplot(241),imshow(I)
title('Original H&E Image')
%% Deconvolution
ns=2; %number of stains

epsilon = 2.0e-5;
niter = 1000;

[CT, M, alpha, beta, gamma] = BCDHE(im2double(I), RM(:,1:ns), epsilon, niter );

disp('completed')

%% Band visualization (OD space)

ns = size(M,2)
concentrations = reshape(CT',m,n,ns);

%figure()
subplot(242),imshow(concentrations(:,:,1))
title('OD H Band')
subplot(246),imshow(concentrations(:,:,2))
title('OD E Band')


%% Band reconstruction (RGB space)
Hrec_OD = reshape((M(:,1)*CT(1,:))',m,n,nc);
Hrec_RGB = OD2intensities(Hrec_OD);

Erec_OD = reshape((M(:,2)*CT(2,:))',m,n,nc);
Erec_RGB = OD2intensities(Erec_OD);

%figure()
subplot(243),imshow(Hrec_RGB)
title('RGB H Band')
subplot(247),imshow(Erec_RGB)
title('RGB E Band')

%% Image Normalization

Iref = imread('Reference.jpg');
%Deconvolution of the reference image
[Cref, Mref, alpha, beta, gamma] = BCDHE(im2double(Iref), RM(:,1:ns), epsilon, niter );
disp('completed')

%% Range adjustment
CT_Rmax = prctile(CT',99)
Cref_Rmax= prctile(Cref',99)
norm_fac=Cref_Rmax./CT_Rmax
CT_norm=CT.*norm_fac';

%Reconstruction
Yrec_norm=Mref(:,1:ns)*CT_norm;
Y2d_norm=reshape(Yrec_norm',m,n,nc);
Irec_norm=OD2intensities(Y2d_norm);

subplot(244),imshow(Irec_norm)
title('Normalized image')
subplot(248),imshow(Iref)
title('Reference')
