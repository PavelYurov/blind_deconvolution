%% Create a C_full matrix.

x1 = im2double(rgb2gray(imread('images/new/circ.png')));
x2 = im2double(rgb2gray(imread('images/new/line.png')));
x3 = im2double(rgb2gray(imread('images/new/poly.png')));
x4 = im2double(rgb2gray(imread('images/new/rect.png')));
x5 = im2double(rgb2gray(imread('images/new/rect_3d.png')));
x6 = im2double(rgb2gray(imread('images/new/star.png')));

[s1,s2] = size(x1);

x1 = x1(:);
x2 = x2(:);
x3 = x3(:);
x4 = x4(:);
x5 = x5(:);
x6 = x6(:);

C_full = [x1 x2 x3 x4 x5 x6];
clear x1 x2 x3 x4 x5 x6
