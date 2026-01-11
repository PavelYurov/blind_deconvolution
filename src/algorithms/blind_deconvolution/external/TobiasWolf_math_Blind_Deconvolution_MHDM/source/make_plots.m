% Copyright (C) 2025 Tobias Wolf <tobias.wolf@aau.at>

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.

% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%creates plots of all images and PSNR, SSIM and relative kernel errors for
%comparison of blind deconvolution with MHDM and grid search for single
%step approach
%Attention! Creates many files!

close all
clear
clc

load("data_comp.mat");

dir = strcat(pwd,'/images_comparison');
mkdir(dir);

%%

%plot true images and kernels
for l = 1:num_Ims
    u_true = cell2mat(u_true_arr(1+(l-1)*num_noises*num_kernels));
    [m,n] = size(u_true);
    figure('Visible','off');
    image(255*u_true); colormap(gray); axis off; pbaspect([m,n,1]); title(strcat('$U^\dagger_{',num2str(l),'}$'),'Interpreter','latex');
    exportgraphics(gcf, strcat(dir,'/u',num2str(l),'.pdf'));
    close
end

for l = 1:num_kernels
    kernel = cell2mat(ker_arr(1+(l-1)*num_noises));
    figure('Visible','off');
    surf(kernel); shading flat; title(strcat('$K^\dagger_{',num2str(l),'}$'),'Interpreter','latex');
    exportgraphics(gcf, strcat(dir,'/k',num2str(l),'.pdf'));
    close
end


%%

%plot all final images
dir_images = strcat(dir,'/images');
mkdir(dir_images);
for l = 1:num_tests
    f = cell2mat(f_arr(l));
    u_MHDM = cell2mat(u_end_MHDM_arr(l));
    u_single = cell2mat(u_end_single_arr(l));
    [m,n] = size(f);
    figure('Visible','off');
    subplot(1,3,1); image(255*f); colormap(gray); pbaspect([m,n,1]); axis off; title(strcat('$f_{',num2str(l),'} (\delta = ', num2str(cell2mat(delta_arr(l))),')$'),'Interpreter','latex');
    subplot(1,3,2); image(255*u_MHDM); colormap(gray); pbaspect([m,n,1]); axis off; title(strcat('MHDM (PSNR = ', num2str(psnr_u_MHDM(l)),')'));
    subplot(1,3,3); image(255*u_single); colormap(gray); pbaspect([m,n,1]); axis off; title(strcat('single (PSNR = ', num2str(psnr_u_single(l)),')'));
    exportgraphics(gcf, strcat(dir_images,'/im',num2str(l),'.pdf'));
    close
end

%%

%plot all final kernels
dir_kernels = strcat(dir,'/kernels');
mkdir(dir_kernels);
for l = 1:num_tests
    k_true = cell2mat(ker_arr(l));
    k_MHDM = cell2mat(k_end_MHDM_arr(l));
    k_single = cell2mat(k_end_single_arr(l));
    range = 1.05*[min([min(min(k_true)),min(min(k_MHDM)),min(min(k_single))]),max([max(max(k_true)),max(max(k_MHDM)),max(max(k_single))])];
    [m,n] = size(k_true);
    figure('visible','off');
    subplot(1,3,1); surf(k_true); shading flat; view(70,5); pbaspect([m,n,m+n]);  title(strcat('$k_{',num2str(l),'} (\delta = ', num2str(cell2mat(delta_arr(l))),')$'),'Interpreter','latex');zlim(range);
    subplot(1,3,2); surf(k_MHDM); shading flat; view(70,5); pbaspect([m,n,m+n]); title({'MHDM',strcat( '(relative error = ', num2str(err_ker_MHDM(l)),'),')});zlim(range);
    subplot(1,3,3); surf(k_single); shading flat; view(70,5);pbaspect([m,n,m+n]);  title({'single',strcat( '(relative error = ', num2str(err_ker_single(l)),'),')});zlim(range);
    exportgraphics(gcf, strcat(dir_kernels,'/ker',num2str(l),'.pdf'));
    close
end


%%
%PSNR, SSIM and kernel error plots
dir_plots_diff = strcat(dir,'/plots_differences');
mkdir(dir_plots_diff);

dir_plots_abs = strcat(dir,'/plots_abs');
mkdir(dir_plots_abs);

%all

psnr_MHDM_better = find(psnr_u_MHDM > psnr_u_single);
psnr_single_better = find(psnr_u_MHDM < psnr_u_single);
figure('visible','off');
plot(psnr_MHDM_better,psnr_u_MHDM(psnr_MHDM_better) - psnr_u_single(psnr_MHDM_better), 'x', 'Color','red')
hold on
plot(psnr_single_better,psnr_u_MHDM(psnr_single_better) - psnr_u_single(psnr_single_better), 'x', 'Color','black')
plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
hold off
xlim([1,num_tests]);
ylim([-0.6,0.6]);
ylabel('$PSNR_{MHDM} - PSNR_{variational}$','Interpreter','latex')
xlabel('Image')
legend('MHDM better','variational better','Location','northwest')
title('Difference PSNR')
exportgraphics(gcf, strcat(dir_plots_diff,'/psnr_all.pdf'));
close


figure('Visible','off');
plot(psnr_u_MHDM, 'x','Color','red')
hold on
plot(psnr_u_single, 'o','Color','black')
hold off
ylabel('PSNR');
xlabel('Image')
title('PSNR')
legend('MHDM', 'variational','Location','southwest')
exportgraphics(gcf, strcat(dir_plots_abs,'/psnr_all.pdf'));
close




ssim_MHDM_better = find(ssim_u_MHDM > ssim_u_single);
ssim_single_better = find(ssim_u_MHDM < ssim_u_single);
figure('visible','off');
plot(ssim_MHDM_better,ssim_u_MHDM(ssim_MHDM_better) - ssim_u_single(ssim_MHDM_better), 'x', 'Color','red')
hold on
plot(ssim_single_better,ssim_u_MHDM(ssim_single_better) - ssim_u_single(ssim_single_better), 'x', 'Color','black')
plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
hold off
xlim([1,num_tests]);
ylim([-0.3,0.3]);
ylabel('$SSIM_{MHDM} - SSIM_{variational}$','Interpreter','latex')
xlabel('Image')
legend('MHDM better','variational better','Location','northwest')
title('Difference SSIM')
exportgraphics(gcf, strcat(dir_plots_diff,'/ssim_all.pdf'));
close


figure('Visible','off');
plot(ssim_u_MHDM, 'x','Color','red')
hold on
plot(ssim_u_single, 'o','Color','black')
hold off
ylabel('SSIM');
xlabel('Image')
title('SSIM')
legend('MHDM', 'variational','Location','southwest')
exportgraphics(gcf, strcat(dir_plots_abs,'/SSIM_all.pdf'));
close


err_ker_MHDM_better = find(err_ker_MHDM < err_ker_single);
err_ker_single_better = find(err_ker_MHDM > err_ker_single);
figure('visible','off');
plot(ssim_MHDM_better,err_ker_MHDM(ssim_MHDM_better) - err_ker_single(ssim_MHDM_better), 'x', 'Color','red')
hold on
plot(err_ker_single_better,err_ker_MHDM(err_ker_single_better) - err_ker_single(err_ker_single_better), 'x', 'Color','black')
plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
hold off
xlim([1,num_tests]);
ylim(1.1*[min(err_ker_MHDM-err_ker_single),-min(err_ker_MHDM-err_ker_single)]);
ylabel('$error_{MHDM} - error_{variational}$','Interpreter','latex')
xlabel('Image')
legend('MHDM better','variational better','Location','northwest')
title('Difference relative error kernels')
exportgraphics(gcf, strcat(dir_plots_diff,'/err_ker_all.pdf'));
close

figure('Visible','off');
plot(err_ker_MHDM, 'x','Color','red')
hold on
plot(err_ker_single, 'o','Color','black')
hold off
ylabel('error kernel');
xlabel('Image')
title('relative error kernels')
legend('MHDM', 'variational','Location','southwest')
exportgraphics(gcf, strcat(dir_plots_abs,'/err_ker_all.pdf'));
close


%by noise level
indices_noise_levels = zeros(num_tests/num_noises,num_noises);
for l = 1:num_noises
    indices_noise_levels(:,l) = l:num_noises:num_tests;
end

for l = 1:num_noises
    noise_level = noise_levels(l);
    indices = indices_noise_levels (:,l);

    psnr_MHDM_better = find(psnr_u_MHDM(indices) > psnr_u_single(indices));
    psnr_single_better = find(psnr_u_MHDM(indices) < psnr_u_single(indices));
    figure('visible','off');
    plot(indices(psnr_MHDM_better),psnr_u_MHDM(indices(psnr_MHDM_better)) - psnr_u_single(indices(psnr_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(psnr_single_better),psnr_u_MHDM(indices(psnr_single_better)) - psnr_u_single(indices(psnr_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim([-0.6,0.6]);
    ylabel('$PSNR_{MHDM} - PSNR_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference PSNR with noise variance $\sigma = ',num2str(noise_level),'$'),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/psnr_noise',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,psnr_u_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,psnr_u_single(indices), 'o','Color','black')
    hold off
    ylabel('PSNR');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('PSNR with noise variance $\sigma = ',num2str(noise_level),'$'),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/psnr_noise',num2str(l),'.pdf'));
    close


    ssim_MHDM_better = find(ssim_u_MHDM(indices) > ssim_u_single(indices));
    ssim_single_better = find(ssim_u_MHDM(indices) < ssim_u_single(indices));
    figure('visible','off');
    plot(indices(ssim_MHDM_better),ssim_u_MHDM(indices(ssim_MHDM_better)) - ssim_u_single(indices(ssim_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(ssim_single_better),ssim_u_MHDM(indices(ssim_single_better)) - ssim_u_single(indices(ssim_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim([-0.3,0.3]);
    ylabel('$SSIM_{MHDM} - SSIM_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference SSIM with noise variance $\sigma = ',num2str(noise_level),'$'),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/ssim_noise',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,ssim_u_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,ssim_u_single(indices), 'o','Color','black')
    hold off
    ylabel('SSIM');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('SSIM with noise variance $\sigma = ',num2str(noise_level),'$'),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/ssim_noise',num2str(l),'.pdf'));
    close

    err_ker_MHDM_better = find(err_ker_MHDM(indices) < err_ker_single(indices));
    err_ker_single_better = find(err_ker_MHDM(indices) > err_ker_single(indices));
    figure('visible','off');
    plot(indices(err_ker_MHDM_better),err_ker_MHDM(indices(err_ker_MHDM_better)) - err_ker_single(indices(err_ker_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(err_ker_single_better),err_ker_MHDM(indices(err_ker_single_better)) - err_ker_single(indices(err_ker_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim(1.1*[min(err_ker_MHDM-err_ker_single),-min(err_ker_MHDM-err_ker_single)]);
    ylabel('$error_{MHDM} - error_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference relative error kernels with noise variance $\sigma = ',num2str(noise_level),'$'),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/err_ker_noise',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,err_ker_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,err_ker_single(indices), 'o','Color','black')
    hold off
    ylabel('error kernel');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('relative error kernels with noise variance $\sigma = ',num2str(noise_level),'$'),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/err_ker_noise',num2str(l),'.pdf'));
    close
end


%by blur
indices_blurs = zeros(num_tests/num_kernels,num_kernels);

for l = 1:num_kernels
    indices_blurs(:,l) = find(ismember(mod(1:num_tests,num_kernels*num_noises),mod((1+(l-1)*num_noises:l*num_noises),num_kernels*num_noises)));
end

for l = 1:num_kernels
    indices = indices_blurs(:,l);

    psnr_MHDM_better = find(psnr_u_MHDM(indices) > psnr_u_single(indices));
    psnr_single_better = find(psnr_u_MHDM(indices) < psnr_u_single(indices));
    figure('visible','off');
    plot(indices(psnr_MHDM_better),psnr_u_MHDM(indices(psnr_MHDM_better)) - psnr_u_single(indices(psnr_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(psnr_single_better),psnr_u_MHDM(indices(psnr_single_better)) - psnr_u_single(indices(psnr_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim([-0.6,0.6]);
    ylabel('$PSNR_{MHDM} - PSNR_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference PSNR for kernel ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/psnr_blur',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,psnr_u_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,psnr_u_single(indices), 'o','Color','black')
    hold off
    ylabel('PSNR');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('PSNR for kernel ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/psnr_blur',num2str(l),'.pdf'));
    close

    ssim_MHDM_better = find(ssim_u_MHDM(indices) > ssim_u_single(indices));
    ssim_single_better = find(ssim_u_MHDM(indices) < ssim_u_single(indices));
    figure('visible','off');
    plot(indices(ssim_MHDM_better),ssim_u_MHDM(indices(ssim_MHDM_better)) - ssim_u_single(indices(ssim_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(ssim_single_better),ssim_u_MHDM(indices(ssim_single_better)) - ssim_u_single(indices(ssim_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim([-0.3,0.3]);
    ylabel('$SSIM_{MHDM} - SSIM_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference SSIM for kernel ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/ssim_blur',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,ssim_u_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,ssim_u_single(indices), 'o','Color','black')
    hold off
    ylabel('SSIM');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('SSIM for kernel ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/ssim_blur',num2str(l),'.pdf'));
    close

    err_ker_MHDM_better = find(err_ker_MHDM(indices) < err_ker_single(indices));
    err_ker_single_better = find(err_ker_MHDM(indices) > err_ker_single(indices));
    figure('visible','off');
    plot(indices(err_ker_MHDM_better),err_ker_MHDM(indices(err_ker_MHDM_better)) - err_ker_single(indices(err_ker_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(err_ker_single_better),err_ker_MHDM(indices(err_ker_single_better)) - err_ker_single(indices(err_ker_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim(1.1*[min(err_ker_MHDM-err_ker_single),-min(err_ker_MHDM-err_ker_single)]);
    ylabel('$error_{MHDM} - error_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference relative error kernels for kernel', num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/err_ker_blur',num2str(l),'.pdf'));
    close


    figure('Visible','off');
    plot(indices,err_ker_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,err_ker_single(indices), 'o','Color','black')
    hold off
    ylabel('error kernel');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('Relative error kernels for kernel ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/err_ker_blur',num2str(l),'.pdf'));
    close

end

%%
%by image
indices_ims = zeros(num_tests/num_Ims,num_Ims);
for l = 1:num_Ims
    indices_ims(:,l) = (1+(l-1)*num_noises*num_kernels):(l*num_noises*num_kernels);
end

for l = 1:num_Ims
    indices = indices_ims(:,l);


    psnr_MHDM_better = find(psnr_u_MHDM(indices) > psnr_u_single(indices));
    psnr_single_better = find(psnr_u_MHDM(indices) < psnr_u_single(indices));
    figure('visible','off');
    plot(indices(psnr_MHDM_better),psnr_u_MHDM(indices(psnr_MHDM_better)) - psnr_u_single(indices(psnr_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(psnr_single_better),psnr_u_MHDM(indices(psnr_single_better)) - psnr_u_single(indices(psnr_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim([-0.6,0.6]);
    ylabel('$PSNR_{MHDM} - PSNR_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference PSNR for image ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/psnr_im',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,psnr_u_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,psnr_u_single(indices), 'o','Color','black')
    hold off
    ylabel('PSNR');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('PSNR for image ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/psnr_im',num2str(l),'.pdf'));
    close

    ssim_MHDM_better = find(ssim_u_MHDM(indices) > ssim_u_single(indices));
    ssim_single_better = find(ssim_u_MHDM(indices) < ssim_u_single(indices));
    figure('visible','off');
    plot(indices(ssim_MHDM_better),ssim_u_MHDM(indices(ssim_MHDM_better)) - ssim_u_single(indices(ssim_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(ssim_single_better),ssim_u_MHDM(indices(ssim_single_better)) - ssim_u_single(indices(ssim_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim([-0.3,0.3]);
    ylabel('$SSIM_{MHDM} - SSIM_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference SSIM for image ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/ssim_im',num2str(l),'.pdf'));
    close

    figure('Visible','off');
    plot(indices,ssim_u_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,ssim_u_single(indices), 'o','Color','black')
    hold off
    ylabel('SSIM');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('SSIM for image ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/ssim_im',num2str(l),'.pdf'));
    close

    err_ker_MHDM_better = find(err_ker_MHDM(indices) < err_ker_single(indices));
    err_ker_single_better = find(err_ker_MHDM(indices) > err_ker_single(indices));
    figure('visible','off');
    plot(indices(err_ker_MHDM_better),err_ker_MHDM(indices(err_ker_MHDM_better)) - err_ker_single(indices(err_ker_MHDM_better)), 'x', 'Color','red')
    hold on
    plot(indices(err_ker_single_better),err_ker_MHDM(indices(err_ker_single_better)) - err_ker_single(indices(err_ker_single_better)), 'x', 'Color','black')
    plot(1:num_tests,zeros(1,num_tests), '--', 'Color','black')
    hold off
    xlim([1,num_tests]);
    ylim(1.1*[min(err_ker_MHDM-err_ker_single),-min(err_ker_MHDM-err_ker_single)]);
    ylabel('$error_{MHDM} - error_{variational}$','Interpreter','latex')
    xlabel('Image')
    legend('MHDM better','variational better','Location','northwest')
    title(strcat('Difference relative error kernels for image', num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_diff,'/err_ker_im',num2str(l),'.pdf'));
    close


    figure('Visible','off');
    plot(indices,err_ker_MHDM(indices), 'x','Color','red')
    hold on
    plot(indices,err_ker_single(indices), 'o','Color','black')
    hold off
    ylabel('error kernel');
    xlabel('Image')
    legend('MHDM', 'variational','Location','southwest')
    title(strcat('Relative error kernels for image ' ,num2str(l)),'Interpreter','latex')
    exportgraphics(gcf, strcat(dir_plots_abs,'/err_ker_im',num2str(l),'.pdf'));
    close
end
