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

%creates blurred and noisy test image, performs blind MHDM, non-blind
%MHDM and single step variational regularization and comparisons of the
%methods



close all
clear
clc
rng('default')
l2= @(u) sqrt(sum(abs(u).^2,'all'));

%create data and save images
u_true= im2double((imread('test_images/im4.jpg')));
if (size(u_true,3) > 1)
    u_true = rgb2gray(u_true);
end

[m,n] = size(u_true);

%create blurred data
experiment = 1; %choose 1 for single heavy blur kernel and 2 for convex combination of Gaussian kernels
switch experiment
    case 1
        dir = strcat(pwd,'/single');
        kernel =  fspecial('gaussian',[m,n],8);

    case 2
        dir = strcat(pwd,'/multiple');
        kernel =  fspecial('gaussian',[m,n],4);
        kernel = (5*kernel+ fspecial('gaussian',[m,n],5)+fspecial('gaussian',[m,n],1)+fspecial('gaussian',[m,n],3))/8;

end


mkdir(dir);
f_blur = real(ifft2(psf2otf(kernel, [m n]).*fft2(u_true)));
f = imnoise(f_blur,'gaussian',0,4e-4);
noise=f-f_blur;
delta = l2(noise);

f_four = fft2(f);

[rr,c] = ismember(f_four,conj(f_four));
c =reshape(c,m*n,1);
M = [c, c(c)];
sortedM = sort(M, 2);
[~, uniqueIdx] = unique(sortedM, 'rows', 'stable');
indices = M(uniqueIdx, :);

figure('visible','off'); surf(kernel); shading flat; title(strcat('$K_{',num2str(experiment),'}^\dagger$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/kernel_true.pdf'));close;

%set figure to visible to prevent it being cut when saving
figure; surf(kernel(n/2-25:n/2+25,n/2-25:n/2+25)); shading flat;title(strcat('$K_{',num2str(experiment),'}^\dagger$'),'Interpreter','latex',FontSize=20);exportgraphics(gcf,strcat(dir,'/kernel_zoom.pdf'));close;

figure('visible','off'); image(255*f_blur); colormap(gray); axis off; pbaspect([m,n,1]);title(strcat('$f_{',num2str(experiment),'}^\delta$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/blurred.pdf'));close;
figure('visible','off'); image(255*f); colormap(gray); axis off; pbaspect([m,n,1]);title(strcat('$f_{',num2str(experiment),'}$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/data.pdf'));close;
figure('visible','off'); image(255*u_true); colormap(gray); axis off; pbaspect([m,n,1]);title('$U^\dagger$','Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/ground_truth.pdf'));close;

tol = 1e-10;



lambda = 14e-5;
mu = 63e4;





r = 1;
s =1e-1;



lambda_0 = lambda;
mu_0 = mu;
tau = 1.001;
stopping = tau*delta;
maxits = 30;

its = 0;



%% compute blind MHDM
[u_end,k_end,u_four_MHDM,k_four_MHDM,its_blind] = blind_deconvolution_MHDM(f,f_four,lambda_0,mu_0,r,s,tol,stopping,maxits,indices);
err_im_blind = l2(u_true-u_end)/l2(u_true);
err_ker_blind = l2(kernel-k_end)/l2(kernel);
psnr_blind = psnr(u_end,u_true);
ssim_blind = ssim(u_end,u_true);

images_blind = u_four_MHDM;
kernels_blind = k_four_MHDM;
parfor l = 1:its_blind
    residual_blind(l) = l2(f-real(ifft2(k_four_MHDM(:,:,l).*u_four_MHDM(:,:,l))))
    images_blind(:,:,l) = real(ifft2(u_four_MHDM(:,:,l)));
    kernels_blind(:,:,l) = real(otf2psf(k_four_MHDM(:,:,l)));
end




%% test vs MHDM with guessed kernel
sigmas = linspace(1,12,1000);


for l = 1:length(sigmas)
    kernel_guess = fspecial('gaussian',[m,n],sigmas(l));
    k_four = psf2otf(kernel_guess,[m,n]);
    [u_end,u_four,its] =  non_blind_MHDM(f,k_four,lambda_0,r,stopping);
    u_MHDM = u_end;
    iterations(l) = its;
    err_im(l) = l2(u_true-u_MHDM)/l2(u_true);
    err_ker(l) = l2(kernel-kernel_guess)/l2(kernel);
    kernels(:,:,l) = kernel_guess;
    psnr_MHDM(l) = psnr(u_MHDM,u_true);
    ssim_MHDM(l) = ssim(u_MHDM,u_true);
    images_MHDM(:,:,l) = u_MHDM;
end




figure('visible','off')
plot(sigmas,err_ker,'Color','black');
hold on
plot(sigmas,ones(1,length(sigmas))*err_ker_blind,'--','Color','black')
hold off
legend('non-blind MHDM','blind deconvolution MHDM','Location','northeast')
xlabel('$\sigma$','Interpreter','latex'); ylabel('relative $L^2$-error','Interpreter','latex')

exportgraphics(gcf,strcat(dir,'/err_kernel_non_blind.pdf'),'ContentType','vector');
close


figure('visible','off')
plot(sigmas,psnr_MHDM,'Color','black');
hold on
plot(sigmas,ones(1,length(sigmas))*psnr_blind,'--','Color','black')
hold off
ylim([-2,24]);
legend('non-blind MHDM','blind deconvolution MHDM','Location','southwest')
xlabel('$\sigma$','Interpreter','latex'); ylabel('PSNR (dB)')

exportgraphics(gcf,strcat(dir,'/PSNR_non_blind.pdf'),'ContentType','vector');
close

figure('visible','off')
plot(sigmas,ssim_MHDM,'Color','black');
hold on
plot(sigmas,ones(1,length(sigmas))*ssim_blind,'--','Color','black')
hold off
legend('non-blind MHDM','blind deconvolution MHDM','Location','southwest')
xlabel('$\sigma$','Interpreter','latex'); ylabel('SSIM')

exportgraphics(gcf,strcat(dir,'/SSIM_non_blind.pdf'),'ContentType','vector');
close


ind_best = find(ssim_MHDM == max(ssim_MHDM));
u_best = images_MHDM(:,:,ind_best);

figure('visible','off'); image(255*u_best);pbaspect([m,n,1]); colormap(gray); axis off; title('non-blind MHDM',FontSize=20)
exportgraphics(gcf,strcat(dir,'/u_best_non_blind.pdf')); close;


%% blind MHDM for different ratios

alt_lambdas = [2e-3,14e-5,2e-3,2e3];
alt_mus = [63e4,1e3,1e3,1e-3];

u_alt = zeros(m,n,length(alt_lambdas));
k_alt = zeros(m,n,length(alt_lambdas));

for l = 1:length(alt_lambdas)
    alt_lambda_0 = alt_lambdas(l);
    alt_mu_0 = alt_mus(l);

    [u_end,k_end,u_four_MHDM,k_four_MHDM,its_blind_alt] = blind_deconvolution_MHDM(f,f_four,alt_lambda_0,alt_mu_0,r,s,tol,stopping,maxits,indices);

    u_alt(:,:,l) = u_end;
    k_alt(:,:,l) = k_end;
end



%% plot blind MHDM iterates


figure('visible','off')
plot(0:its_blind-1,residual_blind,'Color','black');
hold on
plot(0:its_blind-1,sqrt(delta)*ones(1,its_blind),'--',Color='black')
hold off
curtick = get(gca, 'xTick');
xticks(unique(round(curtick)));
legend('residual', '$\delta$','Interpreter','latex')
xlabel('iterate');
ylabel('$\Vert f^\delta - K_i*U_i\Vert_{L^2}$',Interpreter='latex')
exportgraphics(gcf,strcat(dir,'/residual.pdf'),'ContentType','vector');
close

for l = 1:its_blind
    if l == 1
        u = images_blind(:,:,l);
        k = kernels_blind(:,:,l);
    else
        u = images_blind(:,:,l)-images_blind(:,:,l-1);
        k = kernels_blind(:,:,l) - kernels_blind(:,:,l-1);
    end

    %components
    figure('visible','off');
    imagesc(255*u); colormap(gray); axis off; pbaspect([m,n,1]);title(strcat('$u_{',num2str(l-1),'}$'),Interpreter="latex",FontSize=20);
    exportgraphics(gcf,strcat(dir,'/u_comp_',num2str(l),'.pdf'),'ContentType','vector');
    close

    figure('visible','off');
    surf(k); shading flat; view(70,5); title(strcat('$k_{',num2str(l-1),'}$'),Interpreter="latex",FontSize=20); zlim([-3e-3,16e-3]);
    exportgraphics(gcf,strcat(dir,'/k_comp_',num2str(l),'.pdf'));
    close

    %iterates
    figure('visible','off');
    imagesc(255*images_blind(:,:,l)); colormap(gray); axis off; pbaspect([m,n,1]);title(strcat('$U_{',num2str(l-1),'}$'),Interpreter="latex",FontSize=20);
    exportgraphics(gcf,strcat(dir,'/u_it_',num2str(l),'.pdf'),'ContentType','vector');
    close

    figure('visible','off');
    surf(kernels_blind(:,:,l)); shading flat; view(70,5); title(strcat('$K_{',num2str(l-1),'}$'),Interpreter="latex",FontSize=20); zlim([-3e-3,16e-3]);
    exportgraphics(gcf,strcat(dir,'/k_it_',num2str(l),'.pdf'));
    close

end


%%

for l = 1:length(alt_lambdas)
    figure('visible','off'); imagesc(u_alt(:,:,l)); colormap(gray); axis off; pbaspect([n,m,1]);
    exportgraphics(gcf,strcat(dir,'/u_alt_',num2str(l),'.pdf'));close;
    figure('visible','off'); surf(k_alt(:,:,l)); shading flat; view(70,5);
    exportgraphics(gcf,strcat(dir,'/k_alt_',num2str(l),'.pdf'));close;
end




