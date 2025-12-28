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

%creates tex code tables evaluating the experiments from
%non_blind_test.m; The code is stored in the strings
%tab_noise, tab_blur, tab_im

close all
clear
clc

load("data_non_blind.mat")


%%
percentage_psnr = psnr_u_MHDM./psnr_guess_end;
percentage_ssim = ssim_u_MHDM./ssim_guess_end;


indices_noise_levels = zeros(num_tests/num_noises,num_noises);
for l = 1:num_noises
    indices_noise_levels(:,l) = l:num_noises:num_tests;
end

indices_blurs = zeros(num_tests/num_kernels,num_kernels);
for l = 1:num_kernels
    indices_blurs(:,l) = find(ismember(mod(1:num_tests,num_kernels*num_noises),mod((1+(l-1)*num_noises:l*num_noises),num_kernels*num_noises)));
end

indices_ims = zeros(num_tests/num_Ims,num_Ims);
for l = 1:num_Ims
    indices_ims(:,l) = (1+(l-1)*num_noises*num_kernels):(l*num_noises*num_kernels);
end

% by noise
tab_noise= "\begin{tabular}{|c|c|c|}"+newline+"\hline $\sigma_{noise}$  & $\frac{\text{PSNR}_\text{blind}}{\text{PSNR}_{\text{guess}}}$ & $\frac{\text{SSIM}_\text{blind}}{\text{SSIM}_{\text{guess}}}$ \\ \hline"+ newline;
for l = 1:num_noises
    indices = indices_noise_levels (:,l);

    tab_noise = tab_noise + num2str(noise_levels(l),'%.16g');
    val_psnr = mean(percentage_psnr(indices));
    val_ssim = mean(percentage_ssim(indices));
    tab_noise = tab_noise + "&" + num2str(val_psnr,'%.3f') + "&" + num2str(val_ssim,'%.3f') +"\\";
end

tab_noise = tab_noise + "\hline" + newline + "\end{tabular}";

% by blur
tab_blur = "\begin{tabular}{|c|c|c|}"+newline+"\hline $kernel$  & $\frac{\text{PSNR}_\text{blind}}{\text{PSNR}_{\text{guess}}}$& $\frac{\text{SSIM}_\text{blind}}{\text{SSIM}_{\text{guess}}}$ \\ \hline"+ newline;
for l = 1:num_kernels
    indices = indices_blurs(:,l);
    val_psnr = mean(percentage_psnr(indices));
    val_ssim = mean(percentage_ssim(indices));
    tab_blur = tab_blur + num2str(l) + "&" + num2str(val_psnr,'%.3f')+ "&" + num2str(val_ssim,'%.3f') + "\\";
end
tab_blur = tab_blur  + "\hline" + newline + "\end{tabular}";

%by image
tab_im = "\begin{tabular}{|c|c|c|}"+newline+"\hline $kernel$  & $\frac{\text{PSNR}_\text{blind}}{\text{PSNR}_{\text{guess}}}$& $\frac{\text{SSIM}_\text{blind}}{\text{SSIM}_{\text{guess}}}$ \\ \hline"+ newline;
for l = 1:num_Ims
    indices = indices_ims(:,l);
    val_psnr = mean(percentage_psnr(indices));
    val_ssim = mean(percentage_ssim(indices));
    tab_im = tab_im + num2str(l) + "&" + num2str(val_psnr,'%.3f') + "&" + num2str(val_ssim,'%.3f')+ "\\";
end
tab_im = tab_im  + "\hline" + newline + "\end{tabular}";