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
%comparison_MHDM_variational.m; The code is stored in the strings
%tab_noise, tab_blur, tab_ims and tab_noise_cut, tab_blur_cut, tab_ims_cut 



close all
clear
clc

load("data_comp.mat");

%%


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

%table for noise level
tab_noise = "\begin{tabular}{|c|cc|cc|cc|}"+newline+"\hline $\sigma_{noise}$  &$\text{PSNR}_{\text{MHDM}}$ &$\text{PSNR}_{\text{var}}$ & $\text{SSIM}_{\text{MHDM}}$ & $\text{SSIM}_{\text{var}}$ &$\text{err}_{\text{MHDM}}$ &$\text{err}_{\text{var}}$ \\ \hline"+ newline;
for l = 1:num_noises
    indices = indices_noise_levels (:,l);

    tab_noise = tab_noise + num2str(noise_levels(l),'%.16g');

    vals_psnr_MHDM = psnr_u_MHDM(indices);
    avg_psnr_MHDM_noise(l) = mean(vals_psnr_MHDM);
    std_psnr_MHDM_noise(l) = std(vals_psnr_MHDM);

    vals_psnr_single = psnr_u_single(indices);
    avg_psnr_single_noise(l) = mean(vals_psnr_single);
    std_psnr_single_noise(l) = std(vals_psnr_single);

    if avg_psnr_MHDM_noise(l) > avg_psnr_single_noise(l)
        tab_noise = tab_noise + "&" + "\textbf{" + num2str(avg_psnr_MHDM_noise(l),'%.3f') + "} ("+ num2str(std_psnr_MHDM_noise(l),'%.3f') + ")" + "&" + num2str(avg_psnr_single_noise(l),'%.3f') + " (" + num2str(std_psnr_single_noise(l),'%.3f') + ")";
    else
        tab_noise = tab_noise + "&"  + num2str(avg_psnr_MHDM_noise(l),'%.3f') + " ("+ num2str(std_psnr_MHDM_noise(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_psnr_single_noise(l),'%.3f') + "} (" + num2str(std_psnr_single_noise(l),'%.3f') + ")";
    end

    vals_ssim_MHDM = ssim_u_MHDM(indices);
    avg_ssim_MHDM_noise(l) = mean(vals_ssim_MHDM);
    std_ssim_MHDM_noise(l) = std(vals_ssim_MHDM);

    vals_ssim_single = ssim_u_single(indices);
    avg_ssim_single_noise(l) = mean(vals_ssim_single);
    std_ssim_single_noise(l) = std(vals_ssim_single);

    if avg_ssim_MHDM_noise(l) > avg_ssim_single_noise(l)
        tab_noise = tab_noise + "&" + "\textbf{" + num2str(avg_ssim_MHDM_noise(l),'%.3f') + "} ("+ num2str(std_ssim_MHDM_noise(l),'%.3f') + ")" + "&" + num2str(avg_ssim_single_noise(l),'%.3f') + " (" + num2str(std_ssim_single_noise(l),'%.3f') + ")";
    else
        tab_noise = tab_noise + "&"  + num2str(avg_ssim_MHDM_noise(l),'%.3f') + " ("+ num2str(std_ssim_MHDM_noise(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_ssim_single_noise(l),'%.3f') + "} (" + num2str(std_ssim_single_noise(l),'%.3f') + ")";
    end

    vals_err_MHDM = err_ker_MHDM(indices);
    avg_err_MHDM_noise(l) = mean(vals_err_MHDM);
    std_err_MHDM_noise(l) = std(vals_err_MHDM);

    vals_err_single = err_ker_single(indices);
    avg_err_single_noise(l) = mean(vals_err_single);
    std_err_single_noise(l) = std(vals_err_single);

    if avg_err_MHDM_noise(l) < avg_err_single_noise(l)
        tab_noise = tab_noise + "&" + "\textbf{" + num2str(avg_err_MHDM_noise(l),'%.3f') + "} ("+ num2str(std_err_MHDM_noise(l),'%.3f') + ")" + "&" + num2str(avg_err_single_noise(l),'%.3f') + " (" + num2str(std_err_single_noise(l),'%.3f') + ")";
    else
        tab_noise = tab_noise + "&"  + num2str(avg_err_MHDM_noise(l),'%.3f') + " ("+ num2str(std_err_MHDM_noise(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_err_single_noise(l),'%.3f') + "} (" + num2str(std_err_single_noise(l),'%.3f') + ")";
    end

    tab_noise = tab_noise + "\\" ;

end
tab_noise = tab_noise + "\hline" + newline + "\end{tabular}";

%table for blur
tab_blur = "\begin{tabular}{|c|cc|cc|cc|}"+newline+"\hline kernel  &$\text{PSNR}_{\text{MHDM}}$ &$\text{PSNR}_{\text{var}}$ & $\text{SSIM}_{\text{MHDM}}$ & $\text{SSIM}_{\text{var}}$ &$\text{err}_{\text{MHDM}}$ &$\text{err}_{\text{var}}$ \\ \hline"+ newline;
for l = 1:num_kernels
    indices = indices_blurs (:,l);
    tab_blur = tab_blur + num2str(l);

    vals_psnr_MHDM = psnr_u_MHDM(indices);
    avg_psnr_MHDM_blur(l) = mean(vals_psnr_MHDM);
    std_psnr_MHDM_blur(l) = std(vals_psnr_MHDM);

    vals_psnr_single = psnr_u_single(indices);
    avg_psnr_single_blur(l) = mean(vals_psnr_single);
    std_psnr_single_blur(l) = std(vals_psnr_single);

    if avg_psnr_MHDM_blur(l) > avg_psnr_single_blur(l)
        tab_blur = tab_blur + "&" + "\textbf{" + num2str(avg_psnr_MHDM_blur(l),'%.3f') + "} ("+ num2str(std_psnr_MHDM_blur(l),'%.3f') + ")" + "&" + num2str(avg_psnr_single_blur(l),'%.3f') + " (" + num2str(std_psnr_single_blur(l),'%.3f') + ")";
    else
        tab_blur = tab_blur + "&"  + num2str(avg_psnr_MHDM_blur(l),'%.3f') + " ("+ num2str(std_psnr_MHDM_blur(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_psnr_single_blur(l),'%.3f') + "} (" + num2str(std_psnr_single_blur(l),'%.3f') + ")";
    end

    vals_ssim_MHDM = ssim_u_MHDM(indices);
    avg_ssim_MHDM_blur(l) = mean(vals_ssim_MHDM);
    std_ssim_MHDM_blur(l) = std(vals_ssim_MHDM);

    vals_ssim_single = ssim_u_single(indices);
    avg_ssim_single_blur(l) = mean(vals_ssim_single);
    std_ssim_single_blur(l) = std(vals_ssim_single);

    if avg_ssim_MHDM_blur(l) > avg_ssim_single_blur(l)
        tab_blur = tab_blur + "&" + "\textbf{" + num2str(avg_ssim_MHDM_blur(l),'%.3f') + "} ("+ num2str(std_ssim_MHDM_blur(l),'%.3f') + ")" + "&" + num2str(avg_ssim_single_blur(l),'%.3f') + " (" + num2str(std_ssim_single_blur(l),'%.3f') + ")";
    else
        tab_blur = tab_blur + "&"  + num2str(avg_ssim_MHDM_blur(l),'%.3f') + " ("+ num2str(std_ssim_MHDM_blur(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_ssim_single_blur(l),'%.3f') + "} (" + num2str(std_ssim_single_blur(l),'%.3f') + ")";
    end

    vals_err_MHDM = err_ker_MHDM(indices);
    avg_err_MHDM_blur(l) = mean(vals_err_MHDM);
    std_err_MHDM_blur(l) = std(vals_err_MHDM);

    vals_err_single = err_ker_single(indices);
    avg_err_single_blur(l) = mean(vals_err_single);
    std_err_single_blur(l) = std(vals_err_single);

    if avg_err_MHDM_blur(l) < avg_err_single_blur(l)
        tab_blur = tab_blur + "&" + "\textbf{" + num2str(avg_err_MHDM_blur(l),'%.3f') + "} ("+ num2str(std_err_MHDM_blur(l),'%.3f') + ")" + "&" + num2str(avg_err_single_blur(l),'%.3f') + " (" + num2str(std_err_single_blur(l),'%.3f') + ")";
    else
        tab_blur = tab_blur + "&"  + num2str(avg_err_MHDM_blur(l),'%.3f') + " ("+ num2str(std_err_MHDM_blur(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_err_single_blur(l),'%.3f') + "} (" + num2str(std_err_single_blur(l),'%.3f') + ")";
    end

    tab_blur = tab_blur + "\\" ;


end

tab_blur = tab_blur + "\hline" + newline + "\end{tabular}";

%table for images
tab_ims = "\begin{tabular}{|c|cc|cc|cc|}"+newline+"\hline image  &$\text{PSNR}_{\text{MHDM}}$ &$\text{PSNR}_{\text{var}}$ & $\text{SSIM}_{\text{MHDM}}$ & $\text{SSIM}_{\text{var}}$ &$\text{err}_{\text{MHDM}}$ &$\text{err}_{\text{var}}$ \\ \hline"+ newline;
for l = 1:num_Ims
    indices = indices_ims (:,l);
    tab_ims = tab_ims + num2str(l);

    vals_psnr_MHDM = psnr_u_MHDM(indices);
    avg_psnr_MHDM_ims(l) = mean(vals_psnr_MHDM);
    std_psnr_MHDM_ims(l) = std(vals_psnr_MHDM);

    vals_psnr_single = psnr_u_single(indices);
    avg_psnr_single_ims(l) = mean(vals_psnr_single);
    std_psnr_single_ims(l) = std(vals_psnr_single);

        if avg_psnr_MHDM_ims(l) > avg_psnr_single_ims(l)
        tab_ims = tab_ims + "&" + "\textbf{" + num2str(avg_psnr_MHDM_ims(l),'%.3f') + "} ("+ num2str(std_psnr_MHDM_ims(l),'%.3f') + ")" + "&" + num2str(avg_psnr_single_ims(l),'%.3f') + " (" + num2str(std_psnr_single_ims(l),'%.3f') + ")";
    else
        tab_ims = tab_ims + "&"  + num2str(avg_psnr_MHDM_ims(l),'%.3f') + " ("+ num2str(std_psnr_MHDM_ims(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_psnr_single_ims(l),'%.3f') + "} (" + num2str(std_psnr_single_ims(l),'%.3f') + ")";
    end

    vals_ssim_MHDM = ssim_u_MHDM(indices);
    avg_ssim_MHDM_ims(l) = mean(vals_ssim_MHDM);
    std_ssim_MHDM_ims(l) = std(vals_ssim_MHDM);

    vals_ssim_single = ssim_u_single(indices);
    avg_ssim_single_ims(l) = mean(vals_ssim_single);
    std_ssim_single_ims(l) = std(vals_ssim_single);

    if avg_ssim_MHDM_ims(l) > avg_ssim_single_ims(l)
        tab_ims = tab_ims + "&" + "\textbf{" + num2str(avg_ssim_MHDM_ims(l),'%.3f') + "} ("+ num2str(std_ssim_MHDM_ims(l),'%.3f') + ")" + "&" + num2str(avg_ssim_single_ims(l),'%.3f') + " (" + num2str(std_ssim_single_ims(l),'%.3f') + ")";
    else
        tab_ims = tab_ims + "&"  + num2str(avg_ssim_MHDM_ims(l),'%.3f') + " ("+ num2str(std_ssim_MHDM_ims(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_ssim_single_ims(l),'%.3f') + "} (" + num2str(std_ssim_single_ims(l),'%.3f') + ")";
    end

    vals_err_MHDM = err_ker_MHDM(indices);
    avg_err_MHDM_ims(l) = mean(vals_err_MHDM);
    std_err_MHDM_ims(l) = std(vals_err_MHDM);

    vals_err_single = err_ker_single(indices);
    avg_err_single_ims(l) = mean(vals_err_single);
    std_err_single_ims(l) = std(vals_err_single);

    if avg_err_MHDM_ims(l) < avg_err_single_ims(l)
        tab_ims = tab_ims + "&" + "\textbf{" + num2str(avg_err_MHDM_ims(l),'%.3f') + "} ("+ num2str(std_err_MHDM_ims(l),'%.3f') + ")" + "&" + num2str(avg_err_single_ims(l),'%.3f') + " (" + num2str(std_err_single_ims(l),'%.3f') + ")";
    else
        tab_ims = tab_ims + "&"  + num2str(avg_err_MHDM_ims(l),'%.3f') + " ("+ num2str(std_err_MHDM_ims(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_err_single_ims(l),'%.3f') + "} (" + num2str(std_err_single_ims(l),'%.3f') + ")";
    end

    tab_ims = tab_ims + "\\" ;


end

tab_ims = tab_ims + "\hline" + newline + "\end{tabular}";

%%
%tables wit outliers omitted
for l = 1:num_tests

    f = cell2mat(f_arr(l));
    u_true = cell2mat(u_true_arr(l));

    delta = cell2mat(delta_arr(l));

    stopping  = tau*delta;

    u_MHDM = cell2mat(u_end_MHDM_arr(l));
    k_MHDM = cell2mat(k_end_MHDM_arr(l));

    u_single = cell2mat(u_end_single_arr(l));
    k_single = cell2mat(k_end_single_arr(l));

    img_MHDM = real(ifft2(fft2(u_MHDM).*psf2otf(k_MHDM)));
    img_single = real(ifft2(fft2(u_single).*psf2otf(k_single)));

    res_MHDM(l) = l2(f-img_MHDM);
    res_single(l) = l2(f-img_single);

    quot_res_MHDM(l) =  res_MHDM(l)/stopping;
    quot_res_single(l) =  res_single(l)/stopping;
end

outliers = find(abs(quot_res_single.^1 - quot_res_MHDM.^1)>0.1);


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

%table for noise level
tab_noise_cut = "\begin{tabular}{|c|cc|cc|cc|}"+newline+"\hline $\sigma_{noise}$  &$\text{PSNR}_{\text{MHDM}}$ &$\text{PSNR}_{\text{var}}$ & $\text{SSIM}_{\text{MHDM}}$ & $\text{SSIM}_{\text{var}}$ &$\text{err}_{\text{MHDM}}$ &$\text{err}_{\text{var}}$ \\ \hline"+ newline;
for l = 1:num_noises
    indices = setdiff(indices_noise_levels (:,l),outliers);

    tab_noise_cut = tab_noise_cut + num2str(noise_levels(l));

    vals_psnr_MHDM = psnr_u_MHDM(indices);
    avg_psnr_MHDM_noise_cut(l) = mean(vals_psnr_MHDM);
    std_psnr_MHDM_noise_cut(l) = std(vals_psnr_MHDM);

    vals_psnr_single = psnr_u_single(indices);
    avg_psnr_single_noise_cut(l) = mean(vals_psnr_single);
    std_psnr_single_noise_cut(l) = std(vals_psnr_single);

    if avg_psnr_MHDM_noise_cut(l) > avg_psnr_single_noise_cut(l)
        tab_noise_cut = tab_noise_cut + "&" + "\textbf{" + num2str(avg_psnr_MHDM_noise_cut(l),'%.3f') + "} ("+ num2str(std_psnr_MHDM_noise_cut(l),'%.3f') + ")" + "&" + num2str(avg_psnr_single_noise_cut(l),'%.3f') + " (" + num2str(std_psnr_single_noise_cut(l),'%.3f') + ")";
    else
        tab_noise_cut = tab_noise_cut + "&"  + num2str(avg_psnr_MHDM_noise_cut(l),'%.3f') + " ("+ num2str(std_psnr_MHDM_noise_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_psnr_single_noise_cut(l),'%.3f') + "} (" + num2str(std_psnr_single_noise_cut(l),'%.3f') + ")";
    end

    vals_ssim_MHDM = ssim_u_MHDM(indices);
    avg_ssim_MHDM_noise_cut(l) = mean(vals_ssim_MHDM);
    std_ssim_MHDM_noise_cut(l) = std(vals_ssim_MHDM);

    vals_ssim_single = ssim_u_single(indices);
    avg_ssim_single_noise_cut(l) = mean(vals_ssim_single);
    std_ssim_single_noise_cut(l) = std(vals_ssim_single);

    if avg_ssim_MHDM_noise_cut(l) > avg_ssim_single_noise_cut(l)
        tab_noise_cut = tab_noise_cut + "&" + "\textbf{" + num2str(avg_ssim_MHDM_noise_cut(l),'%.3f') + "} ("+ num2str(std_ssim_MHDM_noise_cut(l),'%.3f') + ")" + "&" + num2str(avg_ssim_single_noise_cut(l),'%.3f') + " (" + num2str(std_ssim_single_noise_cut(l),'%.3f') + ")";
    else
        tab_noise_cut = tab_noise_cut + "&"  + num2str(avg_ssim_MHDM_noise_cut(l),'%.3f') + " ("+ num2str(std_ssim_MHDM_noise_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_ssim_single_noise_cut(l),'%.3f') + "} (" + num2str(std_ssim_single_noise_cut(l),'%.3f') + ")";
    end

    vals_err_MHDM = err_ker_MHDM(indices);
    avg_err_MHDM_noise_cut(l) = mean(vals_err_MHDM);
    std_err_MHDM_noise_cut(l) = std(vals_err_MHDM);

    vals_err_single = err_ker_single(indices);
    avg_err_single_noise_cut(l) = mean(vals_err_single);
    std_err_single_noise_cut(l) = std(vals_err_single);

    if avg_err_MHDM_noise_cut(l) < avg_err_single_noise_cut(l)
        tab_noise_cut = tab_noise_cut + "&" + "\textbf{" + num2str(avg_err_MHDM_noise_cut(l),'%.3f') + "} ("+ num2str(std_err_MHDM_noise_cut(l),'%.3f') + ")" + "&" + num2str(avg_err_single_noise_cut(l),'%.3f') + " (" + num2str(std_err_single_noise_cut(l),'%.3f') + ")";
    else
        tab_noise_cut = tab_noise_cut + "&"  + num2str(avg_err_MHDM_noise_cut(l),'%.3f') + " ("+ num2str(std_err_MHDM_noise_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_err_single_noise_cut(l),'%.3f') + "} (" + num2str(std_err_single_noise_cut(l),'%.3f') + ")";
    end

    tab_noise_cut = tab_noise_cut + "\\" ;

end
tab_noise_cut = tab_noise_cut + "\hline" + newline + "\end{tabular}";

%table for blur
tab_blur_cut = "\begin{tabular}{|c|cc|cc|cc|}"+newline+"\hline kernel  &$\text{PSNR}_{\text{MHDM}}$ &$\text{PSNR}_{\text{var}}$ & $\text{SSIM}_{\text{MHDM}}$ & $\text{SSIM}_{\text{var}}$ &$\text{err}_{\text{MHDM}}$ &$\text{err}_{\text{var}}$ \\ \hline"+ newline;
for l = 1:num_kernels
    indices = setdiff(indices_blurs(:,l),outliers);
    tab_blur_cut = tab_blur_cut + num2str(l);

    vals_psnr_MHDM = psnr_u_MHDM(indices);
    avg_psnr_MHDM_blur_cut(l) = mean(vals_psnr_MHDM);
    std_psnr_MHDM_blur_cut(l) = std(vals_psnr_MHDM);

    vals_psnr_single = psnr_u_single(indices);
    avg_psnr_single_blur_cut(l) = mean(vals_psnr_single);
    std_psnr_single_blur_cut(l) = std(vals_psnr_single);

    if avg_psnr_MHDM_blur_cut(l) > avg_psnr_single_blur_cut(l)
        tab_blur_cut = tab_blur_cut + "&" + "\textbf{" + num2str(avg_psnr_MHDM_blur_cut(l),'%.3f') + "} ("+ num2str(std_psnr_MHDM_blur_cut(l),'%.3f') + ")" + "&" + num2str(avg_psnr_single_blur_cut(l),'%.3f') + " (" + num2str(std_psnr_single_blur_cut(l),'%.3f') + ")";
    else
        tab_blur_cut = tab_blur_cut + "&"  + num2str(avg_psnr_MHDM_blur_cut(l),'%.3f') + " ("+ num2str(std_psnr_MHDM_blur_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_psnr_single_blur_cut(l),'%.3f') + "} (" + num2str(std_psnr_single_blur_cut(l),'%.3f') + ")";
    end

    vals_ssim_MHDM = ssim_u_MHDM(indices);
    avg_ssim_MHDM_blur_cut(l) = mean(vals_ssim_MHDM);
    std_ssim_MHDM_blur_cut(l) = std(vals_ssim_MHDM);

    vals_ssim_single = ssim_u_single(indices);
    avg_ssim_single_blur_cut(l) = mean(vals_ssim_single);
    std_ssim_single_blur_cut(l) = std(vals_ssim_single);

    if avg_ssim_MHDM_blur_cut(l) > avg_ssim_single_blur_cut(l)
        tab_blur_cut = tab_blur_cut + "&" + "\textbf{" + num2str(avg_ssim_MHDM_blur_cut(l),'%.3f') + "} ("+ num2str(std_ssim_MHDM_blur_cut(l),'%.3f') + ")" + "&" + num2str(avg_ssim_single_blur_cut(l),'%.3f') + " (" + num2str(std_ssim_single_blur_cut(l),'%.3f') + ")";
    else
        tab_blur_cut = tab_blur_cut + "&"  + num2str(avg_ssim_MHDM_blur_cut(l),'%.3f') + " ("+ num2str(std_ssim_MHDM_blur_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_ssim_single_blur_cut(l),'%.3f') + "} (" + num2str(std_ssim_single_blur_cut(l),'%.3f') + ")";
    end

    vals_err_MHDM = err_ker_MHDM(indices);
    avg_err_MHDM_blur_cut(l) = mean(vals_err_MHDM);
    std_err_MHDM_blur_cut(l) = std(vals_err_MHDM);

    vals_err_single = err_ker_single(indices);
    avg_err_single_blur_cut(l) = mean(vals_err_single);
    std_err_single_blur_cut(l) = std(vals_err_single);

    if avg_err_MHDM_blur_cut(l) < avg_err_single_blur_cut(l)
        tab_blur_cut = tab_blur_cut + "&" + "\textbf{" + num2str(avg_err_MHDM_blur_cut(l),'%.3f') + "} ("+ num2str(std_err_MHDM_blur_cut(l),'%.3f') + ")" + "&" + num2str(avg_err_single_blur_cut(l),'%.3f') + " (" + num2str(std_err_single_blur_cut(l),'%.3f') + ")";
    else
        tab_blur_cut = tab_blur_cut + "&"  + num2str(avg_err_MHDM_blur_cut(l),'%.3f') + " ("+ num2str(std_err_MHDM_blur_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_err_single_blur_cut(l),'%.3f') + "} (" + num2str(std_err_single_blur_cut(l),'%.3f') + ")";
    end

    tab_blur_cut = tab_blur_cut + "\\" ;


end

tab_blur_cut = tab_blur_cut + "\hline" + newline + "\end{tabular}";

%table for images
tab_ims_cut = "\begin{tabular}{|c|cc|cc|cc|}"+newline+"\hline image  &$\text{PSNR}_{\text{MHDM}}$ &$\text{PSNR}_{\text{var}}$ & $\text{SSIM}_{\text{MHDM}}$ & $\text{SSIM}_{\text{var}}$ &$\text{err}_{\text{MHDM}}$ &$\text{err}_{\text{var}}$ \\ \hline"+ newline;
for l = 1:num_Ims
    indices = setdiff(indices_ims(:,l),outliers);
    tab_ims = tab_ims + num2str(l);

    vals_psnr_MHDM = psnr_u_MHDM(indices);
    avg_psnr_MHDM_ims_cut(l) = mean(vals_psnr_MHDM);
    std_psnr_MHDM_ims_cut(l) = std(vals_psnr_MHDM);

    vals_psnr_single = psnr_u_single(indices);
    avg_psnr_single_ims_cut(l) = mean(vals_psnr_single);
    std_psnr_single_ims_cut(l) = std(vals_psnr_single);

        if avg_psnr_MHDM_ims_cut(l) > avg_psnr_single_ims_cut(l)
        tab_ims_cut = tab_ims_cut + "&" + "\textbf{" + num2str(avg_psnr_MHDM_ims_cut(l),'%.3f') + "} ("+ num2str(std_psnr_MHDM_ims_cut(l),'%.3f') + ")" + "&" + num2str(avg_psnr_single_ims_cut(l),'%.3f') + " (" + num2str(std_psnr_single_ims_cut(l),'%.3f') + ")";
    else
        tab_ims_cut = tab_ims_cut + "&"  + num2str(avg_psnr_MHDM_ims_cut(l),'%.3f') + " ("+ num2str(std_psnr_MHDM_ims_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_psnr_single_ims_cut(l),'%.3f') + "} (" + num2str(std_psnr_single_ims_cut(l),'%.3f') + ")";
    end

    vals_ssim_MHDM = ssim_u_MHDM(indices);
    avg_ssim_MHDM_ims_cut(l) = mean(vals_ssim_MHDM);
    std_ssim_MHDM_ims_cut(l) = std(vals_ssim_MHDM);

    vals_ssim_single = ssim_u_single(indices);
    avg_ssim_single_ims_cut(l) = mean(vals_ssim_single);
    std_ssim_single_ims_cut(l) = std(vals_ssim_single);

    if avg_ssim_MHDM_ims_cut(l) > avg_ssim_single_ims_cut(l)
        tab_ims_cut = tab_ims_cut + "&" + "\textbf{" + num2str(avg_ssim_MHDM_ims_cut(l),'%.3f') + "} ("+ num2str(std_ssim_MHDM_ims_cut(l),'%.3f') + ")" + "&" + num2str(avg_ssim_single_ims_cut(l),'%.3f') + " (" + num2str(std_ssim_single_ims_cut(l),'%.3f') + ")";
    else
        tab_ims_cut = tab_ims_cut + "&"  + num2str(avg_ssim_MHDM_ims_cut(l),'%.3f') + " ("+ num2str(std_ssim_MHDM_ims_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_ssim_single_ims_cut(l),'%.3f') + "} (" + num2str(std_ssim_single_ims_cut(l),'%.3f') + ")";
    end

    vals_err_MHDM = err_ker_MHDM(indices);
    avg_err_MHDM_ims_cut(l) = mean(vals_err_MHDM);
    std_err_MHDM_ims_cut(l) = std(vals_err_MHDM);

    vals_err_single = err_ker_single(indices);
    avg_err_single_ims_cut(l) = mean(vals_err_single);
    std_err_single_ims_cut(l) = std(vals_err_single);

    if avg_err_MHDM_ims_cut(l) < avg_err_single_ims_cut(l)
        tab_ims_cut = tab_ims_cut + "&" + "\textbf{" + num2str(avg_err_MHDM_ims_cut(l),'%.3f') + "} ("+ num2str(std_err_MHDM_ims_cut(l),'%.3f') + ")" + "&" + num2str(avg_err_single_ims_cut(l),'%.3f') + " (" + num2str(std_err_single_ims_cut(l),'%.3f') + ")";
    else
        tab_ims_cut = tab_ims_cut + "&"  + num2str(avg_err_MHDM_ims_cut(l),'%.3f') + " ("+ num2str(std_err_MHDM_ims_cut(l),'%.3f') + ")" + "&" + "\textbf{" + num2str(avg_err_single_ims_cut(l),'%.3f') + "} (" + num2str(std_err_single_ims_cut(l),'%.3f') + ")";
    end

    tab_ims_cut = tab_ims_cut + "\\" ;


end

tab_ims_cut = tab_ims_cut + "\hline" + newline + "\end{tabular}";




