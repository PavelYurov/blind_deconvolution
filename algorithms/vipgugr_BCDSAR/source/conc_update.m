function [CT,SigmaC]  = conc_update(YT,CT,M,SigmaM,FTF,beta,alpha)
    ns = size(CT,1);
    tamm = size(YT,2);

    SigmaC = zeros(tamm,ns);
  
    for s=1:ns
        [zminus, ~] = computingEsZs(YT,CT,M);
        expM2 = M(:,s)' * M(:,s) + 3.0 * SigmaM(s);
        auxSigmaC = 1.0 ./( beta * expM2 + alpha(s) * FTF);
        Fzminuss = fft2(reshape(zminus(:,s), size(FTF)));
        cs = ifft2(beta .* auxSigmaC .* Fzminuss);
        CT(s,:) = cs(:)';
        SigmaC(:,s) = auxSigmaC(:);
        %CT(CT < eps) = eps; %%%%% Yo no forzaba no negatividad en cada vuelta. La forzaba al final, en todo caso
    end
end

