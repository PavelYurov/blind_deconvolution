function alpha = alpha_update(CT,FTF,SigmaC)

    ns = size(CT,1);
    tamm = size(CT,2);
    
    alpha = zeros(ns,1);
    for s=1:ns
        tmp = fft2(reshape(CT(s,:),size(FTF)));
        tmp = conj(tmp).* FTF.* tmp;
        norma = sum(tmp(:))/tamm;
        
        tmp = SigmaC(s) .* FTF;
        traza = sum(tmp(:));
        alpha(s) = tamm/(norma + traza);
    end

end

