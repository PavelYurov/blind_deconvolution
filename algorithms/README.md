# Путеводитель по алгоритмам

## Собственные реализации

Сгруппированы по типу подхода в `implementations/`:

### `classic/` — классические итеративные методы

- [`richardson_lucy.py`](implementations/classic/richardson_lucy.py) — Richardson–Lucy (blind/non-blind; пуассоновское MLE) [2][3]
- [`alternating_minimization.py`](implementations/classic/alternating_minimization.py) — MAP blind deconvolution с Huber-регуляризацией, alternating minimization [8][21][30]
- [`expectation_maximization.py`](implementations/classic/expectation_maximization.py) — EM для идентификации blur/восстановления изображения [18][19]

### `bayesian/` — байесовские методы

- [`vbbid_tv.py`](implementations/bayesian/vbbid_tv.py) — Variational Bayesian blind deconvolution с TV prior [10]
- [`bbd_deip.py`](implementations/bayesian/bbd_deip.py) — Bayesian blind deconvolution по паре изображений с разной экспозицией [11]
- [`sb_bid_pe.py`](implementations/bayesian/sb_bid_pe.py) — Sparse Bayesian blind deconvolution с оценкой параметров [12]

### `variational/` — вариационные методы

- [`vabid.py`](implementations/variational/vabid.py) — Variational Bayes blind deconvolution (VAR3, alternating variational) [14]
- [`vapibe.py`](implementations/variational/vapibe.py) — Variational Bayes для совместной оценки параметров/изображения/blur [15]
- [`vbsk_sid_st.py`](implementations/variational/vbsk_sid_st.py) — VB sparse kernel-based blind deconvolution со Student’s-t priors [16]

### `sparse/` — разреженные/TV-подобные регуляризации

- [`vbc_bid.py`](implementations/sparse/vbc_bid.py) — VB compressive blind image deconvolution (VB-ADMM, wavelet sparsity) [13]
- [`pbtvgr.py`](implementations/sparse/pbtvgr.py) — blind deconvolution для Poisson noise: TV + L0 gradient regularization [24]

## Внешние обёртки и источники

Ниже — алгоритмы, которые взяты из других репозиториев.

### `blind_deconvolution/`

- [`BYchao100_Graph_Based_Blind_Image_Deblurring`](https://github.com/BYchao100/Graph-Based-Blind-Image-Deblurring/) — graph-based blind deblurring (оценка PSF через графовую модель/регуляризацию) [8][35]
- [`KanuGaba_Image_Deblurrer`](https://github.com/KanuGaba/Image-Deblurrer/) — Wiener deconvolution (не слепая, motion PSF `LEN/THETA`) [1]
- [`MaxMB_Image_Restoration_Wiener_Blind`](https://github.com/MaxMB/Image_Restoration_Wiener_Blind) — Wiener optimal filtering + baseline `deconvlucy`/`deconvblind` (MATLAB) [1][2][3][6]
- [`TobiasWolf_math_Blind_Deconvolution_MHDM`](https://github.com/TobiasWolf-math/Blind-Deconvolution-MHDM) — multiscale hierarchical decomposition method (MHDM) для blind deconvolution [6][8]
- [`Youngforgithub_Deblurring_Text_Images_via_L0_Regularized_Intensity_and_Gradient`](https://github.com/Youngforgithub/Deblurring-Text-Images-via-L0-Regularized-Intensity-and-Gradient) — L0-regularized intensity+gradient + TV prior (text deblurring) [20][23]
- [`adamalavi_Blind_motion_deblurring_for_license_plates`](https://github.com/ankitVP77/Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning) — CNN-оценка параметров motion PSF (длина/угол по FFT) + Wiener deconvolution [1][5]
- [`alexis_mignon_pydeconv`](https://github.com/alexis-mignon/pydeconv) — MAP blind deconvolution с priors на градиенты (global/local), оптимизация по изображению и PSF [8][9]
- [`ceciledellavalle_BlindDeconvolution`](https://github.com/ceciledellavalle/BlindDeconvolution) — классический blind deconvolution (итеративная оптимизация по изображению и PSF) [6][8]
- [`crewleader_BlindDeconvolutionLowRank`](https://github.com/crewleader/BlindDeconvolutionLowRank) — multi-image blind deconvolution через low-rank representation [36]
- [`dragos2001_Total_Variation_Blind_Deconvolution`](https://github.com/dragos2001/Total_Variation_Blind_Deconvolution) — Total Variation blind deconvolution (Chan–Wong PCG / Perrone–Favaro GD-вариант) [20][21][37]
- [`felixwempe_blind_deconvolution`](https://github.com/felixwempe/blind_deconvolution/) — blind deconvolution через convex programming (Ahmed–Recht–Romberg, CVX) [27][28]
- [`huacheng_Shift_Invariant_Deblurring`](https://github.com/huacheng/Shift-Invariant-Deblurring) — two-phase kernel estimation (Xu et al.) + shock/nonlinear diffusion preprocessing + Shan non-blind deconvolution [20][23]
- [`jtaoz_GKPILE_Deconvolution`](https://github.com/jtaoz/GKPILE-Deconvolution) — generative-based kernel prior + initializer via latent encoding (GKPILE) [5][6][8]
- [`lisiyaoATbnu_low_rank_kernel`](https://github.com/lisiyaoATbnu/low_rank_kernel) — oversized-kernel suppression через low-rank kernel prior; non-blind часть: hyper-Laplacian prior (Krishnan–Fergus) [25][36]
- [`muhammadhamzaazhar_Image_Enhancement_Filters`](https://github.com/muhammadhamzaazhar/Image-Enhancement-Filters) — набор методов: blind deconvolution + Lucy–Richardson + Wiener filtering [1][2][3][6]
- [`mujib2020_Non_blind_and_Blind_Deconvolution_under_Poisson_noise`](https://github.com/mujib2020/Non-blind-and-Blind-Deconvolution-under-Poisson-noise) — EM blind deconvolution и fractional-order total variation (FOTV) под Poisson noise [17][22]
- [`panpanfei_Phase_only_Image_Based_Kernel_Estimation_for_Blind_Motion_Deblurring`](https://github.com/panpanfei/Phase-only-Image-Based-Kernel-Estimation-for-Blind-Motion-Deblurring/) — phase-only kernel estimation (CVPR’19) + L0/TV регуляризация [20][23][34]
- [`qingqu06_MCS_BD`](https://github.com/qingqu06/MCS-BD) — multichannel sparse blind deconvolution (неconvex оптимизация/ландшафт) [33]
- [`vipgugr_BCDSAR`](https://github.com/vipgugr/BCDSAR) — variational Bayesian blind color deconvolution (VB inference) [9]
- [`warrenzha_blind_deconvolution`](https://github.com/warrenzha/blind-deconvolution) — blind deconvolution через convex optimization (в т.ч. 2D-версия) [27][28]
- [`yenhsunlin_blind_deconv`](https://github.com/yenhsunlin/blind_deconv) — Bayesian blind deconvolution с hyper-Laplacian \(L_p, 0<p<1\) регуляризацией (Kotera et al.) [9][25]
- [`zalteck_BCDTV`](https://github.com/zalteck/BCDTV) — TV-based variational Bayesian blind color deconvolution (TV prior + VB/ELBO) [9][20][21]

### `nonblind_deconvolution/`

- [`ztCao_Variational_Bayesian_Blind_Deconvolution_Using_a_Total_Variation_Prior`](https://github.com/2924878374/Variational-Bayesian-Blind-Deconvolution-Using-a-Total-Variation-Prior) — Variational Bayesian blind deconvolution с Total Variation prior (TV1/TV2) [9][20][21]

### `unsorted/`

- [`_23ms410_Blind_Deconvolution`](https://github.com/23ms410/Blind-Deconvolution) — blind deconvolution (базовая постановка/демо) [6][8]
- [`ADY_YDA_Iterative_Blind_Image_Deconvolution`](https://github.com/ADY-YDA/Iterative-Blind-Image-Deconvolution/blob/main/Expectation-Maximization.ipynb) — Expectation–Maximization (EM) для blind deconvolution [17]
- [`CACTuS_AI_Blind_Deconvolution_using_Modulated_Inputs`](https://github.com/CACTuS-AI/Blind-Deconvolution-using-Modulated-Inputs) — blind deconvolution with modulated inputs (bilinear inverse problem / lifting) [28]
- [`CEA_jiangming_DecGMCA`](https://github.com/CEA-jiangming/DecGMCA) — deconvolution + GMCA (sparsity-based demixing / morphological component analysis) [26]
- [`COR_OPT_RobustBlindDeconv`](https://github.com/COR-OPT/RobustBlindDeconv) — robust blind deconvolution: subgradient method и prox-linear method [29][31]
- [`Drorharush_SBD`](https://github.com/Drorharush/SBD) — sparse blind deconvolution (sparsity-driven kernel/latent estimation) [26]
- [`GeekLogan_pyBlindRL`](https://github.com/GeekLogan/pyBlindRL) — blind Richardson–Lucy deconvolution (Poisson likelihood) [2][3][7]
- [`Tmodrzyk_richardson_lucy_python`](https://github.com/Tmodrzyk/richardson-lucy-python) — Richardson–Lucy deconvolution (non-blind) [2][3]
- [`adamalavi_Blind_motion_deblurring_for_license_plates`](https://github.com/ankitVP77/Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning) — CNN-оценка motion PSF + Wiener deconvolution [1][5]
- [`axium_Blind_Image_Deconvolution_using_Deep_Generative_Priors`](https://github.com/axium/Blind-Image-Deconvolution-using-Deep-Generative-Priors) — blind deconvolution с deep generative priors (оптимизация в латентных пространствах генераторов) [5][8]
- [`deu439_sbl_blind_deconvolution`](https://github.com/deu439/sbl-blind-deconvolution) — sparse blind deconvolution (sparsity prior) [26]
- [`fabioviggiano_BlindDeconvolution`](https://github.com/fabioviggiano/BlindDeconvolution) — blind deconvolution (практическая реализация/демо) [6][8]
- [`gandor26_blind_deconvolution_through_a_single_image`](https://github.com/Gandor26/Blind-Deconvolution-through-a-Single-Image) — blind deconvolution из одного изображения (single-image blind deblurring) [6][8]
- [`idiap_semiblindpsfdeconv`](https://github.com/idiap/semiblindpsfdeconv) — semi-blind PSF deconvolution (частично параметризованный PSF) [6][8]
- [`jeffreysblake_funsearch_blind_deconvolution`](https://github.com/jeffreysblake/funsearch-blind-deconvolution) — blind deconvolution (поиск/генерация процедур, экспериментально) [5]
- [`jhell96_Deep_Blind_Deblur`](https://github.com/jhell96/Deep-Blind-Deblur) — deep learning blind deblurring (нейросетевая модель) [5][38]
- [`luczeng_MotionBlur`](https://github.com/luczeng/MotionBlur) — моделирование/инверсия motion blur (forward model + deblurring) [6]
- [`mujib2020_Non_blind_and_Blind_Deconvolution_under_Poisson_noise`](https://github.com/mujib2020/Non-blind-and-Blind-Deconvolution-under-Poisson-noise) — EM и fractional-order TV (FOTV) при Poisson noise [17][22]
- [`sanghviyashiitb_photon_limited_blind`](https://github.com/sanghviyashiitb/photon-limited-blind) — photon-limited blind deconvolution: unsupervised iterative kernel estimation (P4IP, plug-and-play/iterative scheme) [5][8]
- [`tianyishan_Blind_Deconvolution`](https://github.com/tianyishan/Blind_Deconvolution) — PRIDA (Provably Robust Image Deconvolution Algorithm), mirror descent [32]
- [`ys_koshelev_nla_deblur`](https://github.com/ys-koshelev/nla_deblur) — text deblurring: kernel estimation + L0/TV регуляризация (проект Yang) [20][23]

## Литература

### Классика и обзоры

1. **Wiener, N.** — *Extrapolation, Interpolation, and Smoothing of Stationary Time Series* (классическая теория оптимальной линейной фильтрации, Wiener filter)
	([direct.mit.edu](https://direct.mit.edu/books/oa-monograph-pdf/2313079/book_9780262257190.pdf), [archive.org](https://archive.org/details/extrapolationint0000norb))

2. **Richardson, W. H.** — *Bayesian-Based Iterative Method of Image Restoration*  
   ([Optica/OSA](https://opg.optica.org/josa/abstract.cfm?uri=josa-62-1-55), [DOI](https://doi.org/10.1364/JOSA.62.000055), [PDF](https://people.duke.edu/~sf59/Richardson1972.pdf))

3. **Lucy, L. B.** — *An Iterative Technique for the Rectification of Observed Distributions*  
   ([NASA ADS PDF](https://adsabs.harvard.edu/pdf/1974AJ.....79..745L), [DOI](https://doi.org/10.1086/111605))

4. **Starck, J.-L.; Pantin, E.; Murtagh, F.** — *Deconvolution in Astronomy: A Review*  
   ([jstarck.com](https://www.jstarck.com/files/Deconvolution-in-Astronomy-A-Review.pdf), [ui.adsabs.harvard.edu](https://ui.adsabs.harvard.edu/abs/2002PASP..114.1051S/abstract))

5. **Satish et al.** — *A Comprehensive Review of Blind Deconvolution Techniques*  
   ([iieta.org](https://www.iieta.org/journals/ts/paper/10.18280/ts.370321), [iieta.org](https://www.iieta.org/download/file/fid/36059))

### Постановка blind deconvolution + вывод/оценивание

6. **Ayers, G. R.; Dainty, J. C.** — *Iterative Blind Deconvolution Methods*  
   ([opg.optica.org](https://opg.optica.org/abstract.cfm?uri=ol-13-7-547), [DOI](https://doi.org/10.1364/OL.13.000547))

7. **Fish, D. A.; Brinicombe, A. M.; Pike, E. R.; Walker, J. G.** — *Blind Deconvolution by Means of the Richardson–Lucy Algorithm*  
   ([opg.optica.org](https://opg.optica.org/abstract.cfm?uri=josaa-12-1-58), [DOI](https://doi.org/10.1364/JOSAA.12.000058))

8. **Levin, A.; Weiss, Y.; Durand, F.; Freeman, W.** — *Understanding Blind Deconvolution Algorithms*  
   ([TPAMI DOI](https://doi.org/10.1109/TPAMI.2011.148), [CVPR’09 PDF (author version)](https://dspace.mit.edu/bitstream/handle/1721.1/59815/Levin-2009-Understanding%20and%20evaluating%20blind%20deconvolution%20algorithms.pdf?isAllowed=y&sequence=1), [CVPR’09 DOI](https://doi.org/10.1109/CVPR.2009.5206815))

9. **Wipf, D.; Zhang, H.** — *Revisiting Bayesian Blind Deconvolution*  
   ([microsoft.com](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wipf14a.pdf), [dl.acm.org](https://dl.acm.org/doi/abs/10.5555/2627435.2750360))

10. **Babacan, S. D.; Molina, R.; Katsaggelos, A. K. (2009)** — *Variational Bayesian Blind Deconvolution Using a Total Variation Prior*  
    ([DOI](https://doi.org/10.1109/TIP.2008.2007354), [PDF](https://www.dbabacan.info/papers/babacan_TIP09.pdf))

11. **Babacan, S. D.; Wang, J.; Molina, R.; Katsaggelos, A. K. (2010)** — *Bayesian Blind Deconvolution From Differently Exposed Image Pairs*  
    ([DOI](https://doi.org/10.1109/TIP.2010.2052263), [PDF](https://ccia.ugr.es/vip/files/journals/dualexposure.pdf), [Semantic Scholar](https://www.semanticscholar.org/paper/Bayesian-Blind-Deconvolution-From-Differently-Image-Babacan-Wang/d836694e77906536d2bd009cda35df0a2b361096))

12. **Amizic, B.; Molina, R.; Katsaggelos, A. K. (2012)** — *Sparse Bayesian blind image deconvolution with parameter estimation*  
    ([SpringerOpen article](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/1687-5281-2012-20), [DOI](https://doi.org/10.1186/1687-5281-2012-20), [UGR publications page](https://ccia.ugr.es/pi/computationalphotography/publications.html))

13. **Amizic, B.; Spinoulas, L.; Molina, R.; Katsaggelos, A. K. (2013)** — *Variational Bayesian Compressive Blind Image Deconvolution* (EUSIPCO; ID: 1569744671)  
    ([UGR publications page](https://ccia.ugr.es/pi/computationalphotography/publications.html), [EUSIPCO 2013 program](https://eurasip.org/Proceedings/Eusipco/Eusipco2013/program.html), [ResearchGate](https://www.researchgate.net/publication/280923411_Variational_Bayesian_compressive_blind_image_deconvolution))

14. **Likas, A. C.; Galatsanos, N. P. (2004)** — *A variational approach for Bayesian blind image deconvolution*  
    ([PDF](https://www.cs.uoi.gr/~arly/papers/TSP2004.pdf), [DOI](https://doi.org/10.1109/TSP.2004.831119))

15. **Molina, R.; Mateos, J.; Katsaggelos, A. K. (2006)** — *Blind Deconvolution Using a Variational Approach to Parameter, Image, and Blur Estimation*  
    ([DOI](https://doi.org/10.1109/TIP.2006.881972), [ResearchGate](https://www.researchgate.net/publication/6645275_Blind_Deconvolution_Using_a_Variational_Approach_to_Parameter_Image_and_Blur_Estimation))

16. **Tzikas, D. G.; Likas, A. C.; Galatsanos, N. P. (2009)** — *Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution With Student's-t Priors*  
    ([DOI](https://doi.org/10.1109/TIP.2008.2011757), [PubMed](https://pubmed.ncbi.nlm.nih.gov/19278919/), [Semantic Scholar](https://www.semanticscholar.org/paper/ac9c1ac6c8d14bf5618d1858a35b1e6fb5959965))

17. **Dempster, A. P.; Laird, N. M.; Rubin, D. B.** — *Maximum Likelihood from Incomplete Data via the EM Algorithm*  
    ([academic.oup.com](https://academic.oup.com/jrsssb/article/39/1/1/7027539), [doi.org](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x))

18. **Lagendijk, R. L.; Biemond, J.; Boekee, D. E. (1990)** — *Identification and restoration of noisy blurred images using the expectation-maximization algorithm*  
    ([TU Delft repository](https://repository.tudelft.nl/islandora/object/uuid:0836b719-2e2f-4371-b4bd-e89b9596f5a0), [DOI](https://doi.org/10.1109/29.57545))

19. **Katsaggelos, A. K.; Lay, K. T. (1991)** — *Maximum likelihood blur identification and image restoration using the EM algorithm*  
    ([DOI](https://doi.org/10.1109/78.80894), [Semantic Scholar](https://www.semanticscholar.org/paper/a0f388af0d2a1c8c0a7db3087c0f0a8f3bbd7f6f))

### Регуляризация и априоры

20. **Rudin, L. I.; Osher, S.; Fatemi, E.** — *Nonlinear Total Variation Based Noise Removal Algorithms*  
    ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/016727899290242F), [utk.edu](https://web.eecs.utk.edu/~hqi/ece692/references/noise-TV-PhysicaD92.pdf))

21. **Chan, T. F.; Wong, C.-K.** — *Total Variation Blind Deconvolution*  
    ([DOI](https://doi.org/10.1109/83.661187), [PubMed](https://pubmed.ncbi.nlm.nih.gov/18276257/))

22. **Chen et al.** — *Fractional-Order Total Variation for Image Restoration*  
    ([doi.org](https://doi.org/10.1155/2013/585310), [projecteuclid.org](https://projecteuclid.org/journals/abstract-and-applied-analysis/volume-2013/issue-SI25/Fractional-Order-Total-Variation-Image-Restoration-Based-on-Primal-Dual/10.1155/2013/585310.pdf))

23. **Xu, L.; Zheng, S.; Jia, J.** — *Unnatural L0 Sparse Representation for Natural Image Deblurring*  
    ([openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2013/html/Xu_Unnatural_L0_Sparse_2013_CVPR_paper.html), [openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2013/papers/Xu_Unnatural_L0_Sparse_2013_CVPR_paper.pdf))

24. **Dong, W.; Tao, S.; Xu, G.; Chen, Y. (2021)** — *Blind Deconvolution for Poissonian Blurred Image With Total Variation and L0-Norm Gradient Regularizations*  
    ([DOI](https://doi.org/10.1109/TIP.2020.3038518), [DBLP](https://dblp.org/rec/journals/tip/DongTXC21), [ResearchGate](https://www.researchgate.net/publication/347696605_Blind_Deconvolution_for_Poissonian_Blurred_Image_With_Total_Variation_and_L0-Norm_Gradient_Regularizations))

25. **Krishnan, D.; Fergus, R.** — *Fast Image Deconvolution Using Hyper-Laplacian Priors*  
    ([papers.nips.cc](https://papers.nips.cc/paper/3707-fast-image-deconvolution-using-hyper-laplacian-priors), [scispace.com](https://scispace.com/pdf/fast-image-deconvolution-using-hyper-laplacian-priors-4h5488ty79.pdf))

26. **Zeyde, S.; Elad, M.; Protter, M.** — *On Single Image Scale-Up Using Sparse Representations*  
    ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-642-27413-8_47), [technion.ac.il](https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/ImageScaleUp_LNCS.pdf))

### Оптимизация и численные методы

27. **Boyd, S.; Vandenberghe, L.** — *Convex Optimization*  
    ([stanford.edu](https://stanford.edu/~boyd/cvxbook/), [ucla.edu](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf))

28. **Candès, E.; Recht, B.** — *Exact Matrix Completion via Convex Optimization*  
    ([link.springer.com](https://link.springer.com/article/10.1007/s10208-009-9045-5), [link.springer.com](https://link.springer.com/content/pdf/10.1007/s10208-009-9045-5.pdf))

29. **Huber, P. J.** — *Robust Statistics*  
    ([onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/book/10.1002/0471725250))

30. **Huber, P. J. (1964)** — *Robust estimation of a location parameter*  
    ([Project Euclid](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full), [DOI](https://doi.org/10.1214/aoms/1177703732))

31. **Beck, A.; Teboulle, M.** — *A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)*  
    ([epubs.siam.org](https://epubs.siam.org/doi/10.1137/080716542), [tau.ac.il](https://www.tau.ac.il/~becka/FISTA.pdf))

32. **Ravi, Singh et al.** — *Robust Blind Deconvolution via Mirror Descent*  
    ([arxiv.org](https://arxiv.org/abs/1803.08137))

### Специальные модели/сигнальные предположения

33. **Ling, S.; Strohmer, T.** — *Self-Calibration and Biconvex Compressive Sensing*  
    ([arxiv.org](https://arxiv.org/abs/1507.03803))

34. **Joshi et al.** — *Image Deblurring Using Inertial Measurement Sensors*  
    ([doi.org](https://doi.org/10.1145/1778765.1778767), [szeliski.org](https://szeliski.org/papers/Joshi_ImageDeblurringIMUs_SG10.pdf))

35. **Kheradmand, A.; Milanfar, P.** — *A General Framework for Regularized, Similarity-Based Image Restoration*  
    ([doi.org](https://doi.org/10.1109/TIP.2014.2362059), [ucsc.edu](https://users.soe.ucsc.edu/~milanfar/publications/journal/TIP_Amin_Final.pdf))

36. **Ji, H.; Liu, C.; Shen, Z.** — *Blind Motion Deblurring Using Low-Rank Image Priors*  
    ([ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/5206514))

37. **Perrone, D.; Favaro, P.** — *A Clearer Picture of Blind Deconvolution*  
    ([arxiv.org](https://arxiv.org/abs/1412.0251))

### Deep learning / unrolling

38. **Agarwal et al.** — *Deep-URL: A Model-Aware Approach to Blind Deconvolution*  
    ([arxiv.org](https://arxiv.org/abs/2002.01053))
