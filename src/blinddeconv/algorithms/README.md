# Путеводитель по алгоритмам

## Blind Deconvolution (`blind_deconvolution/`)

Методы сгруппированы по типу источника кода - `our_company` и `third_party_company`

### Собственные реализации (`blind_deconvolution/our_company/`)

Имплементации на основе статей

Основные 
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`htp`](blind_deconvolution/our_company/bayesian/htp) | Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors | 2013 | [Источник](<../../../references/htp/(2013) Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors.pdf>) |
| [`fbdhsgp`](blind_deconvolution/our_company/bayesian/fbdhsgp) | Fast Bayesian blind deconvolution with Huber super Gaussian priors | 2017 | [Источники](../../../references/fbdhsgp) |
| [`bid_hbsp`](blind_deconvolution/our_company/bayesian/bid_hbsp) | Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior | 2024 | [Источник](<../../../references/bid_hbsp/(2024) Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior.pdf>) |

Экспериментальные подходы (`blind_deconvolution/our_company/experimaental_approaches`)
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`ard`](blind_deconvolution/our_company/bayesian/bid_hbsp) | Blind Deconvolution with Model Discrepancies | 2017 | [Источник](<../../../references/ard/(2017) Blind Deconvolution with Model Discrepancies.pdf>) |

### Внешние обёртки и источники (`blind_deconvolution/third_party_company/`)

Модуль для алгоритмов, заимствованных из внешних репозиториев и адаптированных под фреймворк.

## Non-Blind Deconvolution (`nonblind_deconvolution/`)

## PSF Estimation (`kernel_estimation/`)

Папка зарезервирована под алгоритмы оценки ядра (PSF/kernel estimation); сейчас в репозитории пустая.

### `unsorted/`

- [`_23ms410_Blind_Deconvolution`](unsorted/_23ms410_Blind_Deconvolution) — [repo](https://github.com/23ms410/Blind-Deconvolution), blind deconvolution (базовая постановка/демо) [6][8]
- [`ADY_YDA_Iterative_Blind_Image_Deconvolution`](unsorted/ADY_YDA_Iterative_Blind_Image_Deconvolution) — [repo](https://github.com/ADY-YDA/Iterative-Blind-Image-Deconvolution/blob/main/Expectation-Maximization.ipynb), Expectation–Maximization (EM) для blind deconvolution [17]
- [`gpl27_deblur`](unsorted/gpl27_deblur) — [repo](https://github.com/gpl27/deblur), High-Quality Motion Deblurring (SIGGRAPH 2008), non-blind deconvolution (локальный порт/обёртка) [1]
- [`CACTuS_AI_Blind_Deconvolution_using_Modulated_Inputs`](unsorted/CACTuS_AI_Blind_Deconvolution_using_Modulated_Inputs) — [repo](https://github.com/CACTuS-AI/Blind-Deconvolution-using-Modulated-Inputs), blind deconvolution with modulated inputs (bilinear inverse problem / lifting) [28]
- [`CEA_jiangming_DecGMCA`](unsorted/CEA_jiangming_DecGMCA) — [repo](https://github.com/CEA-jiangming/DecGMCA), deconvolution + GMCA (sparsity-based demixing / morphological component analysis) [26]
- [`COR_OPT_RobustBlindDeconv`](unsorted/COR_OPT_RobustBlindDeconv) — [repo](https://github.com/COR-OPT/RobustBlindDeconv), robust blind deconvolution: subgradient method и prox-linear method [29][31]
- [`Drorharush_SBD`](unsorted/Drorharush_SBD) — [repo](https://github.com/Drorharush/SBD), sparse blind deconvolution (sparsity-driven kernel/latent estimation) [26]
- [`GeekLogan_pyBlindRL`](unsorted/GeekLogan_pyBlindRL) — [repo](https://github.com/GeekLogan/pyBlindRL), blind Richardson–Lucy deconvolution (Poisson likelihood) [2][3][7]
- [`Tmodrzyk_richardson_lucy_python`](unsorted/Tmodrzyk_richardson_lucy_python) — [repo](https://github.com/Tmodrzyk/richardson-lucy-python), Richardson–Lucy deconvolution (non-blind) [2][3]
- [`adamalavi_Blind_motion_deblurring_for_license_plates`](unsorted/adamalavi_Blind_motion_deblurring_for_license_plates) — [repo](https://github.com/ankitVP77/Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning), CNN-оценка motion PSF + Wiener deconvolution [1][5]
- [`axium_Blind_Image_Deconvolution_using_Deep_Generative_Priors`](unsorted/axium_Blind_Image_Deconvolution_using_Deep_Generative_Priors) — [repo](https://github.com/axium/Blind-Image-Deconvolution-using-Deep-Generative-Priors), blind deconvolution с deep generative priors (оптимизация в латентных пространствах генераторов) [5][8]
- [`deu439_sbl_blind_deconvolution`](unsorted/deu439_sbl_blind_deconvolution) — [repo](https://github.com/deu439/sbl-blind-deconvolution), sparse blind deconvolution (sparsity prior) [26]
- [`fabioviggiano_BlindDeconvolution`](unsorted/fabioviggiano_BlindDeconvolution) — [repo](https://github.com/fabioviggiano/BlindDeconvolution), blind deconvolution (практическая реализация/демо) [6][8]
- [`gandor26_blind_deconvolution_through_a_single_image`](unsorted/gandor26_blind_deconvolution_through_a_single_image) — [repo](https://github.com/Gandor26/Blind-Deconvolution-through-a-Single-Image), blind deconvolution из одного изображения (single-image blind deblurring) [6][8]
- [`idiap_semiblindpsfdeconv`](unsorted/idiap_semiblindpsfdeconv) — [repo](https://github.com/idiap/semiblindpsfdeconv), semi-blind PSF deconvolution (частично параметризованный PSF) [6][8]
- [`jeffreysblake_funsearch_blind_deconvolution`](unsorted/jeffreysblake_funsearch_blind_deconvolution) — [repo](https://github.com/jeffreysblake/funsearch-blind-deconvolution), blind deconvolution (поиск/генерация процедур, экспериментально) [5]
- [`jhell96_Deep_Blind_Deblur`](unsorted/jhell96_Deep_Blind_Deblur) — [repo](https://github.com/jhell96/Deep-Blind-Deblur), deep learning blind deblurring (нейросетевая модель) [5][38]
- [`luczeng_MotionBlur`](unsorted/luczeng_MotionBlur) — [repo](https://github.com/luczeng/MotionBlur), моделирование/инверсия motion blur (forward model + deblurring) [6]
- [`mujib2020_Non_blind_and_Blind_Deconvolution_under_Poisson_noise`](unsorted/mujib2020_Non_blind_and_Blind_Deconvolution_under_Poisson_noise) — [repo](https://github.com/mujib2020/Non-blind-and-Blind-Deconvolution-under-Poisson-noise), EM и fractional-order TV (FOTV) при Poisson noise [17][22]
- [`JohnRagland_Total_Variation_MATLAB_implementation`](unsorted/JohnRagland_Total_Variation_MATLAB_implementation) — TV deconvolution (MATLAB implementation/демо)
- [`sanghviyashiitb_photon_limited_blind`](unsorted/sanghviyashiitb_photon_limited_blind) — [repo](https://github.com/sanghviyashiitb/photon-limited-blind), photon-limited blind deconvolution: unsupervised iterative kernel estimation (P4IP, plug-and-play/iterative scheme) [5][8]
- [`tianyishan_Blind_Deconvolution`](unsorted/tianyishan_Blind_Deconvolution) — [repo](https://github.com/tianyishan/Blind_Deconvolution), PRIDA (Provably Robust Image Deconvolution Algorithm), mirror descent [32]
- [`ys_koshelev_nla_deblur`](unsorted/ys_koshelev_nla_deblur) — [repo](https://github.com/ys-koshelev/nla_deblur), text deblurring: kernel estimation + L0/TV регуляризация (проект Yang) [20][23]

## Литература

### Классика и обзоры

1. **Wiener, N.** — *Extrapolation, Interpolation, and Smoothing of Stationary Time Series* (классическая теория оптимальной линейной фильтрации, Wiener filter)
	([direct.mit.edu](https://direct.mit.edu/books/oa-monograph-pdf/2313079/book_9780262257190.pdf>), [archive.org](https://archive.org/details/extrapolationint0000norb), [читать](<../../../references/Extrapolation, Interpolation, and Smoothing of Stationary Time Series.pdf>))

2. **Richardson, W. H.** — *Bayesian-Based Iterative Method of Image Restoration*  
   ([Optica/OSA](https://opg.optica.org/josa/abstract.cfm?uri=josa-62-1-55), [DOI](https://doi.org/10.1364/JOSA.62.000055), [PDF](https://people.duke.edu/~sf59/Richardson1972.pdf>))

3. **Lucy, L. B.** — *An Iterative Technique for the Rectification of Observed Distributions*  
   ([NASA ADS PDF](https://adsabs.harvard.edu/pdf/1974AJ.....79..745L), [DOI](https://doi.org/10.1086/111605))

4. **Starck, J.-L.; Pantin, E.; Murtagh, F.** — *Deconvolution in Astronomy: A Review*  
   ([jstarck.com](https://www.jstarck.com/files/Deconvolution-in-Astronomy-A-Review.pdf>), [ui.adsabs.harvard.edu](https://ui.adsabs.harvard.edu/abs/2002PASP..114.1051S/abstract))

5. **Satish et al.** — *A Comprehensive Review of Blind Deconvolution Techniques*  
   ([iieta.org](https://www.iieta.org/journals/ts/paper/10.18280/ts.370321), [iieta.org](https://www.iieta.org/download/file/fid/36059), [читать](<../../../references/A Comprehensive Review of Blind Deconvolution Techniques.pdf>))

### Постановка слепой деконволюции + вывод/оценивание

6. **Ayers, G. R.; Dainty, J. C.** — *Iterative Blind Deconvolution Methods*  
   ([opg.optica.org](https://opg.optica.org/abstract.cfm?uri=ol-13-7-547), [DOI](https://doi.org/10.1364/OL.13.000547))

7. **Fish, D. A.; Brinicombe, A. M.; Pike, E. R.; Walker, J. G.** — *Blind Deconvolution by Means of the Richardson–Lucy Algorithm*  
   ([opg.optica.org](https://opg.optica.org/abstract.cfm?uri=josaa-12-1-58), [DOI](https://doi.org/10.1364/JOSAA.12.000058))

8. **Levin, A.; Weiss, Y.; Durand, F.; Freeman, W.** — *Understanding Blind Deconvolution Algorithms*  
   ([TPAMI DOI](https://doi.org/10.1109/TPAMI.2011.148), [CVPR’09 PDF (author version)](https://dspace.mit.edu/bitstream/handle/1721.1/59815/Levin-2009-Understanding%20and%20evaluating%20blind%20deconvolution%20algorithms.pdf?isAllowed=y&sequence=1), [CVPR’09 DOI](https://doi.org/10.1109/CVPR.2009.5206815))

9. **Wipf, D.; Zhang, H.** — *Revisiting Bayesian Blind Deconvolution*  
   ([microsoft.com](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wipf14a.pdf>), [dl.acm.org](https://dl.acm.org/doi/abs/10.5555/2627435.2750360), [читать](<../../../references/Revisiting Bayesian Blind Deconvolution.pdf>))

10. **Babacan, S. D.; Molina, R.; Katsaggelos, A. K. (2009)** — *Variational Bayesian Blind Deconvolution Using a Total Variation Prior*  
    ([DOI](https://doi.org/10.1109/TIP.2008.2007354), [PDF](https://www.dbabacan.info/papers/babacan_TIP09.pdf>), [читать](<../../../references/Variational Bayesian Blind Deconvolution Using a Total Variation Prior.pdf>))

11. **Babacan, S. D.; Wang, J.; Molina, R.; Katsaggelos, A. K. (2010)** — *Bayesian Blind Deconvolution From Differently Exposed Image Pairs*  
    ([DOI](https://doi.org/10.1109/TIP.2010.2052263), [PDF](https://ccia.ugr.es/vip/files/journals/dualexposure.pdf>), [Semantic Scholar](https://www.semanticscholar.org/paper/Bayesian-Blind-Deconvolution-From-Differently-Image-Babacan-Wang/d836694e77906536d2bd009cda35df0a2b361096), [читать](<../../../references/Bayesian Blind Deconvolution From Differently Exposed Image Pairs.pdf>))

12. **Amizic, B.; Molina, R.; Katsaggelos, A. K. (2012)** — *Sparse Bayesian blind image deconvolution with parameter estimation*  
    ([SpringerOpen article](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/1687-5281-2012-20), [DOI](https://doi.org/10.1186/1687-5281-2012-20), [UGR publications page](https://ccia.ugr.es/pi/computationalphotography/publications.html), [читать](<../../../references/Sparse Bayesian blind image deconvolution with parameter estimation.pdf>))

13. **Amizic, B.; Spinoulas, L.; Molina, R.; Katsaggelos, A. K. (2013)** — *Variational Bayesian Compressive Blind Image Deconvolution* (EUSIPCO; ID: 1569744671)  
    ([UGR publications page](https://ccia.ugr.es/pi/computationalphotography/publications.html), [EUSIPCO 2013 program](https://eurasip.org/Proceedings/Eusipco/Eusipco2013/program.html), [ResearchGate](https://www.researchgate.net/publication/280923411_Variational_Bayesian_compressive_blind_image_deconvolution), [читать](<../../../references/Variational Bayesian Compressive Blind Image Deconvolution.pdf>))

14. **Likas, A. C.; Galatsanos, N. P. (2004)** — *A variational approach for Bayesian blind image deconvolution*  
    ([PDF](https://www.cs.uoi.gr/~arly/papers/TSP2004.pdf>), [DOI](https://doi.org/10.1109/TSP.2004.831119), [читать](<../../../references/A variational approach for Bayesian blind image deconvolution.pdf>))

15. **Molina, R.; Mateos, J.; Katsaggelos, A. K. (2006)** — *Blind Deconvolution Using a Variational Approach to Parameter, Image, and Blur Estimation*  
    ([DOI](https://doi.org/10.1109/TIP.2006.881972), [ResearchGate](https://www.researchgate.net/publication/6645275_Blind_Deconvolution_Using_a_Variational_Approach_to_Parameter_Image_and_Blur_Estimation), [читать](<../../../references/Blind Deconvolution Using a Variational Approach to Parameter, Image, and Blur Estimation.pdf>))

16. **Tzikas, D. G.; Likas, A. C.; Galatsanos, N. P. (2009)** — *Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution With Student's-t Priors*  
    ([DOI](https://doi.org/10.1109/TIP.2008.2011757), [PubMed](https://pubmed.ncbi.nlm.nih.gov/19278919/), [Semantic Scholar](https://www.semanticscholar.org/paper/ac9c1ac6c8d14bf5618d1858a35b1e6fb5959965), [читать](<../../../references/Blind Deconvolution Using a Variational Approach to Parameter, Image, and Blur Estimation.pdf>))

17. **Dempster, A. P.; Laird, N. M.; Rubin, D. B.** — *Maximum Likelihood from Incomplete Data via the EM Algorithm*  
    ([academic.oup.com](https://academic.oup.com/jrsssb/article/39/1/1/7027539), [doi.org](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x), [читать](<../../../references/Maximum Likelihood from Incomplete Data via the EM Algorithm.pdf>))

18. **Lagendijk, R. L.; Biemond, J.; Boekee, D. E. (1990)** — *Identification and restoration of noisy blurred images using the expectation-maximization algorithm*  
    ([TU Delft repository](https://repository.tudelft.nl/islandora/object/uuid:0836b719-2e2f-4371-b4bd-e89b9596f5a0), [DOI](https://doi.org/10.1109/29.57545))

19. **Katsaggelos, A. K.; Lay, K. T. (1991)** — *Maximum likelihood blur identification and image restoration using the EM algorithm*  
    ([DOI](https://doi.org/10.1109/78.80894), [Semantic Scholar](https://www.semanticscholar.org/paper/a0f388af0d2a1c8c0a7db3087c0f0a8f3bbd7f6f))

### Регуляризация и априоры

20. **Rudin, L. I.; Osher, S.; Fatemi, E.** — *Nonlinear Total Variation Based Noise Removal Algorithms*  
    ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/016727899290242F), [utk.edu](https://web.eecs.utk.edu/~hqi/ece692/references/noise-TV-PhysicaD92.pdf>), [читать](<../../../references/Nonlinear total variation based noise removal algorithms.pdf>))

21. **Chan, T. F.; Wong, C.-K.** — *Total Variation Blind Deconvolution*  
    ([DOI](https://doi.org/10.1109/83.661187), [PubMed](https://pubmed.ncbi.nlm.nih.gov/18276257/))

22. **Chen et al.** — *Fractional-Order Total Variation for Image Restoration*  
    ([doi.org](https://doi.org/10.1155/2013/585310), [projecteuclid.org](https://projecteuclid.org/journals/abstract-and-applied-analysis/volume-2013/issue-SI25/Fractional-Order-Total-Variation-Image-Restoration-Based-on-Primal-Dual/10.1155/2013/585310.pdf>), [читать](<../../../references/Fractional-Order Total Variation for Image Restoration.pdf>))

23. **Xu, L.; Zheng, S.; Jia, J.** — *Unnatural L0 Sparse Representation for Natural Image Deblurring*  
    ([openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2013/html/Xu_Unnatural_L0_Sparse_2013_CVPR_paper.html), [openaccess.thecvf.com](https://openaccess.thecvf.com/content_cvpr_2013/papers/Xu_Unnatural_L0_Sparse_2013_CVPR_paper.pdf>), [читать](<../../../references/Unnatural L0 Sparse Representation for Natural Image Deblurring.pdf>))

24. **Dong, W.; Tao, S.; Xu, G.; Chen, Y. (2021)** — *Blind Deconvolution for Poissonian Blurred Image With Total Variation and L0-Norm Gradient Regularizations*  
    ([DOI](https://doi.org/10.1109/TIP.2020.3038518), [DBLP](https://dblp.org/rec/journals/tip/DongTXC21), [ResearchGate](https://www.researchgate.net/publication/347696605_Blind_Deconvolution_for_Poissonian_Blurred_Image_With_Total_Variation_and_L0-Norm_Gradient_Regularizations))

25. **Krishnan, D.; Fergus, R.** — *Fast Image Deconvolution Using Hyper-Laplacian Priors*  
    ([papers.nips.cc](https://papers.nips.cc/paper/3707-fast-image-deconvolution-using-hyper-laplacian-priors), [scispace.com](https://scispace.com/pdf/fast-image-deconvolution-using-hyper-laplacian-priors-4h5488ty79.pdf>), [читать](<../../../references/Fast Image Deconvolution Using Hyper-Laplacian Priors.pdf>))

26. **Zeyde, S.; Elad, M.; Protter, M.** — *On Single Image Scale-Up Using Sparse Representations*  
    ([link.springer.com](https://link.springer.com/chapter/10.1007/978-3-642-27413-8_47), [technion.ac.il](https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/ImageScaleUp_LNCS.pdf>), [читать](<../../../references/On Single Image Scale-Up Using Sparse Representations.pdf>))

### Оптимизация и численные методы

27. **Boyd, S.; Vandenberghe, L.** — *Convex Optimization*  
    ([stanford.edu](https://stanford.edu/~boyd/cvxbook/), [ucla.edu](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf>), [читать](<../../../references/Convex Optimization.pdf>))

28. **Candès, E.; Recht, B.** — *Exact Matrix Completion via Convex Optimization*  
    ([link.springer.com](https://link.springer.com/article/10.1007/s10208-009-9045-5), [link.springer.com](https://link.springer.com/content/pdf/10.1007/s10208-009-9045-5.pdf>), [читать](<../../../references/Exact Matrix Completion via Convex Optimization.pdf>))

29. **Huber, P. J.** — *Robust Statistics*  
    ([onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/book/10.1002/0471725250))

30. **Huber, P. J. (1964)** — *Robust estimation of a location parameter*  
    ([Project Euclid](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full), [DOI](https://doi.org/10.1214/aoms/1177703732), [читать](<../../../references/Robust estimation of a location parameter.pdf>))

31. **Beck, A.; Teboulle, M.** — *A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)*  
    ([epubs.siam.org](https://epubs.siam.org/doi/10.1137/080716542), [tau.ac.il](https://www.tau.ac.il/~becka/FISTA.pdf>), [читать](<../../../references/A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).pdf>))

32. **Ravi, Singh et al.** — *Robust Blind Deconvolution via Mirror Descent*  
    ([arxiv.org](https://arxiv.org/abs/1803.08137), [читать](<../../../references/Robust Blind Deconvolution via Mirror Descent.pdf>))

### Специальные модели/сигнальные предположения

33. **Ling, S.; Strohmer, T.** — *Self-Calibration and Biconvex Compressive Sensing*  
    ([arxiv.org](https://arxiv.org/abs/1507.03803), [читать](<../../../references/Self-Calibration and Biconvex Compressive Sensing.pdf>))

34. **Joshi et al.** — *Image Deblurring Using Inertial Measurement Sensors*  
    ([doi.org](https://doi.org/10.1145/1778765.1778767), [szeliski.org](https://szeliski.org/papers/Joshi_ImageDeblurringIMUs_SG10.pdf>), [читать](<../../../references/Image Deblurring Using Inertial Measurement Sensors.pdf>))

35. **Kheradmand, A.; Milanfar, P.** — *A General Framework for Regularized, Similarity-Based Image Restoration*  
    ([doi.org](https://doi.org/10.1109/TIP.2014.2362059), [ucsc.edu](https://users.soe.ucsc.edu/~milanfar/publications/journal/TIP_Amin_Final.pdf>), [читать](<../../../references/A General Framework for Regularized, Similarity-Based Image Restoration.pdf>))

36. **Ji, H.; Liu, C.; Shen, Z.** — *Blind image deblurring using class-adapted image priors*  
    ([ieeexplore.ieee.org](https://www.researchgate.net/publication/319524652_Blind_image_deblurring_using_class-adapted_image_priors), [читать](<../../../references/Blind image deblurring using class-adapted image priors.pdf>))

37. **Perrone, D.; Favaro, P.** — *A Clearer Picture of Blind Deconvolution*  
    ([arxiv.org](https://arxiv.org/abs/1412.0251), [читать](<../../../references/A Clearer Picture of Blind Deconvolution.pdf>))

### Глубокое обучение

38. **Agarwal et al.** — *Deep-URL: A Model-Aware Approach to Blind Deconvolution*  
    ([arxiv.org](https://arxiv.org/abs/2002.01053), [читать](<../../../references/Deep-URL A Model-Aware Approach to Blind Deconvolution.pdf>))
