# Путеводитель по алгоритмам

## Blind Deconvolution (`blind_deconvolution/`)

Методы сгруппированы по типу источника кода - `our_company` и `third_party_company`

### Собственные реализации (`blind_deconvolution/our_company/`)

Имплементации на основе оригинальных научных публикаций, оптимизированные под фреймворк.

#### Основные алгоритмы (Core)
Методы, показавшие наиболее стабильные результаты и высокую устойчивость к различным типам размытия.
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`htp`](blind_deconvolution/our_company/bayesian/htp) | Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors | 2013 | [Источник](<../../../references/htp/(2013) Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors.pdf>) |
| [`lip`](blind_deconvolution/our_company/logarithmic_pds) | Blind Deconvolution via Lower-Bounded Logarithmic Image Priors | 2015 | [Источники](<../../../references/lip>) |
| [`dcp`](blind_deconvolution/our_company/dark_channel_prior) | Blind Image Deblurring Using Dark Channel Prior | 2016 | [Источники](<../../../references/dcp>) |
| [`ecp`](blind_deconvolution/our_company/ecp) | Image Deblurring via Extreme Channels Prior | 2017 | [Источник](<../../../references/ecp/(2017) Image Deblurring via Extreme Channels Prior.pdf>) |
| [`gbbid`](blind_deconvolution/our_company/graph) | Blind Image Deblurring Via Reweighted Graph Total Variation | 2017 | [Источники](<../../../references/gbbid>) |
| [`fbdhsgp`](blind_deconvolution/our_company/bayesian/fbdhsgp) | Fast Bayesian blind deconvolution with Huber super Gaussian priors | 2017 | [Источники](../../../references/fbdhsgp) |
| [`lmgp`](blind_deconvolution/our_company/lmgp) | Blind Image Deblurring with Local Maximum Gradient Prior | 2019 | [Источник](<../../../references/lmgp/(2019) Blind Image Deblurring with Local Maximum Gradient Prior.pdf>) |
| [`pmp`](blind_deconvolution/our_company/patch_wise_minimum_pixels_prior) | Blind Deblurring via Patch-wise Minimal Pixels Prior | 2020 | [Источник](<../../../references/pmp/(2020) A Simple Local Minimal Intensity Prior and An Improved Algorithm for Blind Image Deblurring.pdf>) |
| [`esm`](blind_deconvolution/our_company/esm) | Enhanced Sparse Model for Blind Deblurring | 2020 | [Источник](<../../../references/esm/(2020) Enhanced Sparse Model for Blind Deblurring.pdf>) |
| [`bid_hbsp`](blind_deconvolution/our_company/bayesian/bid_hbsp) | Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior | 2024 | [Источник](<../../../references/bid_hbsp/(2024) Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior.pdf>) |

#### Дополнительные и экспериментальные подходы (`blind_deconvolution/our_company/experimental_approaches`)
Реализации альтернативных подходов и исследовательские наработки.
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`сgmrf`](blind_deconvolution/our_company/experimental_approaches/bayesian/cgmrf) | Bayesian image restoration using compound Gauss-Markov random fields | 2003 | [Источники](<../../../references/cgmrf>) |
| [`rcs`](blind_deconvolution/our_company/experimental_approaches/bayesian/rcs) | Removing Camera Shake from a Single Photograph | 2006 | [Источник](<../../../references/rcs/(2006) Removing Camera Shake from a Single Photograph.pdf>) |
| [`vbskb_bid_sp`](blind_deconvolution/our_company/experimental_approaches/bayesian/vbskb_bid_sp) | Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution With Student's-t Priors | 2009 | [Источники](<../../../references/vbskb_bid_sp>) |
| [`eml`](blind_deconvolution/our_company/experimental_approaches/bayesian/eml) | Efficient Marginal Likelihood Optimization in Blind Deconvolution | 2011 | [Источники](<../../../references/eml>) |
| [`nsm`](blind_deconvolution/our_company/experimental_approaches/nsm) | Blind Deconvolution Using a Normalized Sparsity Measure | 2011 | [Источник](<../../../references/nsm/(2011) Blind Deconvolution Using a Normalized Sparsity Measure.pdf>) |
| [`bdgsp`](blind_deconvolution/our_company/experimental_approaches/bayesian/bdgsp) | Bayesian Blind Deconvolution with General Sparse Image Priors | 2012 | [Источник](<../../../references/bdgsp/(2012) Bayesian Blind Deconvolution with General Sparse Image Priors.pdf>) |
| [`mrf`](blind_deconvolution/our_company/experimental_approaches/mrf) | MRF-based Blind Image Deconvolution | 2012 | [Источник](<../../../references/mrf/(2012) MRF-based Blind Image Deconvolution.pdf>) |
| [`amape_htp`](blind_deconvolution/our_company/experimental_approaches/amape_htp) | Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors (Alt Version) | 2013 | [Источник](<../../../references/amape_htp/(2013) Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors.pdf>) |
| [`pam`](blind_deconvolution/our_company/experimental_approaches/pam) | Total Variation-Projected Alternating Minimization | 2014 | [Источники](<../../../references/pam>) |
| [`vdbke`](blind_deconvolution/our_company/experimental_approaches/bayesian/vdbke) |  Variational Dirichlet Blur Kernel Estimation | 2015 | [Источник](<../../../references/vdbke/(2015) Variational Dirichlet Blur Kernel Estimation.pdf>) |
| [`ard`](blind_deconvolution/our_company/experimental_approaches/bayesian/ard) | Blind Deconvolution with Model Discrepancies | 2017 | [Источник](<../../../references/ard/(2017) Blind Deconvolution with Model Discrepancies.pdf>) |
| [`prida`](blind_deconvolution/our_company/experimental_approaches/prida) | Robust Blind Deconvolution via Mirror Descent | 2018 | [Источник](<../../../references/prida/(2018) Robust Blind Deconvolution via Mirror Descent.pdf>) |
| [`lowrank`](blind_deconvolution/our_company/experimental_approaches/lowrank) | Blind Deconvolution Using Low-Rank Prior | 2019 | [Источники](<../../../references/lowrank>) |
| [`oid`](blind_deconvolution/our_company/experimental_approaches/oid) | Outlier Identifying and Discarding in Blind Image Deblurring | 2020 | [Источник](<../../../references/oid/(2020) Outlier Identifying and Discarding in Blind Image Deblurring.pdf>) |
| [`nscp`](blind_deconvolution/our_company/experimental_approaches/nscp) | Blind Image Deblurring via a Novel Sparse Channel Prior | 2022 | [Источник](<../../../references/nscp/(2022) Blind Image Deblurring via a Novel Sparse Channel Prior.pdf>) |
| [`hsp`](blind_deconvolution/our_company/experimental_approaches/bayesian/hsp) | Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior (Alt Version) | 2024 | [Источники](<../../../references/hsp>) |
| [`aeer`](blind_deconvolution/our_company/experimental_approaches/adaptive_euler_elastica/aeer) | Blind image deconvolution based on adaptive Euler's elastica regularization | 2024 | [Источник](<../../../references/aeer/(2024) Blind image deconvolution based on adaptive Euler's elastica regularization.pdf>) |
| [`aeer_poisson`](blind_deconvolution/our_company/experimental_approaches/adaptive_euler_elastica/aeer_poisson) | Blind Restoration of Poisson Images Using Adaptive Euler's Elastica Regularization | 2024 | [Источник](<../../../references/aeer_poisson/(2024) Blind Restoration of Poisson Images Using Adaptive Euler's Elastica Regularization.pdf>) |
| [`fractional_order`](blind_deconvolution/our_company/experimental_approaches/fractional_order) | Blind Image Deconvolution - When Patch-wise Minimal Pixels Prior Meets Fractional-Order Method | 2025 | [Источник](<../../../references/fractional_order/(2025) Blind Image Deconvolution - When Patch-wise Minimal Pixels Prior Meets Fractional-Order Method.pdf>) |
| [`mhdm`](blind_deconvolution/our_company/experimental_approaches/mhdm) | Applications of multiscale hierarchical decomposition to blind deconvolution | 2025 | [Источники](<../../../references/mhdm>) |

### Внешние обёртки и источники (`blind_deconvolution/third_party_company/`)

Модуль для интеграции сторонних алгоритмов из внешних репозиториев.

## Non-Blind Deconvolution (`nonblind_deconvolution/`)

Зарезервированный модуль для методов восстановления при известном ядре размытия. Наличие данного раздела позволяет использовать фреймворк в двухстадийных схемах (сначала оценка PSF, затем Non-Blind восстановление). 

## PSF Estimation (`kernel_estimation/`)

Специализированный раздел для алгоритмов изолированной оценки ядра размытия. В текущей версии фреймворка функции оценки PSF интегрированы непосредственно в основные алгоритмы Blind Deconvolution.

## Модификации (`mod_denoise/`)

Методы для оценки уровня шума и его подавления. Важны для повышения устойчивости на сильно зашумленных изображениях. Данные алгоритмы используются как для предварительной обработки изображений, так и в качестве внутренних регуляризаторов в итерационных схемах деконволюции (Plug-and-Play). 

| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`vst`](mod_denoise/vst.py) | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | 2013 | [Источник](<../../../references/vst/(2013) Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise.pdf>) |
| [`pyatykh`](mod_denoise/pyatykh_noise_reconstruction.py) | Image Noise Level Estimation by Principal Component Analysis | 2014 | [Источник](<../../../references/pyatykh/(2014) Image Noise Level Estimation by Principal Component Analysis.pdf>) |
| [`chen`](mod_denoise/chen_noise_estimate.py) | An Efficient Statistical Method for Image Noise Level Estimation | 2015 | [Источник](<../../../references/chen/(2015) An Efficient Statistical Method for Image Noise Level Estimation.pdf>) |
| [`act`](mod_denoise/act_denoise.py) | Compressive Sensing Image Restoration Using Adaptive Curvelet Thresholding and Nonlocal Sparse Regularization | 2016 | [Источник](<../../../references/act/(2016) Compressive Sensing Image Restoration Using Adaptive Curvelet Thresholding and Nonlocal Sparse Regularization.pdf>) |
| [`screenot`](mod_denoise/screenot.py) | ScreeNOT Exact MSE-optimal singular value thresholding in correlated noise | 2023 | [Источник](<../../../references/screenot/(2023) ScreeNOT Exact MSE-optimal singular value thresholding in correlated noise.pdf>) |

## Super Resolution (`super_resolution/`)

Модуль для задач повышения пространственного разрешения. Алгоритмы этого раздела дополняют функционал деконволюции, позволяя решать комплексные задачи улучшения качества. Применение Super Resolution перед этапом оценки ядра размытия в Blind Deconvolution особенно полезно на искаженных изображениях с низкой детализацией.

| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`bcsnsp_sr`](super_resolution/our_company/bcsnsp_sr) | Bayesian combination of sparse and non-sparse priors in image super resolution | 2013 | [Источники](<../../../references/bcsnsp_sr>) |
| [`pansharpening`](super_resolution/our_company/selfexsr) | Single Image Super-resolution from Transformed Self-Exemplars | 2015 | [Источник](<../../../references/selfexsr/(2015) Single Image Super-resolution from Transformed Self-Exemplars.pdf>) |
| [`pansharpening`](super_resolution/our_company/pansharpening) | Variational Bayesian Pansharpening with Super-Gaussian Sparse Image Priors | 2020 | [Источники](<../../../references/pansharpening/(2020) Variational Bayesian Pansharpening with Super-Gaussian Sparse Image Priors.pdf>) |



