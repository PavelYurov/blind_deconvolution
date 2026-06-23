# Путеводитель по алгоритмам

## Blind Deconvolution (`blind_deconvolution/`)

Методы сгруппированы по типу источника кода - `our_company` и `third_party_company`

### Собственные реализации (`blind_deconvolution/our_company/`)

Имплементации на основе оригинальных научных публикаций, оптимизированные под фреймворк.

#### Основные алгоритмы (Core)
Методы, показавшие наиболее стабильные и качественные результаты восстановления.
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`htp`](blind_deconvolution/our_company/bayesian/htp) | Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors | 2013 | [Источник](<../../../references/htp/(2013) Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors.pdf>) |
| [`fbdhsgp`](blind_deconvolution/our_company/bayesian/fbdhsgp) | Fast Bayesian blind deconvolution with Huber super Gaussian priors | 2017 | [Источники](../../../references/fbdhsgp) |
| [`bid_hbsp`](blind_deconvolution/our_company/bayesian/bid_hbsp) | Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior | 2024 | [Источник](<../../../references/bid_hbsp/(2024) Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior.pdf>) |

#### Дополнительные подходы (`blind_deconvolution/our_company/experimental_approaches`)
Реализации альтернативных подходов и исследовательские наработки.
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`ard`](blind_deconvolution/our_company/experimental_approaches/bayesian/ard) | Blind Deconvolution with Model Discrepancies | 2017 | [Источник](<../../../references/ard/(2017) Blind Deconvolution with Model Discrepancies.pdf>) |

### Внешние обёртки и источники (`blind_deconvolution/third_party_company/`)

Модуль для алгоритмов, заимствованных из внешних репозиториев и адаптированных под фреймворк.

## Non-Blind Deconvolution (`nonblind_deconvolution/`)

## PSF Estimation (`kernel_estimation/`)

Папка зарезервирована под алгоритмы оценки ядра (PSF/kernel estimation); сейчас в репозитории пустая.

## Denoising (`denoise/`, `mod_denoise/`)

## Super Resolution (`kernel_estimation/`)


