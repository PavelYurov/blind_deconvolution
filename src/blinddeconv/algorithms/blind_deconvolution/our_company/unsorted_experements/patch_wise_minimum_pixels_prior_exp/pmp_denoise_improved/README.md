# pmp_denoise_merge

Диспетчер, объединяющий два пайплайна **без** изменения их кода:

* `pmp_denoise.PMP_BD(..., auto_mode='robust', ...)` — старый,
  с σ-aware внутренними механизмами (denoise_eps, grad_smooth_sigma, λ_*,
  guided blind-denoise, ensemble_denoise). Лучше на **colored / clean / лёгком
  AWGN**.
* `pmp_denoise_fix.PMP_BD_Robust` — новый, с тяжёлым предварительным денойзом
  (BM3D / VST+BM3D / ACT). Лучше на **сильном белом Gaussian / Poisson / impulse**.

## Использование

```python
from src.blinddeconv.algorithms.blind_deconvolution.our_company.\
patch_wise_minimum_pixels_prior.pmp_denoise_merge import PMP_BD_Merged

alg = PMP_BD_Merged(kernel_size=51)
restored, kernel = alg.process(noisy)
print(alg.last_branch, alg.last_descriptor)   # 'robust' | 'legacy'
```

Принудительно зафиксировать ветвь:
```python
alg = PMP_BD_Merged(kernel_size=51, force='robust')   # или 'legacy'
```

Прокинуть свои kwargs в каждую ветвь:
```python
alg = PMP_BD_Merged(
    kernel_size=51,
    legacy_kwargs={'pre_nonblind': 'bm3d'},
    robust_kwargs={},
)
```

## Правила маршрутизации

Применяются по порядку, первый сработавший — выигрывает:

1. `impulse_density ≥ 1%` → **robust** (медиана + ACT каскадно сильнее)
2. PCA Pyatykh `a_norm > 1e-3` (Poisson signature) → **robust** (VST)
3. `_is_truly_correlated` (lag1≥0.5 AND radial-CV≥0.3) → **legacy**
4. Сильный белый: `psd_sigma_norm > 0.015` → **robust**
5. иначе (clean / лёгкий AWGN) → **legacy**

`heavy_sigma_threshold`, `impulse_density_heavy`, `poisson_a_threshold`
можно тюнить через kwargs конструктора.

## Проверка маршрутизации без запуска blind deconv

```bash
.\.venv\Scripts\python.exe -m src.blinddeconv.algorithms.blind_deconvolution.\
our_company.patch_wise_minimum_pixels_prior.pmp_denoise_merge.test_router
```

Для синтетики с известным ядром все 8 эталонных случаев маршрутизируются
в эмпирически побеждающую ветвь (см. таблицу в `pmp_merged.py`).

## Конфликты — нет

Оба алгоритма — отдельные классы, делящие лишь общий API `process(image)`.
Диспетчер инстанцирует *ровно один* из них на каждый вызов `process`,
поэтому никакие настройки между ними не пересекаются.
