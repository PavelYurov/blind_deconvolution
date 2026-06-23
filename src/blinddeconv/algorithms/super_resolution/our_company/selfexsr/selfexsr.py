"""
selfexsr.py

Алгоритм сверхразрешения одиночного изображения с использованием 
трансформированных самоподобных патчей (Self-Exemplars).

Содержание алгоритма:
    1. Инициализация параметров и построение пирамиды изображений.
    2. Поиск патчей-кандидатов (PatchMatch) внутри пирамиды с учетом 
       аффинных и перспективных искажений.
    3. Реконструкция изображения высокого разрешения (HR) на основе 
       взвешенного голосования найденных патчей.
    4. Итеративная обратная проекция (back-projection) для уточнения 
       результата и согласования с исходным изображением низкого разрешения.

Интерфейсная обертка: принимает изображение, выполняет процедуру 
сверхразрешения и возвращает HR-результат вместе с фиктивным (единичным) 
ядром размытия, так как алгоритм не производит слепой оценки ФРТ.

Литература:
[1] J. Huang, A. Singh, and N. Ahuja, 
    "Single Image Super-Resolution from Transformed Self-Exemplars", 
    CVPR 2015.
"""

import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

import numpy as np

# --- Интеграция с базовым классом ---
def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm
# ------------------------------------

from .solvers import sr_demo, sr_init_opt


class SelfExSR(DeconvolutionAlgorithm):
    """
    Класс алгоритма сверхразрешения на основе самоподобных патчей (SelfExSR).

    Параметры алгоритма
    -------------------
    SRF : int
        Коэффициент масштабирования (увеличения разрешения: 2, 3, 4 или 8). 
        По умолчанию 2.
    numIter : int
        Количество итераций поиска PatchMatch на первом (самом грубом) уровне. 
        По умолчанию 15.
    nIterBP : int
        Количество итераций обратной проекции (back-projection). 
        По умолчанию 20.
    usePlaneGuide : bool
        Флаг использования направляющей планарной структуры (упрощенная модель). 
        По умолчанию False.
    useAffine : bool
        Флаг использования аффинных трансформаций в поиске PatchMatch. 
        По умолчанию True.
    """

    def __init__(
        self,
        SRF: int = 2,
        numIter: int = 15,
        nIterBP: int = 20,
        usePlaneGuide: bool = False,
        useAffine: bool = True,
    ):
        super().__init__(name='SelfExSR')

        self.SRF = SRF
        self.numIter = numIter
        self.nIterBP = nIterBP
        self.usePlaneGuide = usePlaneGuide
        self.useAffine = useAffine

        self.history: Dict[str, list] = {}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основной процесс выполнения сверхразрешения."""
        start_time = time.time()

        # Формирование параметров алгоритма
        opt = sr_init_opt(self.SRF)
        opt['numIter'] = self.numIter
        opt['nIterBP'] = self.nIterBP
        opt['usePlaneGuide'] = self.usePlaneGuide
        opt['useAffine'] = self.useAffine

        # Выполнение сверхразрешения
        img_hr = sr_demo(image, self.SRF, opt=opt)

        elapsed = time.time() - start_time

        self.hyperparams = {
            'SRF': self.SRF,
            'numIter': self.numIter,
            'nIterBP': self.nIterBP,
            'usePlaneGuide': self.usePlaneGuide,
            'useAffine': self.useAffine,
            'time': elapsed,
        }

        # Приведение к формату вывода
        img_hr = np.clip(img_hr * 255.0, 0, 255).astype(np.int16)

        # Фиктивное ядро (сохранение интерфейса)
        kernel = np.zeros((3, 3), dtype=np.float64)
        kernel[1, 1] = 1.0

        return img_hr, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('SRF', self.SRF),
            ('numIter', self.numIter),
            ('nIterBP', self.nIterBP),
            ('usePlaneGuide', self.usePlaneGuide),
            ('useAffine', self.useAffine),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams