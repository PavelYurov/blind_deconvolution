import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

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


# Импортируем решатель из локального файла
from .solvers import solve_approx_bivariate_gcd

class BezoutDeconvolution(DeconvolutionAlgorithm):
    """
    Blind image deconvolution through Bezoutians.
    Алгоритм требует несколько (>=2) размытых версий одного изображения.
    """
    
    def __init__(self, name: str = "BezoutDeconv"):
        super().__init__(name)
        # Epsilon влияет на оценку ранга (отсечение шума)
        self.epsilon = 1e-2 
        
    def change_param(self, param: Dict[str, Any]) -> None:
        if 'epsilon' in param:
            self.epsilon = float(param['epsilon'])

    def get_param(self) -> List[Tuple[str, Any]]:
        return [('epsilon', self.epsilon)]

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обработка изображения.
        
        Parameters
        ----------
        image : np.ndarray
            Входные данные. Ожидается стек черно-белых изображений размера (H, W, S),
            где S >= 2.
        """
        start_time = time.time()
        
        # Проверка входных данных
        if image.ndim == 2:
            print(f"[{self.name}] Error: Input must be a stack of images (H, W, S) with S>=2.")
            self.timer = time.time() - start_time
            return image, np.ones((3, 3)) / 9.0
            
        if image.ndim == 3:
            h, w, s = image.shape
            if s < 2:
                print(f"[{self.name}] Error: At least 2 blurred images required. Got {s}.")
                self.timer = time.time() - start_time
                return image[:, :, 0], np.ones((3, 3)) / 9.0
        else:
            print(f"[{self.name}] Error: Invalid shape {image.shape}.")
            self.timer = time.time() - start_time
            return image, np.zeros((3,3))

        # Запуск алгоритма
        try:
            restored, kernel = solve_approx_bivariate_gcd(image, self.epsilon)
        except Exception as e:
            print(f"[{self.name}] Critical error during execution: {e}")
            import traceback
            traceback.print_exc()
            restored = image[:, :, 0]
            kernel = np.ones((5, 5)) / 25.0
            
        self.timer = time.time() - start_time
        return restored, kernel