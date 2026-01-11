"""
Достук с среде Octave.

Автор: Юров П.И.
"""
import os
import sys
from pathlib import Path
from oct2py import Oct2Py
from .octaveconfig import octavesettings

class OctaveEngine:
    """Одна среда Octave для всех алгоритмов."""
    _instance = None

    @classmethod
    def get_instance(cls):
        """Доступ к Octave."""
        if cls._instance is None:
            executable = octavesettings.octave_executable
            octave_bin_dir = str(Path(executable).parent)
            current_sys_path = os.environ.get('PATH', '')
            if octave_bin_dir not in current_sys_path:
                os.environ['PATH'] = octave_bin_dir + os.pathsep + current_sys_path
            os.environ["OCTAVE_EXECUTABLE"] = executable
        
        try:
            cls._instance = Oct2Py()
            
        except Exception as e:
            cls._instance = None
            print("Warning: Failed to initialize Octave Engine, may not able to run some algorithms")
        return cls._instance