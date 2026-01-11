"""
Достук с среде Octave.

Автор: Юров П.И.
"""
import os
import sys
from pathlib import Path
from oct2py import Oct2Py
from configs.octaveconfig import octavesettings

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
            platform_msg = ""
            if sys.platform.startswith("win"):
                platform_msg = "Ensure 'octave-cli.exe' is used, NOT 'octave.exe' or 'octave-gui.exe'."
            else:
                platform_msg = "Ensure the path in .env is correct and has execution permissions."
            raise RuntimeError(
                f"Failed to initialize Octave Engine.\n"
                f"Path: {executable}\n"
                f"Hint: {platform_msg}\n"
                f"Error details: {e}"
            ) from e
        return cls._instance