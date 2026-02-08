import os
import shutil
from typing import Optional, Annotated
from pydantic import Field, AfterValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

def validate_octave_path(v: Optional[str]) -> Optional[str]:
    if v is None:
        print(
            "Warning: Octave executable not found. "
            "Some of algorithms may not work. "
            "Please install Octave or set "
            "OCTAVE_EXECUTABLE in your .env file. "
        )
        return v
    if not os.path.isfile(v):
        raise ValueError(f"The path provided for Octave does not exist: {v}")
    
    return v

OctavePathType = Annotated[Optional[str], AfterValidator(validate_octave_path)]

class OctaveConfig(BaseSettings):
    octave_executable: OctavePathType = Field(
        default_factory=lambda: shutil.which("octave-cli") or shutil.which("octave"),
        description="Path to the Octave CLI executable"
    )

    debug_mode: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

octavesettings = OctaveConfig()