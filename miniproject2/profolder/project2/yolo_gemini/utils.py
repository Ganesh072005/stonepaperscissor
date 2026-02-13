import os
import time
from dataclasses import dataclass
from typing import Iterable, List
from pathlib import Path

from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def cinfo(msg: str) -> str:
    return f"{Fore.CYAN}{msg}{Style.RESET_ALL}"


def csuccess(msg: str) -> str:
    return f"{Fore.GREEN}{msg}{Style.RESET_ALL}"


def cwarn(msg: str) -> str:
    return f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"


def cerror(msg: str) -> str:
    return f"{Fore.RED}{msg}{Style.RESET_ALL}"


def ensure_output_dir(output_dir: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def list_images(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    return [str(fp.resolve()) for fp in p.iterdir() if fp.is_file() and is_image_file(str(fp))]


def validate_existing_path(path: str) -> bool:
    return Path(path).exists()


def validate_file(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.is_file()


@dataclass
class Timer:
    name: str = "timer"
    start: float = 0.0
    end: float = 0.0

    def __enter__(self):
        self.start = time.time()
        print(cinfo(f"Started {self.name}..."))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
        print(csuccess(f"Finished {self.name} in {self.elapsed_ms:.2f} ms ({self.elapsed:.3f} s)"))

    @property
    def elapsed(self) -> float:
        if self.end == 0.0:
            return time.time() - self.start
        return self.end - self.start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_output_path(out_dir: Path, base: str, ext: str) -> str:
    """Generate a collision-safe path inside out_dir with base and ext.
    If base.ext exists, append _001, _002, ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_clean = base.replace(" ", "_")
    candidate = out_dir / f"{base_clean}{ext}"
    if not candidate.exists():
        return str(candidate)
    idx = 1
    while True:
        cand = out_dir / f"{base_clean}_{idx:03d}{ext}"
        if not cand.exists():
            return str(cand)
        idx += 1
