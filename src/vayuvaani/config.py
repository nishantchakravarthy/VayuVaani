"""Load config/settings.json and secrets from .env."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        p = self.root / self.raw["paths"][key]
        if key != "events":
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def city(self) -> dict[str, Any]:
        return self.raw["city"]

    @property
    def sources(self) -> dict[str, Any]:
        return self.raw["sources"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def timezone(self) -> str:
        return self.city["timezone"]


def load_settings(path: str | Path | None = None, root: str | Path | None = None) -> Settings:
    root = Path(root) if root else PROJECT_ROOT
    load_dotenv(root / ".env")
    path = Path(path) if path else root / "config" / "settings.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return Settings(raw=raw, root=root)


def secret(name: str, required: bool = True) -> str | None:
    value = os.getenv(name, "").strip()
    if required and not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and add your key."
        )
    return value or None
