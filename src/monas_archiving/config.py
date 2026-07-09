from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ObjectiveSpec:
    """Column name and direction for one objective."""

    column: str
    direction: str

    def __post_init__(self) -> None:
        direction = self.direction.lower()
        if direction not in {"minimize", "maximize"}:
            raise ValueError(
                f"Invalid direction for {self.column!r}: {self.direction!r}. "
                "Use 'minimize' or 'maximize'."
            )
        object.__setattr__(self, "direction", direction)

    @property
    def normalized_column(self) -> str:
        return f"norm_{self.column}"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a full offline archiving run."""

    run_name: str
    input_path: Path
    output_dir: Path
    objectives: tuple[ObjectiveSpec, ...]
    architecture_id_column: str = "architecture_id"
    chromosome_column: str | None = "chromosome"
    deduplication_key: str = "architecture_id"
    normalization: str = "minmax"
    seed: int = 1
    archivers: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    truncation_sizes: tuple[int, ...] = (5,)
    indicators: tuple[str, ...] = ("igd_plus", "hypervolume", "r2", "epsilon", "hausdorff")
    hv_reference_point: tuple[float, ...] | None = None
    plot: bool = True

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def normalized_objective_columns(self) -> list[str]:
        return [objective.normalized_column for objective in self.objectives]


def _as_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_objectives(raw: Any) -> tuple[ObjectiveSpec, ...]:
    if not raw:
        raise ValueError("Config must define at least one objective.")

    objectives: list[ObjectiveSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each objective must be a mapping with column and direction.")
        column = item.get("column") or item.get("name")
        direction = item.get("direction")
        if not column or not direction:
            raise ValueError("Each objective must define 'column' and 'direction'.")
        objectives.append(ObjectiveSpec(column=str(column), direction=str(direction)))
    return tuple(objectives)


def _load_archivers(raw: Any) -> tuple[dict[str, Any], ...]:
    if not raw:
        return (
            {"name": "pq"},
            {"name": "crowding"},
            {"name": "grid"},
            {"name": "epsilon"},
            {"name": "tight1"},
            {"name": "kmeans"},
            {"name": "entropy"},
            {"name": "hv"},
            {"name": "r2"},
        )

    archivers: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            archivers.append({"name": item})
        elif isinstance(item, dict) and item.get("name"):
            archivers.append(dict(item))
        else:
            raise ValueError("Archivers must be names or mappings with a 'name' field.")
    return tuple(archivers)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML pipeline configuration."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent.parent

    output_dir = _as_path(base_dir, raw.get("output_dir", "results"))
    input_path = _as_path(base_dir, raw["input_path"])

    return PipelineConfig(
        run_name=str(raw.get("run_name", config_path.stem)),
        input_path=input_path,
        output_dir=output_dir,
        objectives=_load_objectives(raw.get("objectives")),
        architecture_id_column=str(raw.get("architecture_id_column", "architecture_id")),
        chromosome_column=raw.get("chromosome_column", "chromosome"),
        deduplication_key=str(raw.get("deduplication_key", raw.get("architecture_id_column", "architecture_id"))),
        normalization=str(raw.get("normalization", "minmax")),
        seed=int(raw.get("seed", 1)),
        archivers=_load_archivers(raw.get("archivers")),
        truncation_sizes=tuple(int(value) for value in raw.get("truncation_sizes", [5])),
        indicators=tuple(str(value) for value in raw.get("indicators", ["igd_plus", "hypervolume", "r2", "epsilon", "hausdorff"])),
        hv_reference_point=tuple(float(value) for value in raw["hv_reference_point"]) if raw.get("hv_reference_point") else None,
        plot=bool(raw.get("plot", True)),
    )


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Write the effective configuration used by a run."""
    payload = {
        "run_name": config.run_name,
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "architecture_id_column": config.architecture_id_column,
        "chromosome_column": config.chromosome_column,
        "deduplication_key": config.deduplication_key,
        "normalization": config.normalization,
        "seed": config.seed,
        "objectives": [
            {"column": objective.column, "direction": objective.direction}
            for objective in config.objectives
        ],
        "archivers": list(config.archivers),
        "truncation_sizes": list(config.truncation_sizes),
        "indicators": list(config.indicators),
        "hv_reference_point": list(config.hv_reference_point) if config.hv_reference_point else None,
        "plot": config.plot,
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
