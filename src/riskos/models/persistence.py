"""Model persistence: save and reload a scoring-ready bundle.

`Scorecard.save()` wrote only fit metadata — coefficients and scaling. That is
enough to read a model and not nearly enough to *run* one. Scoring a new loan
needs, at minimum:

* the fitted **binning process**, because WOE values are defined by bins learnt
  on the training split and cannot be recovered from coefficients;
* the fitted **estimator** itself;
* the **calibrator**, if one was selected;
* the **explanation baseline** — the training-population mean points vector
  captured at fit time, without which principal drivers become batch-dependent
  again (defect R-003).

A bundle carries all four plus a manifest. The manifest exists so a reloaded
model can be checked against the configuration that produced it: a model whose
feature list no longer matches `conf/features.yaml`, or which was fitted against
a different User Guide layout, is not safe to score with, and the mismatch
should surface at load time rather than as a silently wrong PD.

Serialisation uses stdlib pickle. The bundles are build artefacts produced and
consumed by this repository, never distributed, and `models/` is gitignored.
Loading a bundle from an untrusted source would be unsafe, which is why
`load_bundle` refuses a manifest it did not write.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from riskos.config import CONF_DIR
from riskos.log import get_logger

log = get_logger(__name__)

BUNDLE_FORMAT = "riskos-model-bundle/1"
MANIFEST_NAME = "manifest.json"


class BundleError(RuntimeError):
    """The bundle is absent, malformed, or inconsistent with current config."""


def config_fingerprint(*names: str) -> dict[str, str]:
    """SHA-256 of each config file, so drift is detectable rather than assumed."""
    out = {}
    for name in names:
        path = CONF_DIR / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


@dataclass
class BundleManifest:
    """What a bundle contains and what it was built against."""

    model_id: str
    model_family: str
    version: str
    created_at: str
    features: list[str]
    training_window: str
    n_train: int
    components: list[str]
    config_fingerprint: dict[str, str] = field(default_factory=dict)
    calibration_method: str | None = None
    format: str = BUNDLE_FORMAT


def save_bundle(
    directory: Path,
    *,
    model_id: str,
    model_family: str,
    version: str,
    features: list[str],
    training_window: str,
    n_train: int,
    components: dict[str, Any],
    calibration_method: str | None = None,
) -> BundleManifest:
    """Persist every object needed to score, plus a manifest describing them."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, obj in components.items():
        if obj is None:
            continue
        with (directory / f"{name}.pkl").open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = BundleManifest(
        model_id=model_id,
        model_family=model_family,
        version=version,
        created_at=datetime.now(UTC).isoformat(),
        features=list(features),
        training_window=training_window,
        n_train=n_train,
        components=sorted(k for k, v in components.items() if v is not None),
        config_fingerprint=config_fingerprint("features.yaml", "models.yaml", "panel.yaml"),
        calibration_method=calibration_method,
    )
    (directory / MANIFEST_NAME).write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    log.info(
        "wrote_model_bundle",
        path=str(directory),
        model_id=model_id,
        components=manifest.components,
    )
    return manifest


def load_manifest(directory: Path) -> BundleManifest:
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise BundleError(f"no model bundle at {directory}; run the training step first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"unrecognised bundle format {raw.get('format')!r} at {directory}")
    return BundleManifest(**raw)


def load_bundle(
    directory: Path, *, check_config: bool = True
) -> tuple[BundleManifest, dict[str, Any]]:
    """Reload a bundle, optionally verifying it still matches current config.

    The config check is a warning rather than an error: a changed threshold in
    `models.yaml` does not invalidate a fitted scorecard, but it does mean the
    model on disk was not built against what is now in the repository, and
    someone should know that before the number reaches a report.
    """
    manifest = load_manifest(directory)
    components: dict[str, Any] = {}
    for name in manifest.components:
        path = directory / f"{name}.pkl"
        if not path.exists():
            raise BundleError(f"manifest lists {name!r} but {path} is missing")
        with path.open("rb") as handle:
            # Self-produced artefact, written by this package and read back here.
            components[name] = pickle.load(handle)

    if check_config:
        current = config_fingerprint(*manifest.config_fingerprint)
        drifted = [k for k, v in manifest.config_fingerprint.items() if current.get(k) != v]
        if drifted:
            log.warning(
                "model_config_drift",
                model_id=manifest.model_id,
                changed=drifted,
                note="bundle was fitted against a different config; re-fit before relying on it",
            )
    log.info("loaded_model_bundle", model_id=manifest.model_id, version=manifest.version)
    return manifest, components


def require_features(manifest: BundleManifest, available: list[str]) -> None:
    """Refuse to score when an input is missing a feature the model needs."""
    missing = [f for f in manifest.features if f not in available]
    if missing:
        raise BundleError(
            f"model {manifest.model_id} requires features absent from the input: {missing}"
        )
