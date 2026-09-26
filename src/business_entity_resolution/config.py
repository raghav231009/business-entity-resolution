"""
config.py — Central configuration loader
==========================================
Reads ``config.yaml``, resolves paths relative to the project root,
and exposes a single ``Config`` dataclass consumed by every module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Path to the default config.yaml (next to the project root)
# ------------------------------------------------------------------ #
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/business_entity_resolution
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ================================================================== #
#  Dataclass tree
# ================================================================== #

@dataclass
class PathsConfig:
    """All filesystem paths (resolved to absolute)."""
    train_source1: Path = Path()
    train_source2: Path = Path()
    train_source3: Path = Path()
    train_ground_truth: Path = Path()
    test_source1: Path = Path()
    test_source2: Path = Path()
    test_source3: Path = Path()
    models_dir: Path = Path()
    artifacts_dir: Path = Path()
    output_dir: Path = Path()
    candidate_cache_dir: Path = Path()
    feature_cache_dir: Path = Path()
    metrics_dir: Path = Path()
    reports_dir: Path = Path()


@dataclass
class PreprocessingConfig:
    """Normalization settings."""
    legal_suffixes: List[str] = field(default_factory=list)
    address_abbreviations: Dict[str, str] = field(default_factory=dict)


@dataclass
class BlockingConfig:
    """Candidate generation / blocking parameters."""
    # Broad single-field blocks (country alone disabled by default to prevent explosion)
    country_block: bool = False
    name_token_block: bool = True
    name_prefix_block: bool = True
    name_prefix_length: int = 4
    address_token_block: bool = True
    postal_code_block: bool = True
    char_ngram_block: bool = True
    char_ngram_n: int = 3
    char_ngram_min_shared: int = 2

    # Compound / informative multi-channel blocking keys
    country_name_token_block: bool = True
    country_name_prefix_block: bool = True
    country_postal_block: bool = True
    country_address_token_block: bool = True
    name_token_postal_block: bool = True
    name_prefix_postal_block: bool = True
    name_token_address_token_block: bool = True

    # Candidate cap and similarity pre-ranking
    max_candidates_per_s1: int = 200
    similarity_prerank: bool = True
    # Prune uninformative giant blocks that exceed max_block_size
    max_block_size: int = 500


@dataclass
class TfidfConfig:
    """TF-IDF vectorizer settings."""
    name_word_ngram_range: tuple = (1, 2)
    name_char_ngram_range: tuple = (2, 4)
    address_word_ngram_range: tuple = (1, 2)
    address_char_ngram_range: tuple = (2, 4)
    max_features: int = 10000


@dataclass
class FeaturesConfig:
    """Which features to compute."""
    # Name
    name_exact: bool = True
    name_fuzzy_ratio: bool = True
    name_partial_ratio: bool = True
    name_token_sort_ratio: bool = True
    name_token_set_ratio: bool = True
    name_jaccard: bool = True
    name_char_sim: bool = True
    name_tfidf_cosine: bool = True
    name_length_diff: bool = True
    name_token_count_diff: bool = True
    name_common_tokens: bool = True
    # Address
    address_exact: bool = True
    address_fuzzy: bool = True
    address_token_sim: bool = True
    address_jaccard: bool = True
    address_char_sim: bool = True
    address_tfidf_cosine: bool = True
    address_length_diff: bool = True
    address_common_tokens: bool = True
    address_numeric_overlap: bool = True
    postal_code_match: bool = True
    house_number_match: bool = True
    # Country
    country_exact: bool = True
    # Source
    source_indicator: bool = True
    # TF-IDF
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)


@dataclass
class ModelConfig:
    """Model hyper-parameters."""
    type: str = "lightgbm"
    lightgbm: Dict[str, Any] = field(default_factory=dict)
    catboost: Dict[str, Any] = field(default_factory=dict)
    saved_model_name: str = "entity_matcher"


@dataclass
class ExecutionConfig:
    """Execution mode parameters."""
    mode: str = "dev"                     # "dev" | "final"
    max_train_s1_entities: Optional[int] = 50000
    max_validation_s1_entities: Optional[int] = 10000


@dataclass
class TrainingConfig:
    """Training parameters."""
    validation_split: float = 0.2
    negative_sampling_ratio: Optional[float] = None
    max_negatives_per_positive: int = 20
    shuffle: bool = True
    max_train_s1_entities: Optional[int] = 50000
    max_validation_s1_entities: Optional[int] = 10000


@dataclass
class ThresholdConfig:
    """Threshold tuning parameters."""
    search_range: List[float] = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    default: float = 0.50
    selected: Optional[float] = None


@dataclass
class PostprocessingConfig:
    """Singleton-protection & postprocessing rules."""
    min_probability: float = 0.50
    min_name_similarity: float = 0.0
    min_address_similarity: float = 0.0
    require_country_match: bool = False


@dataclass
class CacheConfig:
    """Caching behaviour."""
    enabled: bool = True
    invalidate: bool = False


# ================================================================== #
#  Top-level Config
# ================================================================== #

@dataclass
class Config:
    """Top-level configuration container."""
    random_seed: int = 42
    project_root: Path = _PROJECT_ROOT
    log_level: str = "INFO"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


# ================================================================== #
#  Loader helpers
# ================================================================== #

def _resolve_path(root: Path, raw: str) -> Path:
    """Return an absolute path resolved against *root*."""
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def _build_paths(root: Path, raw: Dict) -> PathsConfig:
    train = raw.get("train", {})
    test = raw.get("test", {})
    return PathsConfig(
        train_source1=_resolve_path(root, train.get("source1", "")),
        train_source2=_resolve_path(root, train.get("source2", "")),
        train_source3=_resolve_path(root, train.get("source3", "")),
        train_ground_truth=_resolve_path(root, train.get("ground_truth", "")),
        test_source1=_resolve_path(root, test.get("source1", "")),
        test_source2=_resolve_path(root, test.get("source2", "")),
        test_source3=_resolve_path(root, test.get("source3", "")),
        models_dir=_resolve_path(root, raw.get("models_dir", "models")),
        artifacts_dir=_resolve_path(root, raw.get("artifacts_dir", "artifacts")),
        output_dir=_resolve_path(root, raw.get("output_dir", "output")),
        candidate_cache_dir=_resolve_path(root, raw.get("candidate_cache_dir", "artifacts/candidate_cache")),
        feature_cache_dir=_resolve_path(root, raw.get("feature_cache_dir", "artifacts/feature_cache")),
        metrics_dir=_resolve_path(root, raw.get("metrics_dir", "artifacts/metrics")),
        reports_dir=_resolve_path(root, raw.get("reports_dir", "reports")),
    )


def _build_blocking(raw: Dict) -> BlockingConfig:
    strats = raw.get("strategies", {})
    return BlockingConfig(
        country_block=strats.get("country_block", False),
        name_token_block=strats.get("name_token_block", True),
        name_prefix_block=strats.get("name_prefix_block", True),
        name_prefix_length=strats.get("name_prefix_length", 4),
        address_token_block=strats.get("address_token_block", True),
        postal_code_block=strats.get("postal_code_block", True),
        char_ngram_block=strats.get("char_ngram_block", True),
        char_ngram_n=strats.get("char_ngram_n", 3),
        char_ngram_min_shared=strats.get("char_ngram_min_shared", 2),
        country_name_token_block=strats.get("country_name_token_block", True),
        country_name_prefix_block=strats.get("country_name_prefix_block", True),
        country_postal_block=strats.get("country_postal_block", True),
        country_address_token_block=strats.get("country_address_token_block", True),
        name_token_postal_block=strats.get("name_token_postal_block", True),
        name_prefix_postal_block=strats.get("name_prefix_postal_block", True),
        name_token_address_token_block=strats.get("name_token_address_token_block", True),
        max_candidates_per_s1=raw.get("max_candidates_per_s1", 200),
        similarity_prerank=raw.get("similarity_prerank", True),
        max_block_size=raw.get("max_block_size", 500),
    )


def _build_features(raw: Dict) -> FeaturesConfig:
    tfidf_raw = raw.pop("tfidf", {})
    tfidf = TfidfConfig(
        name_word_ngram_range=tuple(tfidf_raw.get("name_word_ngram_range", [1, 2])),
        name_char_ngram_range=tuple(tfidf_raw.get("name_char_ngram_range", [2, 4])),
        address_word_ngram_range=tuple(tfidf_raw.get("address_word_ngram_range", [1, 2])),
        address_char_ngram_range=tuple(tfidf_raw.get("address_char_ngram_range", [2, 4])),
        max_features=tfidf_raw.get("max_features", 10000),
    )
    # Build remaining feature flags
    fc = FeaturesConfig(tfidf=tfidf)
    for key in (
        "name_exact", "name_fuzzy_ratio", "name_partial_ratio",
        "name_token_sort_ratio", "name_token_set_ratio",
        "name_jaccard", "name_char_sim", "name_tfidf_cosine",
        "name_length_diff", "name_token_count_diff", "name_common_tokens",
        "address_exact", "address_fuzzy", "address_token_sim",
        "address_jaccard", "address_char_sim", "address_tfidf_cosine",
        "address_length_diff", "address_common_tokens",
        "address_numeric_overlap", "postal_code_match", "house_number_match",
        "country_exact", "source_indicator",
    ):
        if key in raw:
            setattr(fc, key, raw[key])
    return fc


def load_config(
    config_path: Optional[str | Path] = None,
    data_root: Optional[str | Path] = None,
    mode: Optional[str] = None,
) -> Config:
    """
    Load configuration from a YAML file and return a ``Config`` instance.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to ``config.yaml``. Defaults to the file in the project root.
    data_root : str or Path, optional
        Optional root directory for dataset files (e.g. ``data`` or ``data_dev``).
    mode : str, optional
        Execution mode override: ``"dev"`` or ``"final"``.

    Returns
    -------
    Config
    """
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file %s not found — using defaults.", config_path)
        return Config()

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    general = raw.get("general", {})
    proj_root_raw = general.get("project_root", ".")
    proj_root_path = Path(proj_root_raw)
    if not proj_root_path.is_absolute():
        root = (config_path.parent / proj_root_path).resolve()
    else:
        root = proj_root_path.resolve()

    # Determine execution mode ("dev" vs "final")
    exec_raw = raw.get("execution", {})
    configured_mode = mode if mode is not None else exec_raw.get("mode", "dev")
    if configured_mode not in ("dev", "final"):
        logger.warning("Unrecognised mode '%s' — defaulting to 'dev'", configured_mode)
        configured_mode = "dev"

    dev_settings = raw.get("development", {})
    final_settings = raw.get("final", {})

    if configured_mode == "final":
        max_train = final_settings.get("max_train_s1_entities", None)
        max_val = final_settings.get("max_validation_s1_entities", None)
    else:
        max_train = dev_settings.get("max_train_s1_entities", 50000)
        max_val = dev_settings.get("max_validation_s1_entities", 10000)

    cfg = Config(
        random_seed=general.get("random_seed", 42),
        project_root=root,
        log_level=general.get("log_level", "INFO"),
        execution=ExecutionConfig(
            mode=configured_mode,
            max_train_s1_entities=max_train,
            max_validation_s1_entities=max_val,
        ),
        paths=_build_paths(root, raw.get("paths", {})),
        preprocessing=PreprocessingConfig(
            legal_suffixes=raw.get("preprocessing", {}).get("legal_suffixes", []),
            address_abbreviations=raw.get("preprocessing", {}).get("address_abbreviations", {}),
        ),
        blocking=_build_blocking(raw.get("blocking", {})),
        features=_build_features(dict(raw.get("features", {}))),   # dict() copy — we pop from it
        model=ModelConfig(
            type=raw.get("model", {}).get("type", "lightgbm"),
            lightgbm=raw.get("model", {}).get("lightgbm", {}),
            catboost=raw.get("model", {}).get("catboost", {}),
            saved_model_name=raw.get("model", {}).get("saved_model_name", "entity_matcher"),
        ),
        training=TrainingConfig(
            validation_split=raw.get("training", {}).get("validation_split", 0.2),
            negative_sampling_ratio=raw.get("training", {}).get("negative_sampling_ratio"),
            max_negatives_per_positive=raw.get("training", {}).get("max_negatives_per_positive", 20),
            shuffle=raw.get("training", {}).get("shuffle", True),
            max_train_s1_entities=max_train,
            max_validation_s1_entities=max_val,
        ),
        threshold=ThresholdConfig(
            search_range=raw.get("threshold", {}).get("search_range",
                                                       [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]),
            default=raw.get("threshold", {}).get("default", 0.50),
            selected=raw.get("threshold", {}).get("selected"),
        ),
        postprocessing=PostprocessingConfig(
            min_probability=raw.get("postprocessing", {}).get("min_probability", 0.50),
            min_name_similarity=raw.get("postprocessing", {}).get("min_name_similarity", 0.0),
            min_address_similarity=raw.get("postprocessing", {}).get("min_address_similarity", 0.0),
            require_country_match=raw.get("postprocessing", {}).get("require_country_match", False),
        ),
        cache=CacheConfig(
            enabled=raw.get("cache", {}).get("enabled", True),
            invalidate=raw.get("cache", {}).get("invalidate", False),
        ),
    )

    if data_root is not None:
        droot = Path(data_root)
        if not droot.is_absolute():
            droot = (root / droot).resolve()
        cfg.paths.train_source1 = droot / "train" / "train_source1.tsv"
        cfg.paths.train_source2 = droot / "train" / "train_source2.tsv"
        cfg.paths.train_source3 = droot / "train" / "train_source3.tsv"
        cfg.paths.train_ground_truth = droot / "train" / "train_ground_truth.tsv"
        cfg.paths.test_source1 = droot / "test" / "test_source1.tsv"
        cfg.paths.test_source2 = droot / "test" / "test_source2.tsv"
        cfg.paths.test_source3 = droot / "test" / "test_source3.tsv"
        logger.info("Dataset paths overridden with data_root: %s", droot)

    logger.info("Configuration loaded from %s  (project_root=%s, mode=%s, max_train_s1=%s, max_val_s1=%s)",
                config_path, cfg.project_root, cfg.execution.mode,
                cfg.execution.max_train_s1_entities, cfg.execution.max_validation_s1_entities)
    return cfg
