# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Controlled vendor-code loading for embedding and rerank models (opt-in).

Some HF embedding/rerank models ship custom modeling code referenced by
``config.json`` ``auto_map`` (e.g. jina-embeddings-v3 -> XLMRobertaLoRA,
jina-reranker-v2 -> XLMRobertaForSequenceClassification).
The runner refuses the transformers remote-code path (``trust_remote_code``
stays False everywhere); the only sanctioned entry is ``--trust-remote-code``
on ``--task embedding`` or ``--task rerank``, which loads vendor code that is
ALREADY on disk inside the model directory through a local-package bypass:

- the model directory itself is imported as a Python package, so the
  vendor's modeling file and its multi-level relative imports resolve from
  disk — the transformers dynamic-module cache (which copies only one level
  of relative imports, broken on transformers 4.57) is never engaged;
- ``auto_map`` entries must be local (no ``repo--class`` remote prefix);
- every consumed ``*.py`` file is sha256-pinned by ``code_files_identity``
  and travels in the oracle identity (``code_files``), so accuracy runs
  prove which vendor code produced the outputs;
- the vendor flash-attention path is force-disabled (``use_flash_attn``
  False when present) so execution uses native attention.

Standard models are unaffected: with the flag set but no ``*.py`` files in
the model directory the loader falls through to the plain transformers path.
"""
import hashlib
import json
from pathlib import Path
import sys


def code_files_identity(model_path):
    """sha256 of every vendor *.py shipped in the model directory."""
    root = Path(model_path)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}


def load_with_vendor_code(model_path, dtype, auto_key='AutoModel'):
    """Import the model dir as a package and build the auto_map model class.

    auto_key selects the class family from auto_map: 'AutoModel' (embedding)
    or 'AutoModelForSequenceClassification' / 'AutoModelForCausalLM' (rerank).
    """
    import importlib
    import importlib.util
    root = Path(model_path)
    config = json.loads((root / 'config.json').read_text())
    auto_map = config.get('auto_map') or {}
    if auto_key not in auto_map or '--' in str(auto_map[auto_key]):
        raise ValueError('vendor-code load needs a local auto_map ' + auto_key + ' entry')
    module_name, _, class_name = auto_map[auto_key].rpartition('.')
    identity = json.dumps([str(root.resolve()), code_files_identity(root)], sort_keys=True)
    pkg_name = 'flagprof_vendor_' + hashlib.sha256(identity.encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        pkg_name, root / (module_name + '.py'), submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = package
    spec.loader.exec_module(package)
    model_cls = getattr(package, class_name)
    config_cls = None
    config_map = auto_map.get('AutoConfig')
    if config_map and '--' not in str(config_map):
        config_module, _, config_class = config_map.rpartition('.')
        config_cls = getattr(importlib.import_module(pkg_name + '.' + config_module), config_class)
    if hasattr(model_cls, 'from_pretrained'):
        cfg = config_cls.from_pretrained(str(root), local_files_only=True) if config_cls else None
        if cfg is not None and hasattr(cfg, 'use_flash_attn'):
            cfg.use_flash_attn = False
        load_kwargs = dict(local_files_only=True, torch_dtype=dtype)
        if cfg is not None:
            model = model_cls.from_pretrained(str(root), config=cfg, **load_kwargs)
        else:
            model = model_cls.from_pretrained(str(root), **load_kwargs)
    else:
        cfg = config_cls() if config_cls else None
        if cfg is not None and hasattr(cfg, 'use_flash_attn'):
            cfg.use_flash_attn = False
        model = model_cls(cfg) if cfg is not None else model_cls()
    return model
