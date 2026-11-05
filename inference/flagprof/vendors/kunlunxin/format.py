# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Independent, versioned P800 contract over the observed Kineto syntax."""
from dataclasses import dataclass
from flagprof.vendors.nvidia.format import Taxonomy as KinetoTaxonomy
from flagprof.vendors.nvidia.format import NATIVE_CALL_CATEGORIES, ANNOTATION_CATEGORIES
from flagprof.vendors.contracts import load_contract, closure
from flagprof.vendors.fingerprint import format_identity
import flagprof.vendors.nvidia.format as shared
VERSION = '1'
@dataclass(frozen=True)
class Taxonomy(KinetoTaxonomy):
    version: str = VERSION
    def __post_init__(self):
        if self.version != VERSION:raise ValueError('unsupported P800 taxonomy version: '+str(self.version))
        object.__setattr__(self,'contract',load_contract(__file__,self.version))
    def identity(self):
        return dict(format_identity(self.version,(*closure(__file__,self.version),shared.__file__)),revision=self.contract['revision'])
