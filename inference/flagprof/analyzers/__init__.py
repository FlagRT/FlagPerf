# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Standard-library analyzers over public stored evidence."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Need:
    events: tuple = ()
    views: tuple = ()
    capabilities: tuple = ()


@dataclass
class Requires:
    any_of: tuple


@dataclass
class Result:
    id: str
    status: str
    data: Any
    coverage: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    _owner: Any = field(default=None, repr=False, compare=False)

    def close(self):
        if self._owner is not None:
            workspace, temporary = self._owner
            self._owner=None
            try:workspace.close()
            finally:temporary.cleanup()

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def __del__(self):self.close()


@dataclass
class Analyzer:
    id: str
    function: Any
    requires: Requires
    requires_min_schema: str = '1.0'

    def run(self, store, params=None):
        params=params or {}
        from .workspace import Workspace, Limits
        if isinstance(store,Workspace):
            return self.function(store,params)
        # Public StoreView entry: own derived state until Result.close(), so large
        # results remain replayable without materializing them before return.
        from flagprof.store import StoreView
        from pathlib import Path
        import tempfile
        if not isinstance(store,StoreView):raise TypeError('Analyzer.run requires StoreView')
        directory=params.get('work')
        if directory:Path(directory).mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix='analyzer-',dir=directory)
        try:
            limits=Limits(**params.get('limits',{}))
            workspace=Workspace(store,temporary.name,limits)
        except BaseException:
            temporary.cleanup();raise
        try:
            workspace.ingest()
            if self.id=='route.coverage':
                from .core import perf_op
                perf_op(workspace,params)
            result=self.function(workspace,params)
            workspace.check_budget()
        except (ValueError,OSError,KeyError) as error:
            result=Result(self.id,'unavailable',workspace.table(self.id),workspace.coverage(),[str(error)])
        except BaseException:
            workspace.close();temporary.cleanup();raise
        result._owner=(workspace,temporary)
        return result

    def descriptor(self):
        return dict(id=self.id, kind='analyzer', version='1', module=self.function.__module__,
                    factory='make_'+self.id.replace('.', '_'), requires=[], requires_min_schema=self.requires_min_schema)
