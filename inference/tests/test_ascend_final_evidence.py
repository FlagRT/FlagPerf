# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Real SQLite rows, sealed inputs and workload declarations for Ascend final."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from flagprof.offline.checkpoints import fileinfo
from flagprof.offline.final_evidence import identities
from flagprof.vendors.ascend.evidence import schema_version, stage_format_evidence
from flagprof.vendors.ascend.session import AscendProfiler
from flagprof.vendors.contracts import identity_facts
from flagprof.vendors.final_session import FinalSession


class AscendFinalEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root/'sealed-raw'/'data'
        self.trace = self.root/'sealed-trace'/'data'/'trace.json'
        self.export = self.root/'official'
        for directory in (self.raw, self.trace.parent, self.export):
            directory.mkdir(parents=True)
        self.trace.write_text('[]')
        self.official_trace = self.export/'trace_view.json'
        self.official_trace.write_text('[]')
        self.db = self.export/'ascend_pytorch_profiler_0.db'
        self.manifest = {'files': {}}

    def config(self, name, version):
        path = self.raw/name
        path.write_text(json.dumps({'cann_version': version, 'config': {
            'experimental_config': {'_data_simplification': True}}}))
        self.manifest['files']['data/'+name] = fileinfo(path)

    def capture_facts(self, metadata=None):
        return identities(self.root, metadata or {}, 'ascend', self.trace, 'capture-id', self.manifest)

    def test_uses_sealed_values_including_other_versions(self):
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE META_DATA(name, value)')
            db.execute('INSERT INTO META_DATA VALUES (?,?)', ('SCHEMA_VERSION', '2.7.test'))
        stage_format_evidence(self.official_trace, self.trace.parent, self.db.stat().st_size)
        self.config('profiler_info.json', '9.1.test')
        record = self.capture_facts()
        facts = identity_facts(record)
        self.assertEqual(facts['sdk']['version'], '9.1.test')
        self.assertEqual(facts['exporter']['schema_version'], '2.7.test')
        proof = facts['exporter']['source'][-1]
        self.assertEqual(proof['field'], 'META_DATA[SCHEMA_VERSION]')
        self.assertEqual(proof['sha256'], fileinfo(self.root/proof['path'])['sha256'])
        self.assertFalse(facts['execution'])
        self.assertTrue(AscendProfiler().data_simplification)

    def test_missing_conflicting_and_oversize_sources_stay_unknown(self):
        self.config('profiler_info_a.json', '9.0.0')
        self.config('profiler_info_b.json', None)
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE META_DATA(name, value)')
            db.executemany('INSERT INTO META_DATA VALUES (?,?)',
                           [('SCHEMA_VERSION', '1.2.0'), ('SCHEMA_VERSION', '1.1.0')])
        self.assertIsNone(schema_version(self.db))
        stage_format_evidence(self.official_trace, self.trace.parent, 1)
        self.assertFalse((self.trace.parent/self.db.name).exists())
        facts = identity_facts(self.capture_facts())
        self.assertFalse(facts['sdk'])
        self.assertIsNone(facts['exporter']['schema_version'])
        self.db.write_bytes(b'not a database')
        self.assertIsNone(schema_version(self.db))

    def test_execution_source_and_mixed_modes(self):
        session = FinalSession(None, self.root, None)
        session.status = 'running'
        code = self.test_execution_source_and_mixed_modes.__code__
        session.declare_execution_mode('eager', code, code.co_firstlineno)
        execution = identity_facts(self.capture_facts(session.meta))['execution']
        self.assertEqual(execution['mode'], 'eager')
        self.assertEqual(execution['source']['sha256'], hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        session.declare_execution_mode('compiled', code, code.co_firstlineno)
        execution = identity_facts(self.capture_facts(session.meta))['execution']
        self.assertEqual(execution['mode'], 'mixed')
        self.assertEqual([s['mode'] for s in execution['source']], ['eager', 'compiled'])
