"""Internal importer factory; no proxy process or public Writer default change."""
from ._buffered import BufferedWriter


class _ImporterWriter(BufferedWriter):
    def __exit__(self, exc_type, *rest):
        self.close()
        # A durable receipt alone does not establish successful final close or
        # gzip completion. Surface failure before a service can publish success.
        if exc_type is None and (not self._closed or self.errors):
            raise OSError('import writer did not close successfully: '+ '; '.join(self.errors))


def offline_writer(run, *, config=None, **options):
    if config is None:
        from flagprof.vendors.import_common import importer_writer_config
        config = importer_writer_config()
    return _ImporterWriter(run, kind='importer', config=config, **options)
