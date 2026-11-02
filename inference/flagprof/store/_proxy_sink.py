"""Consumer-owned synchronous sink sharing importer disk mechanics."""
import threading

from ._buffered import BufferedWriter


class ProxySink(BufferedWriter):
    def __init__(self, identity):
        self._identity = identity
        self._parent_lost = False
        self._durable = 0
        self.consumer_buffered_records_peak = 0
        from .writer import WriterConfig
        super().__init__(identity['run_dir'], config=WriterConfig(**identity['config']))
        self._thread.join(10)
        self.defaults = identity['defaults']

    def _process_identity(self):
        return self._identity['proc']

    def _allocate_fragment(self, **unused):
        return self._identity['fragment']

    def emit(self, record, **unused):
        return True

    def _disk_main(self):
        self._ready.set()

    @property
    def _written(self):
        return self.written_events

    def _write_line(self, *args, **kwargs):
        super()._write_line(*args, **kwargs)
        self.consumer_buffered_records_peak = max(self.consumer_buffered_records_peak, self._handle.events)

    def _durable_barrier(self, barrier):
        super()._durable_barrier(barrier)
        self._durable = self._written

    def _publish_fragment(self):
        if self._parent_lost:
            self.fragment['state'] = 'active'
        super()._publish_fragment()
