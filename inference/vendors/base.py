"""Common vendor contract. Unsupported capabilities fail explicitly."""
import os


class VendorAdapter:
    def environment(self, cfg):
        env = os.environ.copy()
        env.update({k: str(v) for k, v in cfg['vendors'][cfg['runtime']['vendor']]['env'].items()})
        env.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        devices = cfg['runtime']['devices'] if cfg['runtime'].get('parallelism') == 'tp' else [cfg['runtime']['device']]
        if not env.get('FLAGPERF_CONTAINER_DEVICE'):
            env[self.visible_devices] = ','.join(map(str, devices))
        return env

    def initialize(self, cfg):
        raise ValueError('vendor adapter does not implement device execution')

    def device_session(self, backend):
        from runtime.device_session import DeviceSession, TorchDeviceSession
        if isinstance(backend, DeviceSession):
            return backend
        return TorchDeviceSession(backend, self.device_type+' torch backend', getattr(self, 'stream_id', None))

    def configure_numerics(self, cfg, backend):
        pass

    def numerical_settings(self):
        return None

    def container_arguments(self, cfg, root):
        raise ValueError('vendor adapter does not support container launch')

    def validate_image(self, cfg, image_id):
        pass

    def validate_launch_assets(self, cfg):
        pass

    def before_after_snapshot(self):
        return None

    def telemetry_provider(self):
        """Optional management-plane provider, loaded only in the collector."""
        return None

    def worker_environment(self, cfg, root, compiler_required):
        return self.environment(cfg)

    def compiler_identity(self, cfg):
        raise ValueError('selected vendor has no compiler adapter')

    def profiler(self, cfg, root, warmup, active):
        raise ValueError('selected vendor has no diagnostic profiler adapter')
