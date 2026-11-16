"""DCMI/npu-smi health enum normalization, frozen by T15 capability evidence."""


def health_events(entity, code, timestamp):
    if code == 0:
        return [dict(t='health',ts=timestamp,kind='heartbeat',sev='info',
                     detail=dict(scope='device_health',entity=entity,healthy=True,
                                 normalization='ascend-health/1'))]
    if code not in (1, 2, 3):
        return []
    return [dict(t='health', ts=timestamp, kind='device_error',
                 sev={1: 'warn', 2: 'error', 3: 'fatal'}[code],
                 detail=dict(entity=entity, health_code=code, source='ascend.health',
                             normalization='ascend-health/1'))]
