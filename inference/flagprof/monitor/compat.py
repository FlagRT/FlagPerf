"""Versioned descriptor adaptation for historical T15 inventories, no SDK imports."""


def descriptor(value):
    result = dict(value)
    if 'roles' not in result:
        name = result['name']
        roles = []
        if result.get('busy_indicator') or name in (
                'nvidia.gpu.activity', 'ascend.aicore.activity', 'ascend.npu.activity'):
            roles.append('activity')
        legacy = {'nvidia.violation.power': 'restriction.cumulative.power',
                  'nvidia.violation.thermal': 'restriction.cumulative.thermal',
                  'power.draw': 'power', 'temperature.chip': 'temperature'}
        if name in legacy:
            roles.append(legacy[name])
        result.update(roles=roles, adapter='t15-descriptor/1')
    return result
