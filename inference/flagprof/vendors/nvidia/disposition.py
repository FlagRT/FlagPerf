# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Source disposition delegates activity and category facts to the format."""
from __future__ import annotations

def classifier(events, inventory, taxonomy):
    raw = {str(r['loc']): r['category'] for r in inventory}
    context = taxonomy.context(events)
    def classify(e):
        category = raw.get(str(e['_linker_loc']))
        if category is not None:
            return category, 'not_normalized'
        if e.get('cat') in taxonomy.contract['roles']['allowed_non_device_categories']:
            return None, 'scope_annotation'
        if taxonomy.is_device(e, context):
            return 'unknown', 'unrecognized_device_activity'
        return None, 'host_or_metadata'
    return classify
