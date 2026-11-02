"""CANN source roles, preserving legacy raw categories verbatim."""
from __future__ import annotations
from .format import queue_key_valid

def classifier(events, inventory, taxonomy):
    raw={str(r['loc']):r['category'] for r in inventory}
    def classify(e):
        category=raw.get(str(e['_linker_loc']))
        if category is not None:
            return category,'unrecognized_device_activity'
        if e.get('name','').startswith(('Enqueue','Dequeue')):
            value=e.get('args',{}).get('correlation_id')
            if not queue_key_valid(value):
                return None,'invalid_queue_key'
        return None,'host_or_metadata'
    return classify
