"""Structural selection; the discovery hooks also cover the first forward."""
from .base import Probe, clock


class ModuleProbe(Probe):
    id = 'module'
    produces = ('span',)
    requires = ('torch',)

    def __init__(self, manager):
        super().__init__(manager)
        self.roots = {}
        self.selected = {}
        self.explicit = False
        self.handles = []

    def register_model(self, model, *, explicit=True):
        import torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError('registered model must be a torch module')
        self.explicit |= explicit
        if id(model) in self.roots:
            return
        self.roots[id(model)] = model
        def shape(node):
            return (type(node), tuple((name, shape(child)) for name, child in node.named_children()))
        def visit(node, path, depth, folded=False, group=None):
            children = list(node.named_children())
            container = isinstance(node, (torch.nn.Sequential, torch.nn.ModuleList))
            if depth <= 4 and (depth == 0 or children) and not (container and depth):
                self.selected[id(node)] = (path, dict(cls=type(node).__qualname__,
                    **(dict(folded=True, fold_group=group) if folded else {})))
            if folded or depth >= 4:
                return
            groups = {}
            for i, (_, child) in enumerate(children):
                groups.setdefault(shape(child), []).append(i)
            hidden = {}
            if self.config['layers'] != 'all':
                for indexes in groups.values():
                    if len(indexes) >= 4:
                        for i in indexes[2:-1]:
                            hidden[i] = path + '/fold:' + children[indexes[0]][0]
            for i, (name, child) in enumerate(children):
                visit(child, path + '.' + name, depth + 1, i in hidden, hidden.get(i))
        visit(model, '<root>' + (str(len(self.roots)) if len(self.roots) > 1 else ''), 0)
        self.manager.state['module_selection'] = dict(selected=len(self.selected), hooks=len(self.handles),
            layers=self.config['layers'], folded=[dict(name=n, **attrs) for n, attrs in self.selected.values() if attrs.get('folded')])

    def install(self):
        if self.installed:
            return
        from torch.nn.modules.module import register_module_forward_pre_hook, register_module_forward_hook
        self.installed = True
        try:
            self.handles.append(register_module_forward_pre_hook(lambda m, a: self.safe(self.enter, m)))
            self.handles.append(register_module_forward_hook(lambda m, a, o: self.safe(self.leave, m), always_call=True))
        except BaseException:
            self.uninstall()
            raise

    def enter(self, model):
        state = self.context.probe_state()
        stack = state['modules']
        root = state['root'] is None and (id(model) in self.roots or not stack and not self.explicit)
        if root:
            self.register_model(model, explicit=False)
            start = self.config['each_start'] if self.config['dispatch'] == 'each-window' else -1
            self.context.begin_forward(start, self.config['each_count'] if start >= 0 else 0)
        item = dict(model=model, root=root, token=None, start=clock())
        # Skip children of folded layers while retaining the folded layer span.
        suppressed = any(x.get('folded') for x in stack)
        if state['root'] is not None and id(model) in self.selected and not suppressed:
            name, attrs = self.selected[id(model)]
            parent = self.context.current_scope()
            item.update(token=self.context.push_scope(name), name=name, attrs=dict(attrs, **(dict(phase=state.get('forward_phase', 'unknown')) if root else {})),
                        parent=parent.corr if parent else None, folded=attrs.get('folded', False))
        stack.append(item)

    def leave(self, model):
        state = self.context.probe_state()
        stack = state['modules']
        if not stack or stack[-1]['model'] is not model:
            raise RuntimeError('module callback ownership mismatch')
        item = stack.pop()
        try:
            if item['token']:
                self.context.pop_scope(item['token'])
                event = dict(t='span', ph='x', cat='module', name=item['name'], corr=item['token'].corr,
                             attrs=item['attrs'], ts=item['start'], dur=clock()-item['start'])
                if item['parent'] is not None:
                    event['parent'] = item['parent']
                self.context.writer.emit(event)
        finally:
            if item['root']:
                state['root'] = None

    def uninstall(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        super().uninstall()
