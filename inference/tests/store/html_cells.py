# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Independent report cell reader for export regression checks."""
from html.parser import HTMLParser


class Cells(HTMLParser):
    """Map both legacy vertical tables and current horizontal row tables."""
    def __init__(self):
        super().__init__();self.table=None;self.cell=None;self.values={};self.rows={};self.stack=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='table':
            self.stack.append(self.table)
            if 'data-panel' in attrs:self.table=(attrs['data-panel'],int(attrs.get('data-row',0)))
        if tag=='td' and 'data-panel' in attrs and 'data-row' in attrs:
            self.table=(attrs['data-panel'],int(attrs['data-row']))
        if tag=='td' and self.table and 'data-field' in attrs:
            panel,row=self.table;self.rows[panel]=max(self.rows.get(panel,0),row+1)
            self.cell=(*self.table,attrs['data-field']);self.values[self.cell]=[attrs.get('data-unit'),[]]
    def handle_endtag(self,tag):
        if tag=='td':self.cell=None
        if tag=='table':self.table=self.stack.pop()
    def handle_data(self,value):
        if self.cell:self.values[self.cell][1].append(value)
