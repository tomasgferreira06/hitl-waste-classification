"""Três grupos exatos com rótulos Kaggle originais e a exclusão decidida por revisão humana."""

import hashlib
from html import escape
import json
from pathlib import Path

import ipywidgets as widgets

from dataset_protocol import DATASET_HANDLE, dataset_root, exact_groups, load_exclusions, load_original_inventory, read_partitions
from dataset_review import preview_image


def load_reviewed_exact_cases():
    """Usa o inventário Kaggle original (pré-exclusão), só para análise e documentação."""
    records = load_original_inventory()
    exclusions = load_exclusions()
    splits = {key: name for name, ids in read_partitions().items()
              if name in ('train', 'validation', 'test') for key in ids}
    cases = []
    for group in exact_groups(records):
        if len({record['label'] for record in group}) < 2:
            continue
        ids = sorted(record['image_id'] for record in group)
        key = json.dumps([DATASET_HANDLE, 'exact', ids], separators=(',', ':'))
        cases.append(dict(case_id=f'exact-{hashlib.sha256(key.encode()).hexdigest()}', images=[
            dict(image_id=record['image_id'], relative_path=record['relative_path'], original_label=record['label'],
                 status='excluded' if record['relative_path'] in exclusions else 'kept',
                 sha256=record['sha256'], split=splits.get(record['image_id'], 'excluída'))
            for record in sorted(group, key=lambda row: row['relative_path'])]))
    return {'exact': sorted(cases, key=lambda case: case['case_id'])}


def case_panel(case, root):
    heading = widgets.HTML(
        f'<div style="overflow-wrap:anywhere"><b>Identificador:</b> <code>{escape(case["case_id"])}</code>'
        '<br><b>Tipo:</b> duplicado exato SHA-256 com rótulos contraditórios, revisto por anotador humano</div>')
    cards = []
    for member in case['images']:
        cards.append(widgets.VBox([
            widgets.HTML(f'<b style="font-size:18px;color:{"#c0392b" if member["status"] == "excluded" else "#1e8449"}">'
                         f'{"EXCLUÍDA" if member["status"] == "excluded" else "MANTIDA"}</b><br>'
                         f'<b style="font-size:18px">Classe original Kaggle: {escape(member["original_label"])}</b><br>'
                         f'<b>Ficheiro:</b> {escape(Path(member["relative_path"]).name)}<br>'
                         f'<code>{escape(member["relative_path"])}</code><br>Split: {escape(member["split"])}'),
            widgets.Image(value=preview_image(root, member), format='png', width=440),
        ], layout=widgets.Layout(min_width='460px', width='460px')))
    images = widgets.HBox(cards, layout=widgets.Layout(flex_flow='row nowrap', overflow='auto', width='100%'))
    return widgets.VBox([heading, images])


def exact_conflicts_view(cases, root=None):
    root = dataset_root() if root is None else Path(root)
    return widgets.VBox([case_panel(case, root) for case in cases['exact']])


