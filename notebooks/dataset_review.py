"""Fila de revisão humana sobre os manifestos existentes, sem recalcular a EDA."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from dataset_protocol import DATASET_HANDLE, MANIFEST_DIR, PROJECT_ROOT, dataset_root, exact_groups, read_csv, resolve_image


REVIEW_DIR = PROJECT_ROOT / 'data' / 'review'
DECISIONS = {
    'pending': 'Pendente de revisão',
    'confirmed_duplicate': 'Duplicado confirmado',
    'different_images': 'Imagens diferentes',
    'incorrect_label': 'Rótulo incorreto',
    'ambiguous': 'Caso ambíguo',
}


def build_registry(manifests=MANIFEST_DIR):
    manifests = Path(manifests)
    protocol = json.loads((manifests / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['dataset'] != DATASET_HANDLE:
        raise ValueError('Versão do dataset incompatível com a revisão.')
    for filename, expected in protocol['manifest_sha256'].items():
        if hashlib.sha256((manifests / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Manifesto alterado: {filename}')
    records = read_csv(manifests / 'inventory.csv')
    lookup = {record['image_id']: record for record in records}
    # Os grupos nunca atravessam splits; assim, uma imagem excluída herda o split do seu grupo.
    group_split = {lookup[row['image_id']]['group_id']: name for name in ('train', 'validation', 'test')
                   for row in read_csv(manifests / f'{name}.csv')}
    splits = {record['image_id']: group_split[record['group_id']] for record in records}
    cases = []

    def add_case(kind, members, distance=None):
        ids = sorted(record['image_id'] for record in members)
        key = json.dumps([DATASET_HANDLE, kind, ids], separators=(',', ':'))
        cross_class = len({record['label'] for record in members}) > 1
        priority = (1 if cross_class else 3) if kind == 'exact' else (2 if cross_class else 4)
        cases.append(dict(case_id=f'{kind}-{hashlib.sha256(key.encode()).hexdigest()}',
                          priority=priority, match_type=kind, perceptual_distance=distance,
                          cross_class=cross_class, images=[
                              dict(image_id=record['image_id'], relative_path=record['relative_path'],
                                   original_label=record['label'], sha256=record['sha256'],
                                   split=splits[record['image_id']])
                              for record in sorted(members, key=lambda r: r['relative_path'])]))

    for group in exact_groups(records):
        add_case('exact', group)
    for pair in read_csv(manifests / 'visual_candidates.csv'):
        add_case('visual', [lookup[pair['image_id_a']], lookup[pair['image_id_b']]], int(pair['distance']))
    cases.sort(key=lambda case: (case['priority'], case['perceptual_distance'] or 0, case['case_id']))
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('A fila contém casos repetidos.')
    return dict(schema=1, dataset=DATASET_HANDLE, default_status='pending',
                source_manifest_sha256=protocol['manifest_sha256'], cases=cases)


def prepare_review(directory=REVIEW_DIR, manifests=MANIFEST_DIR):
    """Criar uma vez; uma nova execução nunca sobrescreve a fila ou decisões."""
    directory = Path(directory)
    registry = build_registry(manifests)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'cases.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != registry:
            raise ValueError('A fila existente difere dos manifestos. Preservar a revisão e criar uma nova versão explicitamente.')
    else:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(registry, ensure_ascii=False, indent=2) + '\n')
    log = directory / 'decisions.jsonl'
    if not log.exists():
        log.touch(exist_ok=False)
    read_decisions(registry, directory)
    return registry


def read_decisions(registry, directory=REVIEW_DIR):
    cases = {case['case_id'] for case in registry['cases']}
    events, seen, previous = [], set(), {}
    with (Path(directory) / 'decisions.jsonl').open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                if event['case_id'] not in cases or event['decision'] not in DECISIONS:
                    raise ValueError('Caso ou decisão desconhecidos.')
                if not event['justification'].strip() or not event['reviewer'].strip():
                    raise ValueError('Justificação/revisor em falta.')
                reviewed_at = datetime.fromisoformat(event['reviewed_at_utc'])
                if reviewed_at.utcoffset() != timezone.utc.utcoffset(reviewed_at):
                    raise ValueError('A data deve incluir timezone UTC.')
                if event['event_id'] in seen or event['previous_event_id'] != previous.get(event['case_id']):
                    raise ValueError('Histórico duplicado ou cadeia de decisões inválida.')
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f'Histórico inválido na linha {line_number}: {error}') from error
            seen.add(event['event_id'])
            previous[event['case_id']] = event['event_id']
            events.append(event)
    return events


def record_decision(case_id, decision, justification, reviewer, directory=REVIEW_DIR):
    directory = Path(directory)
    registry = json.loads((directory / 'cases.json').read_text(encoding='utf-8'))
    if case_id not in {case['case_id'] for case in registry['cases']}:
        raise ValueError('Identificador de caso desconhecido.')
    if decision not in DECISIONS:
        raise ValueError('Decisão desconhecida.')
    if not justification.strip() or not reviewer.strip():
        raise ValueError('Preencher revisor e justificação antes de guardar.')
    events = read_decisions(registry, directory)
    previous = next((event['event_id'] for event in reversed(events) if event['case_id'] == case_id), None)
    event = dict(event_id=str(uuid4()), case_id=case_id, decision=decision,
                 justification=justification.strip(), reviewer=reviewer.strip(),
                 reviewed_at_utc=datetime.now(timezone.utc).isoformat(), previous_event_id=previous)
    with (directory / 'decisions.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    return event


def case_rows(registry, events):
    latest = {event['case_id']: event for event in events}
    return [dict(case_id=case['case_id'], priority=case['priority'], match_type=case['match_type'],
                 classes=' | '.join(image['original_label'] for image in case['images']),
                 paths=' | '.join(image['relative_path'] for image in case['images']),
                 splits=' | '.join(image['split'] for image in case['images']),
                 perceptual_distance=case['perceptual_distance'],
                 status=latest.get(case['case_id'], {}).get('decision', 'pending'))
            for case in registry['cases']]


def preview_image(root, member):
    """Validar só os ficheiros abertos, sem repetir hashing de todo o dataset."""
    content = resolve_image(root, member['relative_path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != member['sha256']:
        raise ValueError(f'Imagem diferente da EDA: {member["relative_path"]}')
    with Image.open(BytesIO(content)) as image:
        preview = ImageOps.exif_transpose(image).convert('RGB')
        preview.thumbnail((640, 640))
        output = BytesIO()
        preview.save(output, format='PNG')
        return output.getvalue()


def review_widget(root=None, directory=REVIEW_DIR):
    import ipywidgets as widgets

    registry = prepare_review(directory)
    root = dataset_root() if root is None else Path(root)
    cases = {case['case_id']: case for case in registry['cases']}
    priority = widgets.Dropdown(options=[('Todas as prioridades', 0)] + [(f'Prioridade {n}', n) for n in range(1, 5)])
    pending_only = widgets.Checkbox(value=True, description='Apenas pendentes')
    selector = widgets.Dropdown(description='Caso', layout=widgets.Layout(width='100%'))
    panel = widgets.VBox()
    decision = widgets.Dropdown(options=[(label, key) for key, label in DECISIONS.items()], description='Decisão',
                                layout=widgets.Layout(width='100%'))
    reviewer = widgets.Text(description='Revisor', layout=widgets.Layout(width='100%'))
    justification = widgets.Textarea(description='Justificação', layout=widgets.Layout(width='100%', height='90px'))
    save = widgets.Button(description='Acrescentar decisão', button_style='primary')
    message = widgets.HTML()

    def render(change=None):
        save.disabled = selector.value is None
        justification.value = ''
        if selector.value is None:
            panel.children = (widgets.HTML('Nenhum caso neste filtro.'),)
            return
        case = cases[selector.value]
        history = [event for event in read_decisions(registry, directory) if event['case_id'] == case['case_id']]
        status = history[-1]['decision'] if history else 'pending'
        decision.value = status
        heading = widgets.HTML(f'<b>P{case["priority"]} · {escape(case["match_type"])} · '
                               f'{escape(DECISIONS[status])}</b><br><code>{case["case_id"]}</code><br>'
                               f'Distância pHash: {case["perceptual_distance"] if case["perceptual_distance"] is not None else "não aplicável (SHA-256)"}')
        cards = []
        try:
            for member in case['images']:
                cards.append(widgets.VBox([
                    widgets.Image(value=preview_image(root, member), format='png', width=300),
                    widgets.HTML(f'Classe original: <b>{escape(member["original_label"])}</b><br>'
                                 f'<code>{escape(member["relative_path"])}</code><br>Split: <b>{escape(member["split"])}</b>'),
                ], layout=widgets.Layout(width='320px')))
        except (OSError, ValueError) as error:
            save.disabled = True
            panel.children = (heading, widgets.HTML(f'<b>Não foi possível validar as imagens:</b> {escape(str(error))}'))
            return
        audit = '<br>'.join(f'{escape(e["reviewed_at_utc"])} — {escape(e["reviewer"])} — '
                           f'{escape(DECISIONS[e["decision"]])}: {escape(e["justification"])}' for e in history)
        panel.children = (heading, widgets.HBox(cards, layout=widgets.Layout(flex_flow='row wrap')),
                          widgets.HTML('<b>Histórico:</b><br>' + (audit or 'Sem decisões humanas.')))

    def refresh(change=None):
        rows = case_rows(registry, read_decisions(registry, directory))
        selected = selector.value
        options = [(f'P{row["priority"]} · {row["match_type"]} · {row["classes"]} · {row["case_id"][:22]}', row['case_id'])
                   for row in rows if (not priority.value or row['priority'] == priority.value)
                   and (not pending_only.value or row['status'] == 'pending')]
        selector.options = options
        selector.value = selected if selected in {key for _, key in options} else (options[0][1] if options else None)
        render()

    def append(change):
        try:
            event = record_decision(selector.value, decision.value, justification.value, reviewer.value, directory)
            refresh()
            message.value = f'Decisão acrescentada ao histórico: <code>{event["event_id"]}</code>.'
        except (OSError, ValueError) as error:
            message.value = f'<b>Não guardado:</b> {escape(str(error))}'

    selector.observe(render, names='value')
    priority.observe(refresh, names='value')
    pending_only.observe(refresh, names='value')
    save.on_click(append)
    refresh()
    return widgets.VBox([widgets.HBox([priority, pending_only]), selector, panel, reviewer, decision, justification, save, message])


if __name__ == '__main__':
    registry = prepare_review()
    rows = case_rows(registry, read_decisions(registry))
    print(json.dumps(dict(cases=len(rows), priorities=dict(sorted(Counter(row['priority'] for row in rows).items())),
                          statuses=dict(Counter(row['status'] for row in rows))), ensure_ascii=False, indent=2))
