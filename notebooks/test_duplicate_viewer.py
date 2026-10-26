"""Validar os três grupos originais e a indicação de excluída/mantida."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import ipywidgets as widgets
import nbformat
from PIL import Image

import dataset_protocol as protocol
import dataset_review as review
import duplicate_viewer as viewer


class DuplicateViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = protocol.dataset_root()
        cls.cases = viewer.load_reviewed_exact_cases()
        cls.inventory = {r['image_id']: r for r in protocol.read_csv(protocol.MANIFEST_DIR / 'inventory.csv')}

    def check_panel(self, panel, case):
        heading, images = panel.children
        self.assertIn(case['case_id'], heading.value)
        self.assertEqual(len(images.children), len(case['images']))
        for card, member in zip(images.children, case['images']):
            metadata, image = card.children
            self.assertIn(member['original_label'], metadata.value)
            self.assertIn(member['relative_path'], metadata.value)
            self.assertEqual(member['original_label'], self.inventory[member['image_id']]['label'])
            self.assertIn(Path(member['relative_path']).name, metadata.value)
            self.assertEqual(member['status'] == 'excluded', member['relative_path'] in protocol.excluded_paths())
            self.assertIn('EXCLUÍDA' if member['status'] == 'excluded' else 'MANTIDA', metadata.value)
            self.assertEqual(image.width, '440')
            self.assertIsInstance(image, widgets.Image)
            with Image.open(BytesIO(bytes(image.value))) as decoded:
                self.assertEqual(decoded.format, 'PNG')
                self.assertGreater(min(decoded.size), 0)
            self.assertEqual(hashlib.sha256(protocol.resolve_image(self.root, member['relative_path']).read_bytes()).hexdigest(),
                             member['sha256'])

    def test_only_all_requested_cross_class_cases_are_loaded(self):
        self.assertEqual({kind: len(cases) for kind, cases in self.cases.items()}, {'exact': 3})
        expected = json.loads((review.REVIEW_DIR / 'cases.json').read_text(encoding='utf-8'))['cases']
        self.assertEqual({c['case_id'] for cases in self.cases.values() for c in cases},
                         {c['case_id'] for c in expected if c['cross_class'] and c['match_type'] == 'exact'})
        for case in self.cases['exact']:
            self.assertEqual(len(case['images']), 2)
            self.assertEqual(sorted(m['status'] for m in case['images']), ['excluded', 'kept'])
            self.assertEqual(len({m['original_label'] for m in case['images']}), 2)
        self.assertEqual({m['relative_path'] for c in self.cases['exact'] for m in c['images'] if m['status'] == 'excluded'},
                         set(protocol.excluded_paths()))

    def test_all_exact_groups_and_all_members_are_displayed(self):
        widget = viewer.exact_conflicts_view(self.cases, self.root)
        self.assertEqual(len(widget.children), 3)
        for panel, case in zip(widget.children, self.cases['exact']):
            self.check_panel(panel, case)
        group = deepcopy(self.cases['exact'][0])
        group['images'].append(deepcopy(group['images'][0]))
        self.check_panel(viewer.case_panel(group, self.root), group)
        widget.close()

    def test_visual_interface_is_removed(self):
        notebook = nbformat.read(protocol.PROJECT_ROOT / 'notebooks/01_waste_classification_hitl.ipynb', as_version=4)
        self.assertNotIn('eda-visual-conflicts', {c.id for c in notebook.cells})
        self.assertNotIn('eda-visual-conflicts-method', {c.id for c in notebook.cells})
        self.assertNotIn('widgets', notebook.metadata)
        source = '\n'.join(cell.source for cell in notebook.cells)
        self.assertNotIn('Visual duplicate candidates across different classes', source)
        self.assertNotIn('visual_candidates_view', source)
        self.assertFalse(hasattr(viewer, 'visual_candidates_view'))

    def test_notebook_cells_run_independently_without_writes_or_eda(self):
        protected = list(protocol.MANIFEST_DIR.iterdir()) + list(review.REVIEW_DIR.iterdir())
        before = {p: p.read_bytes() for p in protected if p.is_file()}
        notebook = nbformat.read(protocol.PROJECT_ROOT / 'notebooks/01_waste_classification_hitl.ipynb', as_version=4)
        nbformat.validate(notebook)
        namespace = {}
        with patch('IPython.display.display') as display, \
                patch.object(protocol, 'scan_dataset', side_effect=AssertionError('Não recalcular EDA')), \
                patch.object(protocol, 'write_manifests', side_effect=AssertionError('Não escrever manifestos')), \
                patch.object(review, 'prepare_review', side_effect=AssertionError('Não preparar revisão')), \
                patch.object(review, 'record_decision', side_effect=AssertionError('Não anotar')):
            for cell in notebook.cells:
                if cell.id in ('eda-exact-conflicts', 'eda-label-comparison'):
                    exec(compile(cell.source, '<notebook viewer>', 'exec'), namespace)
            self.assertEqual(display.call_count, 3)
        self.assertEqual(before, {p: p.read_bytes() for p in protected if p.is_file()})
        for name in ('eda_exact_conflicts_ui',):
            namespace[name].close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
