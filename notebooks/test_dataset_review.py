"""Testes da fila e do histórico; todas as decisões de teste ficam em diretórios temporários."""

from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

import dataset_protocol as protocol
import dataset_review as review


class DatasetReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = review.build_registry()

    def temporary_review(self, directory):
        with patch.object(review, 'build_registry', return_value=self.registry):
            return review.prepare_review(directory)

    def test_complete_prioritized_queue_and_metadata(self):
        cases = self.registry['cases']
        self.assertEqual(len(cases), 252)
        self.assertEqual(Counter(case['priority'] for case in cases), {1: 3, 2: 55, 3: 15, 4: 179})
        self.assertEqual(len({case['case_id'] for case in cases}), 252)
        self.assertEqual([c['priority'] for c in cases], sorted(c['priority'] for c in cases))
        root = protocol.dataset_root()
        for case in cases:
            self.assertGreaterEqual(len(case['images']), 2)
            self.assertEqual(len({image['split'] for image in case['images']}), 1)
            self.assertEqual(case['cross_class'], len({image['original_label'] for image in case['images']}) > 1)
            for image in case['images']:
                self.assertEqual(image['original_label'], image['relative_path'].split('/')[0])
                self.assertTrue(protocol.resolve_image(root, image['relative_path']).is_file())
            if case['match_type'] == 'exact':
                self.assertIsNone(case['perceptual_distance'])
                self.assertEqual(len({image['sha256'] for image in case['images']}), 1)
            else:
                self.assertIn(case['perceptual_distance'], range(5))
                self.assertEqual(len(case['images']), 2)
        self.assertTrue(all(row['status'] == 'pending' for row in review.case_rows(self.registry, [])))

    def test_case_ids_are_independent_of_input_order(self):
        with patch.object(review, 'read_csv', side_effect=lambda path: list(reversed(protocol.read_csv(path)))):
            reordered = review.build_registry()
        self.assertEqual(self.registry, reordered)

    def test_preparation_preserves_human_history_and_refuses_changed_snapshot(self):
        with TemporaryDirectory() as directory:
            self.temporary_review(directory)
            review.record_decision(self.registry['cases'][0]['case_id'], 'ambiguous', 'Necessita de revisão adicional.', 'Teste', directory)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            self.temporary_review(directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})
            changed = deepcopy(self.registry)
            changed['cases'] = changed['cases'][1:]
            with patch.object(review, 'build_registry', return_value=changed), self.assertRaisesRegex(ValueError, 'fila existente'):
                review.prepare_review(directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})

    def test_decisions_are_append_only_with_auditable_revisions(self):
        with TemporaryDirectory() as directory:
            self.temporary_review(directory)
            case_id = self.registry['cases'][0]['case_id']
            first = review.record_decision(case_id, 'ambiguous', 'Não é possível decidir o rótulo.', 'Revisor A', directory)
            original = (Path(directory) / 'decisions.jsonl').read_bytes()
            second = review.record_decision(case_id, 'incorrect_label', 'Proposta a rever; não aplicar.', 'Revisor B', directory)
            self.assertTrue((Path(directory) / 'decisions.jsonl').read_bytes().startswith(original))
            self.assertEqual(second['previous_event_id'], first['event_id'])
            self.assertIsNotNone(datetime.fromisoformat(first['reviewed_at_utc']).tzinfo)
            self.assertEqual(review.read_decisions(self.registry, directory), [first, second])
            self.assertEqual(review.case_rows(self.registry, [first, second])[0]['status'], 'incorrect_label')

    def test_invalid_decisions_do_not_write_and_corrupt_history_fails(self):
        with TemporaryDirectory() as directory:
            self.temporary_review(directory)
            case_id = self.registry['cases'][0]['case_id']
            for args in [('unknown', 'pending', 'reason', 'reviewer'), (case_id, 'unknown', 'reason', 'reviewer'),
                         (case_id, 'pending', ' ', 'reviewer'), (case_id, 'pending', 'reason', ' ')]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    review.record_decision(*args, directory=directory)
            log = Path(directory) / 'decisions.jsonl'
            self.assertEqual(log.read_bytes(), b'')
            log.write_text('invalid json\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'linha 1'):
                review.read_decisions(self.registry, directory)
            with self.assertRaises(ValueError):
                review.record_decision(case_id, 'pending', 'reason', 'reviewer', directory)
            self.assertEqual(log.read_text(encoding='utf-8'), 'invalid json\n')

    def test_previews_are_read_only_and_validate_content(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'class' / 'sample.png'
            image_path.parent.mkdir()
            Image.new('RGB', (100, 200), 'red').save(image_path)
            original = image_path.read_bytes()
            member = dict(relative_path='class/sample.png', sha256=hashlib.sha256(original).hexdigest())
            preview = review.preview_image(directory, member)
            with Image.open(BytesIO(preview)) as image:
                self.assertEqual(image.size, (100, 200))
            self.assertEqual(original, image_path.read_bytes())
            image_path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'diferente da EDA'):
                review.preview_image(directory, member)

    def test_widget_filters_renders_and_saves_only_on_explicit_click(self):
        with TemporaryDirectory() as directory:
            self.temporary_review(directory)
            with patch.object(review, 'prepare_review', return_value=self.registry):
                widget = review.review_widget(protocol.dataset_root(), directory)
            priority, pending = widget.children[0].children
            selector = widget.children[1]
            self.assertEqual(len(selector.options), 252)
            self.assertEqual((Path(directory) / 'decisions.jsonl').read_bytes(), b'')
            priority.value = 1
            self.assertEqual(len(selector.options), 3)
            self.assertEqual(len(widget.children[2].children[1].children), 2)
            case_id = selector.value
            widget.children[3].value = 'Revisor de teste'
            widget.children[4].value = 'ambiguous'
            widget.children[5].value = 'Interação simulada apenas no teste.'
            widget.children[6].click()
            self.assertEqual(len(selector.options), 2)
            self.assertEqual(review.read_decisions(self.registry, directory)[0]['case_id'], case_id)
            pending.value = False
            self.assertEqual(len(selector.options), 3)
            widget.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
