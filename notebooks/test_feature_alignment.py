"""Alinhamento features <-> manifestos e métrica Weighted F1."""

import hashlib
import unittest

import numpy as np
from sklearn.metrics import f1_score

import dataset_protocol as protocol


def _id(relative):
    return hashlib.sha256(relative.encode()).hexdigest()


class FeatureAlignmentTests(unittest.TestCase):
    def test_rows_follow_ids_not_order(self):
        names = [r'C:\x\metal\m1.jpg', '/y/paper/p1.jpg', 'plastic/pl1.jpg']
        rows = protocol.feature_rows(names, [_id('plastic/pl1.jpg'), _id('metal/m1.jpg')])
        self.assertEqual(rows.tolist(), [2, 0])

    def test_missing_id_raises(self):
        with self.assertRaises(ValueError):
            protocol.feature_rows(['metal/m1.jpg'], [_id('paper/p1.jpg')])

    def test_manifest_ids_map_to_effective_inventory(self):
        records = protocol.load_effective_inventory()
        names = [r['relative_path'] for r in records]
        parts = protocol.read_partitions()
        rows = protocol.feature_rows(names, parts['train'])
        self.assertEqual(len(rows), 10856)
        self.assertTrue(all(_id(names[r]) in set(parts['train']) for r in rows[:50]))

    def test_macro_f1_ignores_class_frequency(self):
        y, p = [0, 0, 0, 1], [0, 0, 0, 0]
        self.assertAlmostEqual(f1_score(y, p, average='macro', zero_division=0), (6 / 7 + 0) / 2, places=6)
        self.assertGreater(f1_score(y, p, average='weighted', zero_division=0), f1_score(y, p, average='macro', zero_division=0))


if __name__ == '__main__':
    unittest.main()
