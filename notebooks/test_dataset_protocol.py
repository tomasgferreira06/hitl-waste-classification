"""uv run --locked python -m unittest discover -s notebooks -p test_dataset_protocol.py -v"""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

import dataset_protocol as protocol


class DatasetProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = protocol.MANIFEST_DIR
        cls.records = protocol.read_csv(cls.directory / 'inventory.csv')
        cls.lookup = {record['image_id']: record for record in cls.records}
        cls.candidates = protocol.read_csv(cls.directory / 'visual_candidates.csv')
        cls.parts = {name: [row['image_id'] for row in protocol.read_csv(cls.directory / f'{name}.csv')]
                     for name in ('train', 'validation', 'test', 'labeled_initial', 'unlabeled')}

    def test_inventory_and_paths(self):
        self.assertEqual(len(self.records), 15515)
        self.assertEqual(len(self.lookup), len(self.records))
        self.assertEqual(Counter(record['label'] for record in self.records), protocol.EXPECTED_COUNTS)
        self.assertFalse([record for record in self.records if record['error']])
        root = protocol.dataset_root()
        for record in self.records:
            relative = PurePosixPath(record['relative_path'])
            self.assertEqual(relative.parts[0], record['label'])
            self.assertEqual(len(relative.parts), 2)
            self.assertTrue(protocol.resolve_image(root, relative.as_posix()).is_file())
            self.assertEqual(PureWindowsPath('C:/dataset').joinpath(*relative.parts).parts[-2:], relative.parts)
            self.assertEqual(PurePosixPath('/dataset').joinpath(*relative.parts).parts[-2:], relative.parts)
        for path in ('/etc/passwd', '../image.jpg', 'a/../../image.jpg', 'C:/image.jpg', 'a\\image.jpg'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                protocol.resolve_image(root, path)

    def test_manifest_checksums_and_content_hash_samples(self):
        metadata = json.loads((self.directory / 'protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(metadata['dataset'], protocol.DATASET_HANDLE)
        self.assertEqual(metadata['seed'], 42)
        for filename, checksum in metadata['manifest_sha256'].items():
            self.assertEqual(hashlib.sha256((self.directory / filename).read_bytes()).hexdigest(), checksum, filename)
        root = protocol.dataset_root()
        for label in protocol.EXPECTED_COUNTS:
            record = next(record for record in self.records if record['label'] == label)
            with protocol.resolve_image(root, record['relative_path']).open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), record['sha256'])

    def test_exclusive_exhaustive_partitions_and_classes(self):
        membership = {}
        for name in ('train', 'validation', 'test'):
            ids = self.parts[name]
            self.assertEqual(len(ids), len(set(ids)), name)
            self.assertEqual({self.lookup[key]['label'] for key in ids}, set(protocol.EXPECTED_COUNTS), name)
            for key in ids:
                self.assertNotIn(key, membership, f'{key} pertence a dois conjuntos')
                membership[key] = name
        self.assertEqual(set(membership), set(self.lookup))

    def test_exact_and_candidate_groups_never_cross_splits(self):
        hashes, groups = {}, {}
        membership = {key: name for name in ('train', 'validation', 'test') for key in self.parts[name]}
        for key, name in membership.items():
            record = self.lookup[key]
            self.assertEqual(hashes.setdefault(record['sha256'], name), name, record['relative_path'])
            self.assertEqual(groups.setdefault(record['group_id'], name), name, record['relative_path'])
        for pair in self.candidates:
            self.assertEqual(membership[pair['image_id_a']], membership[pair['image_id_b']])

    def test_stratification_tolerances(self):
        for name, target in (('train', .7), ('validation', .1), ('test', .2)):
            ids = self.parts[name]
            self.assertLessEqual(abs(len(ids) / len(self.records) - target), .01, name)
            counts = Counter(self.lookup[key]['label'] for key in ids)
            for label, total in protocol.EXPECTED_COUNTS.items():
                self.assertLessEqual(abs(counts[label] / total - target), .02, f'{name}: {label}')

    def test_initial_pool_and_hidden_unlabeled_truth(self):
        initial, unlabeled = map(set, (self.parts['labeled_initial'], self.parts['unlabeled']))
        self.assertEqual(len(initial), 200)
        self.assertFalse(initial & unlabeled)
        self.assertEqual(initial | unlabeled, set(self.parts['train']))
        self.assertFalse((initial | unlabeled) & set(self.parts['validation'] + self.parts['test']))
        counts = Counter(self.lookup[key]['label'] for key in initial)
        self.assertEqual(set(counts), set(protocol.EXPECTED_COUNTS))
        train_counts = Counter(self.lookup[key]['label'] for key in self.parts['train'])
        for label, total in train_counts.items():
            self.assertLessEqual(abs(counts[label] - 200 * total / len(self.parts['train'])), 1)
        sizes = Counter(record['group_id'] for record in self.records)
        self.assertTrue(all(sizes[self.lookup[key]['group_id']] == 1 for key in initial))
        labeled_view, unlabeled_view = protocol.load_pool_views()
        self.assertEqual(set(unlabeled_view), unlabeled)
        self.assertTrue(all(set(row) == {'image_id', 'label', 'label_source'} for row in labeled_view))
        self.assertEqual({row['label_source'] for row in labeled_view}, {'initial_simulation_oracle'})
        self.assertTrue(all(len(key) == 64 and '/' not in key for key in unlabeled_view))
        self.assertTrue(all(set(row) == {'image_id'} for row in protocol.read_csv(self.directory / 'unlabeled.csv')))
        root = protocol.dataset_root()
        image = protocol.load_pool_image(root, unlabeled_view[0])
        self.assertEqual(image.mode, 'RGB')
        self.assertFalse(image.info)
        self.assertFalse(getattr(image, 'filename', ''))
        with self.assertRaises(ValueError):
            protocol.load_pool_image(root, self.parts['validation'][0])
        with self.assertRaises(ValueError):
            protocol.load_pool_image(root, self.parts['test'][0])

    def test_reproducibility_including_manifest_bytes(self):
        with TemporaryDirectory() as directory:
            reproduced = protocol.write_manifests(self.records, self.candidates, directory)
            self.assertEqual(reproduced, self.parts)
            for source in self.directory.glob('*.csv'):
                self.assertEqual(source.read_bytes(), (Path(directory) / source.name).read_bytes(), source.name)
            self.assertEqual((self.directory / 'protocol.json').read_bytes(), (Path(directory) / 'protocol.json').read_bytes())
        reordered, _ = protocol.make_partitions(list(reversed(self.records)), list(reversed(self.candidates)))
        self.assertEqual(reordered, self.parts)

    def test_visual_index_matches_bruteforce_on_small_fixture(self):
        values = [0, 1, 15, 31, 1 << 63, (1 << 64) - 1]
        records = [dict(image_id=str(i), label='a' if i % 2 else 'b', sha256=str(i), phash=f'{value:016x}')
                   for i, value in enumerate(values)]
        expected = {(str(i), str(j)) for i in range(len(values)) for j in range(i + 1, len(values))
                    if (values[i] ^ values[j]).bit_count() <= 4}
        actual = {(pair['image_id_a'], pair['image_id_b']) for pair in protocol.visual_candidates(records)}
        self.assertEqual(actual, expected)
        records[1]['sha256'] = records[0]['sha256']
        self.assertNotIn(('0', '1'), {(p['image_id_a'], p['image_id_b']) for p in protocol.visual_candidates(records)})

    def test_transitive_groups_include_cross_class_exact_duplicates(self):
        records = [dict(image_id='a', sha256='same', label='one'),
                   dict(image_id='b', sha256='same', label='two'), dict(image_id='c', sha256='different', label='two')]
        groups = protocol.split_groups(records, [dict(image_id_a='b', image_id_b='c')])
        self.assertEqual(set(groups.values()), {'a'})

    def test_corruption_is_reported_and_cache_is_invalidated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            for label in protocol.EXPECTED_COUNTS:
                path = root / label / 'sample.png'
                path.parent.mkdir(parents=True)
                Image.new('RGB', (10, 20), 'red').save(path)
            cache = Path(directory) / 'cache.json'
            first = protocol.scan_dataset(root, cache)
            target = root / 'battery' / 'sample.png'
            Image.new('RGB', (30, 40), 'blue').save(target)
            changed = protocol.scan_dataset(root, cache)
            self.assertNotEqual(first[0]['sha256'], changed[0]['sha256'])
            self.assertEqual(changed[0]['width'], 30)
            target.write_bytes(b'not an image')
            broken = protocol.scan_dataset(root, cache)
            self.assertTrue(broken[0]['error'])
            with self.assertRaisesRegex(ValueError, 'ilegíveis'):
                protocol.make_partitions(broken, [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
