"""Verificar a exclusão de três duplicados mal rotulados sem alterar rótulos, splits ou imagens."""

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest

import nbformat
import numpy as np
from PIL import Image

import dataset_protocol as protocol


EXCLUDED = {
    'plastic/plastic159.jpg': 'white-glass/white-glass129.jpg',
    'metal/metal459.jpg': 'white-glass/white-glass412.jpg',
    'white-glass/white-glass761.jpg': 'plastic/plastic556.jpg',
}
EXPECTED_EFFECTIVE = {
    'battery': 945, 'biological': 985, 'brown-glass': 607, 'cardboard': 891, 'clothes': 5325, 'green-glass': 629,
    'metal': 768, 'paper': 1050, 'plastic': 864, 'shoes': 1977, 'trash': 697, 'white-glass': 774,
}
# Hashes das divisões congeladas, antes das exclusões; reconstruídas somando as 3 imagens às listas atuais.
PRE_EXCLUSION_HASHES = {
    'train': 'a7970980ef45714c66930340abbfe2d74eeca8ad2329526838d072b30fd0f67a',
    'validation': '14524a2b08c32ac38378f3e683a1ed7a6312014fd6f7ad0b0135ed510cc47bd8',
    'test': '4ea7f723beaa98a1ec280719a856ebc035c26dd689ff92d268ebd3a21bf1d047',
    'labeled_initial': '36d604783cb77517c6395b8a83932725beaa91486ea9c004986363ae625f689d',
    'unlabeled': 'f75a6f75115fc503c25613822030a82585f6bbc2e617b2cbcf2129439135723c',
}
ORIGINAL_INVENTORY_SHA256 = '9c315012922a9e6b9b6ea326dbc5d1215305c3c578b2e4b978a12670267aeff2'


class ExcludedImagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = protocol.MANIFEST_DIR
        cls.original = protocol.load_original_inventory()
        cls.effective = protocol.load_effective_inventory()
        cls.exclusions = protocol.load_exclusions()
        cls.notebook = nbformat.read(protocol.PROJECT_ROOT / 'notebooks/01_waste_classification_hitl.ipynb', as_version=4)
        cls.by_path = {r['relative_path']: r for r in cls.original}

    def test_exactly_three_authorized_exclusions_with_provenance(self):
        self.assertEqual({k: v['kept_relative_path'] for k, v in self.exclusions.items()}, EXCLUDED)
        for relative, row in self.exclusions.items():
            self.assertEqual(row['original_label'], relative.split('/')[0])
            self.assertTrue(row['reason'] and row['provenance'] == 'revisão visual humana')
            self.assertEqual(self.by_path[relative]['sha256'], self.by_path[row['kept_relative_path']]['sha256'])
        self.assertEqual(protocol.excluded_paths(), frozenset(EXCLUDED))

    def test_counts_labels_and_kept_copies(self):
        self.assertEqual(len(self.original), 15515)
        self.assertEqual(len(self.effective), 15512)
        self.assertEqual(Counter(r['label'] for r in self.effective), EXPECTED_EFFECTIVE)
        effective_paths = {r['relative_path'] for r in self.effective}
        self.assertFalse(effective_paths & set(EXCLUDED))
        self.assertTrue(set(EXCLUDED.values()) <= effective_paths)
        # Rótulos preservados: sempre a pasta Kaggle original, sem coluna/política de correção.
        self.assertTrue(all(r['label'] == r['relative_path'].split('/')[0] for r in self.original))
        self.assertNotIn('original_label', self.original[0])
        self.assertEqual(hashlib.sha256((self.directory / 'inventory.csv').read_bytes()).hexdigest(), ORIGINAL_INVENTORY_SHA256)
        self.assertEqual(protocol.load_effective_inventory(), self.effective)  # determinístico

    def test_splits_pool_and_metadata(self):
        excluded_ids = {self.by_path[p]['image_id'] for p in EXCLUDED}
        parts = protocol.read_partitions()
        for name, ids in parts.items():
            self.assertFalse(excluded_ids & set(ids), name)
            self.assertEqual(ids, sorted(ids))
        # Os 3 ids estavam em train e unlabeled: somá-los reconstitui as listas congeladas byte a byte.
        for name in ('train', 'unlabeled'):
            restored = 'image_id\n' + ''.join(f'{key}\n' for key in sorted(set(parts[name]) | excluded_ids))
            self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(), PRE_EXCLUSION_HASHES[name], name)
        for name in ('validation', 'test', 'labeled_initial'):
            self.assertEqual(hashlib.sha256((self.directory / f'{name}.csv').read_bytes()).hexdigest(),
                             PRE_EXCLUSION_HASHES[name], name)
        self.assertEqual({n: len(i) for n, i in parts.items()},
                         dict(train=10856, validation=1552, test=3104, labeled_initial=200, unlabeled=10656))
        self.assertEqual(set(parts['train']), set(parts['labeled_initial']) | set(parts['unlabeled']))
        lookup = {r['image_id']: r for r in self.effective}
        self.assertTrue(all(row['label'] == lookup[row['image_id']]['label']
                            for row in protocol.read_csv(self.directory / 'labeled_initial.csv')))
        self.assertEqual(sum(len(parts[n]) for n in ('train', 'validation', 'test')), 15512)
        metadata = json.loads((self.directory / 'protocol.json').read_text(encoding='utf-8'))
        self.assertEqual((metadata['original_dataset_size'], metadata['effective_dataset_size'], metadata['excluded_count']),
                         (15515, 15512, 3))
        self.assertEqual(metadata['effective_class_counts'], EXPECTED_EFFECTIVE)
        self.assertEqual(metadata['excluded_images_sha256'], protocol.excluded_images_revision())
        self.assertEqual(metadata['pre_exclusion_manifest_sha256']['train.csv'], PRE_EXCLUSION_HASHES['train'])
        self.assertNotIn('label_corrections_sha256', metadata)

    def test_refresh_is_idempotent_and_does_not_resplit(self):
        with TemporaryDirectory() as directory:
            for source in self.directory.iterdir():
                if source.is_file():
                    copyfile(source, Path(directory) / source.name)
            protocol.refresh_effective_dataset(directory)
            for source in self.directory.iterdir():
                if source.is_file():
                    self.assertEqual(source.read_bytes(), (Path(directory) / source.name).read_bytes(), source.name)

    def test_pool_view_and_image_loader_reject_excluded(self):
        labeled, unlabeled = protocol.load_pool_views()
        excluded_ids = {self.by_path[p]['image_id'] for p in EXCLUDED}
        self.assertFalse(excluded_ids & (set(unlabeled) | {row['image_id'] for row in labeled}))
        with self.assertRaises(KeyError):
            protocol.load_pool_image(protocol.dataset_root(), next(iter(excluded_ids)))

    def test_original_kaggle_images_are_intact(self):
        root = protocol.dataset_root()
        for record in self.original:
            with protocol.resolve_image(root, record['relative_path']).open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), record['sha256'], record['relative_path'])
        self.assertEqual(sum(1 for p in root.rglob('*') if p.is_file()), 15515)

    def test_garbage_dataset_skips_exclusions_and_keeps_folder_labels(self):
        from torch.utils.data import Dataset

        source = next(cell.source for cell in self.notebook.cells if 'class GarbageDataset(' in cell.source)
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'GarbageDataset')
        namespace = dict(Dataset=Dataset, Path=Path, Image=Image, os=os, excluded_paths=protocol.excluded_paths)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<GarbageDataset>', 'exec'), namespace)
        root = protocol.dataset_root()
        dataset = namespace['GarbageDataset'](root)
        relatives = {Path(p).relative_to(root).as_posix() for p in dataset.image_paths}
        self.assertEqual(len(dataset), 15512)
        self.assertEqual(relatives, {r['relative_path'] for r in self.effective})
        for filename, label in zip(dataset.image_paths, dataset.labels):
            self.assertEqual(dataset.class_names[label], Path(filename).parent.name)
        self.assertEqual(Counter(dataset.class_names[i] for i in dataset.labels), EXPECTED_EFFECTIVE)

    def test_stale_feature_archives_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'features.npz'
            np.savez_compressed(path, X=np.zeros((3, 2)))
            with self.assertRaisesRegex(ValueError, 'desatualizado'):
                protocol.load_feature_archive(path)
            np.savez_compressed(path, X=np.zeros((3, 2)), label_corrections_sha256='old', excluded_images_sha256='old')
            with self.assertRaisesRegex(ValueError, 'desatualizado'):
                protocol.load_feature_archive(path)
            np.savez_compressed(path, X=np.zeros((3, 2)), excluded_images_sha256=protocol.excluded_images_revision())
            with protocol.load_feature_archive(path) as archive:
                self.assertEqual(archive['X'].shape, (3, 2))

    def test_no_active_label_correction_policy(self):
        self.assertFalse((self.directory / 'label_corrections.csv').exists())
        project = protocol.PROJECT_ROOT
        for relative in ('notebooks/dataset_protocol.py', 'notebooks/duplicate_viewer.py', 'notebooks/validate_dataset.py',
                         'notebooks/dataset_review.py'):
            text = (project / relative).read_text(encoding='utf-8')
            for forbidden in ('load_label_corrections', 'effective_label', 'label_corrections', 'corrected_label'):
                self.assertNotIn(forbidden, text, f'{relative}: {forbidden}')
        for cell in self.notebook.cells:
            if cell.cell_type == 'code':
                for forbidden in ('label_corrections', 'effective_label', 'corrected_label'):
                    self.assertNotIn(forbidden, cell.source)
        nbformat.validate(self.notebook)
        source = '\n'.join(cell.source for cell in self.notebook.cells)
        self.assertEqual(source.count("load_feature_archive('garbage_features.npz')"), 4)
        self.assertNotIn("np.load('garbage_features.npz')", source)
        self.assertIn('excluded_images_sha256=excluded_images_revision()', source)
        self.assertNotIn('visual_candidates_view', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
