"""Validar apenas o dataset e os caminhos HITL: uv run --locked python notebooks/validate_dataset.py."""

import ast
from collections import Counter
import json
from pathlib import Path
from unittest.mock import Mock, patch

import kagglehub
import nbformat
import ipywidgets as widgets


notebook_path = Path(__file__).with_name('01_waste_classification_hitl.ipynb')
notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
nbformat.validate(nbformat.from_dict(notebook))
sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
namespace = {}
exec(sources[0], namespace)
exec(next(source for source in sources if 'class GarbageDataset(' in source), namespace)
root = namespace['DATASET_PATH']
assert not root.is_relative_to(notebook_path.resolve().parents[1])

# A segunda chamada deve reutilizar os mesmos ficheiros na cache.
snapshot = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
assert Path(kagglehub.dataset_download(namespace['DATASET_HANDLE'])) == namespace['path']
assert snapshot == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in snapshot}

dataset = namespace['GarbageDataset'](root)
expected_counts = {
    'battery': 945, 'biological': 985, 'brown-glass': 607, 'cardboard': 891,
    'clothes': 5325, 'green-glass': 629, 'metal': 768, 'paper': 1050,
    'plastic': 864, 'shoes': 1977, 'trash': 697, 'white-glass': 774,
}
assert set(dataset.class_names) == namespace['EXPECTED_CLASSES']
counts = Counter(dataset.labels)
actual_counts = {name: counts[index] for index, name in enumerate(dataset.class_names)}
assert actual_counts == expected_counts, actual_counts
assert len(dataset) == 15512
for name, count in actual_counts.items():
    print(f'{name}: {count}')
    image, label, filename = dataset[dataset.labels.index(dataset.class_to_idx[name])]
    assert image.mode == 'RGB' and min(image.size) > 0
    assert dataset.class_names[label] == name
    folder = Path(filename).parent.name
    for stored_path in (filename, f'./garbage_classification/{folder}/{Path(filename).name}',
                        f'C:\\old-cache\\{folder}\\{Path(filename).name}'):
        assert namespace['fix_path'](stored_path) == filename
print(f'Total: {len(dataset)}; leitura RGB validada nas 12 classes.')
try:
    namespace['fix_path']('battery/missing-image.jpg')
except FileNotFoundError:
    pass
else:
    raise AssertionError('Um caminho inexistente deve produzir FileNotFoundError.')

# Executar apenas as atribuicoes de caminhos das duas secoes HITL.
namespace['data'] = {'filenames': [f'C:\\old\\battery\\{Path(dataset.image_paths[0]).name}']}
for source in sources:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'filenames' for t in node.targets):
            if "data['filenames']" in ast.unparse(node):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<HITL paths>', 'exec'), namespace)
                assert namespace['filenames'] == [dataset.image_paths[0]]

# Verificar a leitura nas duas interfaces sem instanciar/treinar modelos.
for source in sources:
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ('HITLLabeler', 'least_confidence_query', 'update_display'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<HITL>', 'exec'), namespace)
namespace.update(current_sample_idx=0, current_probs=namespace['np'].array([1.0] + [0.0] * 11),
                 class_names=dataset.class_names, user_labeled_idx=[], user_unlabeled_idx=[0],
                 img_widget=Mock(), info_label=Mock(), display=Mock(), clear_output=Mock())
namespace['update_display']()
assert namespace['img_widget'].value == Path(dataset.image_paths[0]).read_bytes()
labeler = namespace['HITLLabeler'].__new__(namespace['HITLLabeler'])
labeler.__dict__.update(X=namespace['np'].zeros((1, 1)), unlabeled_idx=[0], labeled_idx=[],
                        filenames=namespace['filenames'], out=widgets.Output(), model=Mock())
labeler.model.predict_proba.return_value = namespace['np'].array([[1.0] + [0.0] * 11])
with patch.object(namespace['Image'], 'open', wraps=namespace['Image'].open) as image_open:
    labeler.next_iteration()
    image_open.assert_called_once_with(dataset.image_paths[0])
print('OK: cache reutilizada, caminhos Windows/POSIX e ambas as interfaces HITL; sem treino.')
