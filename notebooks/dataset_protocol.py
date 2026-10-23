"""EDA e manifestos comuns às experiências; não importa nem treina modelos."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import kagglehub
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / 'data' / 'manifests'
DATASET_HANDLE = 'mostafaabla/garbage-classification/versions/1'
SEED = 42
EXPECTED_COUNTS = {
    'battery': 945, 'biological': 985, 'brown-glass': 607, 'cardboard': 891,
    'clothes': 5325, 'green-glass': 629, 'metal': 769, 'paper': 1050,
    'plastic': 865, 'shoes': 1977, 'trash': 697, 'white-glass': 775,
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PHASH_DISTANCE = 4


def dataset_root():
    return Path(kagglehub.dataset_download(DATASET_HANDLE)) / 'garbage_classification'


def resolve_image(root, relative_path):
    """Os manifestos usam POSIX; rejeitar caminhos absolutos ou que saiam da raiz."""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or '..' in relative.parts or '\\' in relative_path or ':' in relative_path:
        raise ValueError(f'Caminho relativo inválido: {relative_path}')
    path = Path(root).joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'Caminho fora do dataset: {relative_path}')
    return path


def scan_dataset(root, cache_path=PROJECT_ROOT / 'artifacts' / 'eda' / 'image_cache.json'):
    """Reutiliza hashes/decodificação se o caminho, tamanho e mtime_ns não mudaram."""
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if set(classes) != set(EXPECTED_COUNTS):
        raise ValueError(f'Classes inesperadas: {classes}')
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    if cache.get('schema') != 'sha256-phash-dct64-v1' or cache.get('dataset') != DATASET_HANDLE:
        cache = {}
    previous = cache.get('files', {})
    updated, records = {}, []
    files = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: p.relative_to(root).as_posix())
    for index, path in enumerate(files):
        relative = path.relative_to(root).as_posix()
        if len(PurePosixPath(relative).parts) != 2:
            raise ValueError(f'Estrutura inesperada: {relative}')
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        old = previous.get(relative)
        if old and old['signature'] == signature:
            record = old['record']
        else:
            with path.open('rb') as stream:
                sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            record = dict(image_id=hashlib.sha256(relative.encode()).hexdigest(), relative_path=relative,
                          label=path.parent.name, bytes=stat.st_size, sha256=sha256,
                          width='', height='', aspect_ratio='', format='', mode='', phash='', error='')
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    record.update(width=image.width, height=image.height,
                                  aspect_ratio=image.width / image.height, format=image.format, mode=image.mode)
                    gray = ImageOps.exif_transpose(image).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
                    coefficients = dctn(np.asarray(gray, dtype=float), type=2, norm='ortho')[:8, :8].ravel()
                    bits = coefficients > np.median(coefficients[1:])
                    record['phash'] = f'{sum(int(bit) << i for i, bit in enumerate(bits)):016x}'
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                record['error'] = f'{type(error).__name__}: {error}'
        updated[relative] = dict(signature=signature, record=record)
        records.append(record)
        if (index + 1) % 1000 == 0:
            print(f'EDA: {index + 1}/{len(files)} imagens', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(dict(schema='sha256-phash-dct64-v1', dataset=DATASET_HANDLE,
                                         files=updated), ensure_ascii=False), encoding='utf-8')
    return records


def exact_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['sha256']].append(record)
    return [group for _, group in sorted(groups.items()) if len(group) > 1]


def visual_candidates(records):
    """pHash 64 bits, distância <=4; cinco blocos evitam testar todos os pares."""
    buckets, pairs = defaultdict(list), []
    for index, record in enumerate(records):
        if not record['phash']:
            continue
        value = int(record['phash'], 16)
        keys, offset = [], 0
        for block, width in enumerate((13, 13, 13, 13, 12)):
            keys.append((block, (value >> offset) & ((1 << width) - 1)))
            offset += width
        candidates = {other for key in keys for other in buckets[key]}
        for other in sorted(candidates):
            previous = records[other]
            if previous['sha256'] == record['sha256']:
                continue
            distance = (value ^ int(previous['phash'], 16)).bit_count()
            if distance <= PHASH_DISTANCE:
                pairs.append(dict(image_id_a=previous['image_id'], image_id_b=record['image_id'],
                                  distance=distance, cross_class=previous['label'] != record['label']))
        for key in keys:
            buckets[key].append(index)
    return pairs


def split_groups(records, candidates):
    """Componentes conexas exatas + candidatos visuais, sem confirmar estes últimos."""
    parent = {record['image_id']: record['image_id'] for record in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for group in exact_groups(records):
        for record in group[1:]:
            union(group[0]['image_id'], record['image_id'])
    for pair in candidates:
        union(pair['image_id_a'], pair['image_id_b'])
    return {key: find(key) for key in parent}


def make_partitions(records, candidates, seed=SEED):
    if any(record['error'] for record in records):
        raise ValueError('Há imagens ilegíveis; rever a EDA antes de criar divisões.')
    records = sorted(records, key=lambda record: record['relative_path'])
    group_ids = split_groups(records, candidates)
    labels = [record['label'] for record in records]
    groups = [group_ids[record['image_id']] for record in records]
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    partitions = dict(train=[], validation=[], test=[])
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(records)), labels, groups)):
        name = 'test' if fold < 2 else 'validation' if fold == 2 else 'train'
        partitions[name].extend(records[index]['image_id'] for index in indices)
    for ids in partitions.values():
        ids.sort()

    # Começar em grupos singleton evita aliases já conhecidos no unlabeled pool.
    sizes = defaultdict(int)
    for group in groups:
        sizes[group] += 1
    train_ids = set(partitions['train'])
    counts = {label: sum(record['image_id'] in train_ids and record['label'] == label for record in records)
              for label in sorted(set(labels))}
    quotas = {label: 200 * count / len(train_ids) for label, count in counts.items()}
    allocated = {label: int(quota) for label, quota in quotas.items()}
    remainder = 200 - sum(allocated.values())
    for label in sorted(quotas, key=lambda key: (-(quotas[key] - allocated[key]), key))[:remainder]:
        allocated[label] += 1
    rng = np.random.default_rng(seed)
    initial = []
    for label, count in allocated.items():
        eligible = [record['image_id'] for record in records if record['image_id'] in train_ids
                    and record['label'] == label and sizes[group_ids[record['image_id']]] == 1]
        if count < 1 or len(eligible) < count:
            raise ValueError(f'Não é possível selecionar o pool inicial estratificado: {label}')
        initial.extend(rng.choice(eligible, size=count, replace=False).tolist())
    partitions['labeled_initial'] = sorted(initial)
    partitions['unlabeled'] = sorted(train_ids - set(initial))
    return partitions, group_ids


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_manifests(records, candidates, directory=MANIFEST_DIR):
    directory = Path(directory)
    partitions, groups = make_partitions(records, candidates)
    inventory = [dict(record, group_id=groups[record['image_id']]) for record in records]
    write_csv(directory / 'inventory.csv', inventory, list(inventory[0]))
    write_csv(directory / 'visual_candidates.csv', candidates,
              ['image_id_a', 'image_id_b', 'distance', 'cross_class'])
    lookup = {record['image_id']: record for record in records}
    for name, ids in partitions.items():
        rows = [dict(image_id=image_id) for image_id in ids]
        if name == 'labeled_initial':
            rows = [dict(row, label=lookup[row['image_id']]['label'], label_source='initial_simulation_oracle')
                    for row in rows]
        write_csv(directory / f'{name}.csv', rows, list(rows[0]))
    metadata = dict(dataset=DATASET_HANDLE, seed=SEED, split='StratifiedGroupKFold(10): test folds 0,1; validation 2; train 3..9',
                    grouping='connected components: SHA-256 exact + conservative pHash candidates',
                    phash='DCT-II ortho 32x32 grayscale (EXIF transposed), 8x8 coefficients, median excluding DC',
                    phash_hamming_threshold=PHASH_DISTANCE, initial_size=200,
                    initial_selection='class-proportional largest remainder; singleton groups; numpy default_rng(42)',
                    counts={name: len(ids) for name, ids in partitions.items()},
                    manifest_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(directory.glob('*.csv'))})
    (directory / 'protocol.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return partitions


def load_pool_views(directory=MANIFEST_DIR):
    """A vista do aprendiz não recebe a tabela de ground truth nem caminhos com classes."""
    directory = Path(directory)
    labeled = read_csv(directory / 'labeled_initial.csv')
    unlabeled = tuple(row['image_id'] for row in read_csv(directory / 'unlabeled.csv'))
    return labeled, unlabeled


def load_pool_image(root, image_id, directory=MANIFEST_DIR):
    """Fronteira de I/O: devolver pixels RGB, sem rótulo ou caminho na vista do aprendiz."""
    records = read_csv(Path(directory) / 'inventory.csv')
    record = next((record for record in records if record['image_id'] == image_id), None)
    if record is None:
        raise KeyError(f'Identificador desconhecido: {image_id}')
    allowed = {row['image_id'] for row in read_csv(Path(directory) / 'train.csv')}
    if image_id not in allowed:
        raise ValueError('Acesso a imagem fora do treino através da vista Active Learning.')
    with Image.open(resolve_image(root, record['relative_path'])) as image:
        rgb = image.convert('RGB')
        rgb.info.clear()
        return rgb


if __name__ == '__main__':
    records = scan_dataset(dataset_root())
    candidates = visual_candidates(records)
    partitions = write_manifests(records, candidates)
    print(json.dumps(dict(images=len(records), exact_groups=len(exact_groups(records)),
                          visual_pairs=len(candidates), splits={key: len(value) for key, value in partitions.items()}), indent=2))
