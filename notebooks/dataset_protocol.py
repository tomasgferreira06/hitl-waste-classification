"""EDA e manifestos comuns às experiências; não importa nem treina modelos."""

from collections import Counter, defaultdict
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
EXCLUSIONS_FILE = MANIFEST_DIR / 'excluded_images.csv'
EXCLUSION_COLUMNS = ['relative_path', 'original_label', 'reason', 'kept_relative_path', 'provenance']
PARTITIONS = ('train', 'validation', 'test', 'labeled_initial', 'unlabeled')


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


def load_exclusions(path=EXCLUSIONS_FILE):
    """Imagens excluídas do dataset efetivo; os ficheiros Kaggle nunca são alterados."""
    exclusions = {}
    for row in read_csv(path):
        relative = PurePosixPath(row['relative_path'])
        kept = PurePosixPath(row['kept_relative_path'])
        if (list(row) != EXCLUSION_COLUMNS or len(relative.parts) != 2 or len(kept.parts) != 2
                or relative.parts[0] != row['original_label'] or row['original_label'] not in EXPECTED_COUNTS
                or any(c in row['relative_path'] + row['kept_relative_path'] for c in '\\:')
                or '..' in relative.parts + kept.parts
                or not row['reason'].strip() or not row['provenance'].strip()):
            raise ValueError(f'Exclusão inválida: {row}')
        if row['relative_path'] in exclusions:
            raise ValueError(f'Exclusão repetida: {relative}')
        exclusions[row['relative_path']] = row
    return exclusions


def excluded_paths(path=EXCLUSIONS_FILE):
    return frozenset(load_exclusions(path))


def excluded_images_revision(path=EXCLUSIONS_FILE):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original_inventory(directory=MANIFEST_DIR):
    """Dataset Kaggle completo (15515), só para EDA/documentação; nunca para experiências."""
    return read_csv(Path(directory) / 'inventory.csv')


def load_effective_inventory(directory=MANIFEST_DIR):
    directory = Path(directory)
    exclusions = load_exclusions(directory / 'excluded_images.csv')
    records = load_original_inventory(directory)
    by_path = {record['relative_path']: record for record in records}
    for relative, row in exclusions.items():
        kept = by_path.get(row['kept_relative_path'])
        if (relative not in by_path or by_path[relative]['label'] != row['original_label'] or kept is None
                or kept['relative_path'] in exclusions or kept['sha256'] != by_path[relative]['sha256']):
            raise ValueError(f'Exclusão inconsistente com o inventário: {relative}')
    return [record for record in records if record['relative_path'] not in exclusions]


def read_partitions(directory=MANIFEST_DIR):
    return {name: [row['image_id'] for row in read_csv(Path(directory) / f'{name}.csv')] for name in PARTITIONS}


def load_feature_archive(path, exclusions_path=EXCLUSIONS_FILE):
    archive = np.load(path, allow_pickle=False)
    if ('excluded_images_sha256' not in archive.files
            or str(archive['excluded_images_sha256'].item()) != excluded_images_revision(exclusions_path)):
        archive.close()
        raise ValueError(f'Artefacto de features desatualizado: {path}. Regenerar com as exclusões atuais; não reutilizar silenciosamente.')
    return archive


def refresh_effective_dataset(directory=MANIFEST_DIR):
    """Remover as exclusões das divisões/pools existentes; nunca recalcular grupos, pools ou splits."""
    directory = Path(directory)
    metadata_path = directory / 'protocol.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    original = load_original_inventory(directory)
    records = load_effective_inventory(directory)
    excluded_ids = {row['image_id'] for row in original} - {row['image_id'] for row in records}
    lookup = {record['image_id']: record for record in records}
    # Fixar uma única vez os hashes anteriores às exclusões (rastreabilidade das divisões congeladas).
    metadata.setdefault('pre_exclusion_manifest_sha256', metadata.pop('manifest_sha256'))
    for name in PARTITIONS:
        rows = read_csv(directory / f'{name}.csv')
        if name == 'labeled_initial' and any(row['image_id'] in excluded_ids for row in rows):
            raise ValueError('Uma exclusão pertence ao pool inicial; requer substituição estratificada documentada.')
        write_csv(directory / f'{name}.csv', [row for row in rows if row['image_id'] not in excluded_ids], list(rows[0]))
    partitions = read_partitions(directory)
    if any(row['label'] != lookup[row['image_id']]['label'] for row in read_csv(directory / 'labeled_initial.csv')):
        raise ValueError('Rótulo do pool inicial diferente do rótulo original Kaggle.')
    metadata.pop('split_membership_sha256', None)
    metadata.update(
        label_semantics='label = original Kaggle label (never changed); excluded_images.csv removes human-reviewed mislabeled duplicates',
        excluded_images_sha256=excluded_images_revision(directory / 'excluded_images.csv'),
        original_dataset_size=len(original), effective_dataset_size=len(records), excluded_count=len(excluded_ids),
        original_class_counts=dict(sorted(Counter(r['label'] for r in original).items())),
        effective_class_counts=dict(sorted(Counter(r['label'] for r in records).items())),
        effective_split_class_counts={name: dict(sorted(Counter(lookup[key]['label'] for key in ids).items()))
                                      for name, ids in partitions.items()},
        counts={name: len(ids) for name, ids in partitions.items()},
        manifest_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob('*.csv'))})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    return partitions


def write_manifests(records, candidates, directory=MANIFEST_DIR):
    directory = Path(directory)
    if (directory / 'protocol.json').exists():
        return refresh_effective_dataset(directory)
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
    (directory / 'protocol.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    return partitions


def load_pool_views(directory=MANIFEST_DIR):
    """A vista do aprendiz não recebe a tabela de ground truth nem caminhos com classes."""
    directory = Path(directory)
    labeled = read_csv(directory / 'labeled_initial.csv')
    lookup = {record['image_id']: record['label'] for record in load_effective_inventory(directory)}
    labeled = [dict(row, label=lookup[row['image_id']]) for row in labeled]
    unlabeled = tuple(row['image_id'] for row in read_csv(directory / 'unlabeled.csv'))
    return labeled, unlabeled


def load_pool_image(root, image_id, directory=MANIFEST_DIR):
    """Fronteira de I/O: devolver pixels RGB, sem rótulo ou caminho na vista do aprendiz."""
    records = load_effective_inventory(directory)
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
    partitions = refresh_effective_dataset()
    print(json.dumps(dict(effective_counts=Counter(record['label'] for record in load_effective_inventory()),
                          splits={key: len(value) for key, value in partitions.items()}), indent=2))


def feature_rows(filenames, image_ids):
    """Linhas do arquivo de features para cada image_id, alinhadas por ID (sha256 de 'classe/ficheiro'), nunca por posição."""
    row_of = {hashlib.sha256('/'.join(str(f).replace('\\', '/').split('/')[-2:]).encode()).hexdigest(): i
              for i, f in enumerate(filenames)}
    missing = [i for i in image_ids if i not in row_of]
    if missing:
        raise ValueError(f'{len(missing)} IDs dos manifestos sem features (ex.: {missing[0]}); regenerar features.')
    return np.array([row_of[i] for i in image_ids])
