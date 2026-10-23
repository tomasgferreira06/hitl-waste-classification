"""Smoke test CUDA/YOLO26s Classification, sem treino ou alteracoes ao notebook."""

import hashlib
import os
from pathlib import Path
import platform
from time import perf_counter

import kagglehub
import torch
import torchvision


artifacts = Path(__file__).resolve().parents[1] / 'artifacts'
artifacts.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(artifacts / 'ultralytics')

import ultralytics
from ultralytics import YOLO


print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}; Torchvision: {torchvision.__version__}')
print(f'Ultralytics: {ultralytics.__version__}; CUDA runtime: {torch.version.cuda}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA indisponivel. Executar uv sync --locked e verificar o driver NVIDIA.')

print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print(f'VRAM total: {torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB')
assert (torch.ones(3, device='cuda:0') * 2).tolist() == [2.0, 2.0, 2.0]

weights = artifacts / 'yolo26s-cls.pt'
model = YOLO(str(weights), task='classify')
assert model.task == 'classify' and len(model.names) == 1000
print(f'Arquitetura: {type(model.model).__name__}; classes ImageNet: {len(model.names)}')
print(f'Parametros do checkpoint: {sum(p.numel() for p in model.model.parameters()):,}')
with weights.open('rb') as stream:
    print(f'Pesos SHA-256: {hashlib.file_digest(stream, "sha256").hexdigest()}')
model.to('cuda:0')
assert next(model.model.parameters()).device == torch.device('cuda:0')

dataset = Path(kagglehub.dataset_download('mostafaabla/garbage-classification/versions/1'))
class_dir = dataset / 'garbage_classification' / 'battery'
image_path = next(iter(sorted(p for p in class_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})), None)
if image_path is None:
    raise FileNotFoundError(f'Nenhuma imagem em {class_dir}')
print(f'Imagem de smoke test: {image_path}')

torch.cuda.synchronize(0)
torch.cuda.reset_peak_memory_stats(0)
started = perf_counter()
with torch.inference_mode():
    results = model.predict(source=str(image_path), device=0, imgsz=224,
                            save=False, save_txt=False, save_crop=False, verbose=False)
torch.cuda.synchronize(0)
elapsed_ms = (perf_counter() - started) * 1000

assert len(results) == 1 and results[0].probs is not None
assert model.predictor.device == torch.device('cuda:0')
probs = results[0].probs.data
assert probs.is_cuda and probs.shape == (1000,)
assert torch.isfinite(probs).all().item()
assert ((probs >= 0) & (probs <= 1)).all().item()
assert torch.isclose(probs.sum(), probs.new_tensor(1.0), atol=1e-4).item()
print(f'Probabilidades validas: {probs.numel()}; soma: {probs.sum().item():.6f}; device: {probs.device}')
print(f'Inferencia Ultralytics: {results[0].speed["inference"]:.2f} ms')
print(f'Primeira chamada predict, incluindo preparacao/warmup: {elapsed_ms:.2f} ms')
print(f'CUDA alocada: {torch.cuda.memory_allocated(0) / 2**20:.2f} MiB; '
      f'pico alocado: {torch.cuda.max_memory_allocated(0) / 2**20:.2f} MiB; '
      f'pico reservado: {torch.cuda.max_memory_reserved(0) / 2**20:.2f} MiB')
print('OK: inferencia em device=0 sem CUDA OOM. Classes ImageNet; sem avaliacao das 12 classes de residuos.')
