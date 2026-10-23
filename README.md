# hitl-waste-classification
Human-in-the-Loop Active Learning for Waste Classification

PracticalLab 1 da unidade curricular Trustworthy and Responsible Artificial Intelligence (TRAI).

## Estrutura

- `notebooks/01_waste_classification_hitl.ipynb`: notebook com download automático do dataset; implementação científica original mantida.
- `data/`: dados do projeto.
- `artifacts/`: resultados e modelos.
- `paper/`: material para o relatório.

## Ambiente Python 3.12

Com o [uv](https://docs.astral.sh/uv/) instalado, executar na raiz do repositório:

```powershell
uv sync --locked
uv run --locked jupyter lab
```

Abrir `notebooks/01_waste_classification_hitl.ipynb` e selecionar o kernel do ambiente `.venv`.

## Dataset Kaggle

O notebook descarrega automaticamente [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/versions/1)
através de `kagglehub`. A versão efetivamente descarregada e fixada é **1**:

```python
path = Path(kagglehub.dataset_download("mostafaabla/garbage-classification/versions/1"))
DATASET_PATH = path / "garbage_classification"
```

O diretório devolvido contém `garbage_classification/`; as 12 pastas das classes
encontram-se dentro desse subdiretório. O notebook verifica esta estrutura antes
de criar `GarbageDataset`.

Os ficheiros são utilizados diretamente na cache nativa (`~/.cache/kagglehub/`
por omissão), fora do repositório. As execuções seguintes reutilizam a versão
em cache, sem `force_download` ou cópias para `data/`. O identificador inclui
`/versions/1` para evitar alterações silenciosas por novas versões no Kaggle,
conforme a [documentação do kagglehub](https://github.com/Kaggle/kagglehub#download-dataset).
A primeira execução requer acesso à Internet. Não foram necessárias credenciais
para descarregar este dataset público na validação.

As duas secções HITL resolvem os caminhos guardados nas features para a cache
da versão fixa, incluindo caminhos antigos de Windows ou POSIX. Um ficheiro
inexistente produz um erro explícito. Executar primeiro as células de imports
e de configuração do dataset, mesmo ao reutilizar features existentes.
`garbage_features.npz` continua relativo ao diretório de trabalho do kernel;
o seu conteúdo não é recalculado nem validado por esta integração.

### Validação sem treino

```powershell
uv run --locked python notebooks/validate_dataset.py
```

Este comando valida a estrutura do notebook, a reutilização da cache, a contagem
de imagens, a leitura RGB de uma imagem por classe com `GarbageDataset` e a
leitura de imagens nas duas interfaces HITL. Não extrai features nem treina modelos.

Resultado observado na versão 1:

| Classe | Imagens |
| --- | ---: |
| battery | 945 |
| biological | 985 |
| brown-glass | 607 |
| cardboard | 891 |
| clothes | 5325 |
| green-glass | 629 |
| metal | 769 |
| paper | 1050 |
| plastic | 865 |
| shoes | 1977 |
| trash | 697 |
| white-glass | 775 |
| **Total** | **15515** |

As pastas `data/` e `artifacts/` mantêm-se disponíveis; o notebook utiliza a
cache do Kaggle para as imagens e o caminho original para as features.
