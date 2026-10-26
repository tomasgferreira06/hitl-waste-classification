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

Distribuição original Kaggle v1 (15 515 imagens) e dataset efetivo após curadoria humana (15 512):

| Classe | Original Kaggle | Efetivo |
| --- | ---: | ---: |
| battery | 945 | 945 |
| biological | 985 | 985 |
| brown-glass | 607 | 607 |
| cardboard | 891 | 891 |
| clothes | 5325 | 5325 |
| green-glass | 629 | 629 |
| metal | 769 | 768 |
| paper | 1050 | 1050 |
| plastic | 865 | 864 |
| shoes | 1977 | 1977 |
| trash | 697 | 697 |
| white-glass | 775 | 774 |
| **Total** | **15515** | **15512** |

Após inspeção visual humana de três grupos de duplicados exatos com rótulos contraditórios,
`data/manifests/excluded_images.csv` exclui as três cópias incorretamente rotuladas e
mantém as corretas (os rótulos Kaggle nunca são alterados):

| Excluída | Mantida |
| --- | --- |
| `plastic/plastic159.jpg` | `white-glass/white-glass129.jpg` |
| `metal/metal459.jpg` | `white-glass/white-glass412.jpg` |
| `white-glass/white-glass761.jpg` | `plastic/plastic556.jpg` |

`inventory.csv` continua a ser o inventário Kaggle completo (EDA); as experiências usam
`load_effective_inventory()`, `GarbageDataset` e os manifestos de divisão, todos sem as
exclusões. Os splits não foram refeitos: apenas se removeram os três ids de `train` e
`unlabeled` (treino 10856, validação 1552, teste 3104, pool inicial 200, unlabeled 10656).
Futuras experiências A e B, incluindo YOLO, devem partir de `load_effective_inventory()`
e nunca percorrer diretamente as pastas Kaggle.

Reiniciar o kernel depois desta alteração. Os arquivos de features guardam o SHA-256 de
`excluded_images.csv`; os três pontos de carregamento recusam arquivos sem essa revisão
ou com revisão diferente (por exemplo, o antigo `notebooks/garbage_features.npz`, que deve
ser regenerado, não reutilizado). Nenhum ficheiro foi apagado.

As pastas `data/` e `artifacts/` mantêm-se disponíveis; o notebook utiliza a
cache do Kaggle para as imagens e o caminho original para as features.

## Experiência B: validar YOLO26s Classification com CUDA

No Windows, `pyproject.toml` seleciona as builds CUDA 13.0 de PyTorch e
Torchvision pelo índice oficial `cu130`, conforme a
[integração uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
As versões exatas estão em `uv.lock`. O driver NVIDIA validado é o 591.74;
`nvidia-smi` indica suporte até CUDA 13.1, enquanto o runtime das builds é 13.0.
Não é necessário instalar o CUDA Toolkit completo para este teste.

Executar na raiz do projeto:

```powershell
nvidia-smi
uv sync --locked
uv run --locked python notebooks/validate_yolo_cuda.py
```

O script exige CUDA, testa uma operação na GPU e carrega os pesos oficiais
`yolo26s-cls.pt`, [pré-treinados em ImageNet](https://docs.ultralytics.com/tasks/classify/).
Mantém a cabeça original de 1000 classes e executa uma inferência a 224 × 224,
com `device=0`, numa imagem de `battery` da versão 1 do dataset Kaggle em cache.
Não executa treino ou extração de features, nem modifica a Experiência A.

O resultado mostra versões, GPU, número de parâmetros, SHA-256 dos pesos,
validação das probabilidades, tempo e memória CUDA. O tempo da primeira chamada
inclui preparação e warmup; o tempo `inference` é o reportado pelo Ultralytics.
A memória reportada pertence ao alocador PyTorch, não ao total de processos da GPU.
Uma inferência bem-sucedida não garante que um futuro fine-tuning caiba nos 6 GB.
As previsões são classes ImageNet e não medem desempenho nas 12 classes de resíduos.

Os pesos e a configuração local Ultralytics ficam em `artifacts/`, excluídos do
Git. O script desativa a gravação das previsões. Pesos `*.pt` e pastas `runs/`
também são ignorados; o dataset permanece na cache externa Kaggle.
A primeira execução requer Internet para os pesos e, se ausente, o dataset.

## EDA e protocolo comum às experiências A e B

O início do notebook contém tabelas, gráficos, auditoria de integridade/duplicados
e criação de manifestos. **Executar apenas essa secção; parar no separador da
implementação histórica.** As células originais dos modelos permanecem intactas
e ainda usam os seus splits antigos: não utilizar Run All nesta fase.

O [protocolo experimental](paper/experimental_protocol.md) documenta os resultados,
os problemas históricos e as decisões pendentes. Macro-F1 é a métrica principal;
AULC de Macro-F1 de validação mede eficiência de anotação. O teste fica reservado
à avaliação final, sem selecionar estratégias/configurações.

Os manifestos em `data/manifests/` são comuns às duas experiências: treino 10859,
validação 1552, teste 3104; pool inicial 200 e não rotulado 10659, exclusivamente
do treino. SHA-256 exatos e candidatos pHash são agrupados conservadoramente para
não atravessar splits. As três exclusões humanas estão em `excluded_images.csv`;
não foram modificadas imagens. Os 234 candidatos visuais existentes permanecem
como resultados históricos, sem recalcular ou apresentar a interface dos 55 pares.
As atribuições dos restantes IDs e o pool inicial não mudaram; as exclusões
removem só três imagens do treino não rotulado.

```powershell
uv run --locked python notebooks/dataset_protocol.py
uv run --locked python -m unittest discover -s notebooks -p test_dataset_protocol.py -v
```

O inventário completo é reservado à auditoria/oráculo; a vista do aprendiz obtém
apenas IDs opacos para o pool não rotulado. Hashing e leitura de imagens usam cache
em `artifacts/eda/`; apagar essa cache força nova auditoria. Gráficos e miniaturas
ficam nessa pasta ignorada pelo Git; não guardar outputs de miniaturas no notebook
versionado. Apenas metadados e identificadores, sem imagens, são versionáveis.

## Duplicados exatos após revisão humana

```powershell
uv run --locked jupyter lab notebooks/01_waste_classification_hitl.ipynb
```

Depois da análise de duplicados na EDA, executar **Human-reviewed exact duplicates
with conflicting original labels**: apresenta a tabela de exclusões e os 3 grupos
lado a lado, com ficheiro, classe Kaggle, grupo e indicação de excluída/mantida, a partir
dos dados originais (pré-exclusão). A distribuição original e a efetiva são apresentadas separadamente.
Foi removida toda a subsecção visual dos 55 pares, incluindo os seus outputs.

O comando `dataset_protocol.py` aplica as exclusões e atualiza as estatísticas,
de forma idempotente; não sorteia splits nem recalcula hashes ou candidatos.
Os ficheiros históricos em `data/review/` ficam preservados, anteriores à curadoria,
e não alteram o dataset efetivo. Não há uma nova interface de revisão.

```powershell
uv run --locked python -m unittest discover -s notebooks -p 'test_*.py' -v
```
