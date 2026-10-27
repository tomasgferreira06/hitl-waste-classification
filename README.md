# hitl-waste-classification
Human-in-the-Loop Active Learning for Waste Classification

PracticalLab 1 da unidade curricular Trustworthy and Responsible Artificial Intelligence (TRAI).

## Estrutura

- `notebooks/01_waste_classification_hitl.ipynb`: notebook único com **todo o código Python do projeto**
  (dataset, EDA, protocolo, features, Active Learning e HITL). Não há módulos `.py` locais.
- `notebooks/garbage_features.npz`: features MobileNetV2 pré-extraídas.
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
inexistente produz um erro explícito. As features ficam em `notebooks/garbage_features.npz`
(caminho resolvido a partir da raiz do projeto); só são extraídas se o ficheiro não existir
ou com `REGENERATE_FEATURES = True`.

### Execução

O notebook executa-se sequencialmente (*Restart & Run All*). As etapas demoradas ou que
escrevem artefactos estão protegidas por flags na secção 2 (`REGENERATE_FEATURES`,
`REFRESH_MANIFESTS`, `RUN_EXPERIMENT_A`).

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

`inventory.csv` continua a ser o inventário Kaggle completo (EDA); o pipeline usa
`load_effective_inventory()`, `GarbageDataset` e os manifestos de divisão, todos sem as
exclusões. Os splits não foram refeitos: apenas se removeram os três ids de `train` e
`unlabeled` (treino 10856, validação 1552, teste 3104, pool inicial 200, unlabeled 10656).
O pipeline parte de `load_effective_inventory()`
e nunca percorrer diretamente as pastas Kaggle.

Reiniciar o kernel depois desta alteração. Os arquivos de features guardam o SHA-256 de
`excluded_images.csv`; os três pontos de carregamento recusam arquivos sem essa revisão
ou com revisão diferente (por exemplo, o antigo `notebooks/garbage_features.npz`, que deve
ser regenerado, não reutilizado). Nenhum ficheiro foi apagado.

As pastas `data/` e `artifacts/` mantêm-se disponíveis; o notebook utiliza a
cache do Kaggle para as imagens e o caminho original para as features.

## Ambiente (PyTorch CUDA)

No Windows, `pyproject.toml` seleciona as builds CUDA 13.0 de PyTorch e
Torchvision (MobileNetV2) pelo índice oficial `cu130`, conforme a
[integração uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
As versões exatas estão em `uv.lock`.

## EDA e protocolo experimental

As secções 4–6 do notebook contêm tabelas, gráficos, auditoria de integridade/duplicados
e os manifestos. As células históricas dos modelos (pool de 2%, interface de 150 imagens)
permanecem intactas e ainda usam os seus splits antigos; não apresentar os seus resultados
como conformes ao protocolo.

O [protocolo experimental](paper/experimental_protocol.md) documenta os resultados,
os problemas históricos e as decisões pendentes. Macro-F1 é a métrica principal;
AULC de Macro-F1 de validação mede eficiência de anotação. O teste fica reservado
à avaliação final, sem selecionar estratégias/configurações.

Os manifestos em `data/manifests/` são a base do pipeline: treino 10859,
validação 1552, teste 3104; pool inicial 200 e não rotulado 10659, exclusivamente
do treino. SHA-256 exatos e candidatos pHash são agrupados conservadoramente para
não atravessar splits. As três exclusões humanas estão em `excluded_images.csv`;
não foram modificadas imagens. Os 234 candidatos visuais existentes permanecem
como resultados históricos, sem recalcular ou apresentar a interface dos 55 pares.
As atribuições dos restantes IDs e o pool inicial não mudaram; as exclusões
removem só três imagens do treino não rotulado.

A reaplicação das exclusões aos manifestos corre com `REFRESH_MANIFESTS = True` (secção 6.2).

O inventário completo é reservado à auditoria/oráculo; a vista do aprendiz obtém
apenas IDs opacos para o pool não rotulado. Hashing e leitura de imagens usam cache
em `artifacts/eda/`; apagar essa cache força nova auditoria. Gráficos e miniaturas
ficam nessa pasta ignorada pelo Git; não guardar outputs de miniaturas no notebook
versionado. Apenas metadados e identificadores, sem imagens, são versionáveis.

## Active Learning (protocolo)

Comparação Random / Least Confidence / Margin / Entropy com features MobileNetV2 + Logistic Regression,
5 seeds, 200 rótulos iniciais e execução contínua até 5 000 rótulos (batch 100). **Macro-F1 de validação** é a
métrica principal; AULC normalizada e o nº de imagens para atingir 90%/92% medem a eficiência de anotação.
O teste não é usado para escolher estratégias. Detalhes em `paper/experimental_protocol.md`.

No notebook, `RUN_EXPERIMENT_A = True` (secção 8.4) executa o protocolo completo (retomável, ~15 min).

Os resultados vão para `artifacts/active_learning/` (fora do Git); a secção de comparação do notebook
só lê esses ficheiros e produz as curvas, a tabela comparativa, a AULC e a eficiência de anotação.

## Duplicados exatos após revisão humana

```powershell
uv run --locked jupyter lab notebooks/01_waste_classification_hitl.ipynb
```

Depois da análise de duplicados na EDA, executar **Human-reviewed exact duplicates
with conflicting original labels**: apresenta a tabela de exclusões e os 3 grupos
lado a lado, com ficheiro, classe Kaggle, grupo e indicação de excluída/mantida, a partir
dos dados originais (pré-exclusão). A distribuição original e a efetiva são apresentadas separadamente.
Foi removida toda a subsecção visual dos 55 pares, incluindo os seus outputs.

`refresh_effective_dataset()` (secção 6.2) aplica as exclusões e atualiza as estatísticas,
de forma idempotente; não sorteia splits nem recalcula hashes ou candidatos.
Os ficheiros históricos em `data/review/` ficam preservados, anteriores à curadoria,
e não alteram o dataset efetivo. Não há uma nova interface de revisão.
