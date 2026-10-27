# EDA e protocolo experimental comum

## Estado e âmbito

Dataset: [Garbage Classification, Kaggle v1](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/versions/1),
identificador `mostafaabla/garbage-classification/versions/1`. São mantidas todas as
15 515 imagens e as 12 classes originais. Não se treinou qualquer modelo, não se
extraíram features e não se alteraram os algoritmos ou interfaces HITL.

As experiências A (MobileNetV2 pré-treinada + Logistic Regression) e B
(YOLO26s Classification + fine-tuning) deverão consumir os mesmos manifestos.
Os modelos históricos ainda **não** os consomem: esta tarefa prepara o protocolo,
sem migrar os ciclos de treino. Executar apenas a secção EDA do notebook; parar
antes do separador «Implementação científica histórica». Não utilizar Run All.

## Resultados originais da auditoria Kaggle

As contagens da tabela seguinte são as originais, anteriores à curadoria humana.

| Classe | Total | % | Treino | Validação | Teste | Inicial | Não rotulado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| battery | 945 | 6.091 | 662 | 94 | 189 | 12 | 650 |
| biological | 985 | 6.349 | 688 | 99 | 198 | 13 | 675 |
| brown-glass | 607 | 3.912 | 425 | 60 | 122 | 8 | 417 |
| cardboard | 891 | 5.743 | 624 | 89 | 178 | 11 | 613 |
| clothes | 5325 | 34.322 | 3728 | 533 | 1064 | 69 | 3659 |
| green-glass | 629 | 4.054 | 440 | 63 | 126 | 8 | 432 |
| metal | 769 | 4.956 | 538 | 77 | 154 | 10 | 528 |
| paper | 1050 | 6.768 | 735 | 105 | 210 | 14 | 721 |
| plastic | 865 | 5.575 | 605 | 86 | 174 | 11 | 594 |
| shoes | 1977 | 12.743 | 1383 | 198 | 396 | 25 | 1358 |
| trash | 697 | 4.492 | 489 | 70 | 138 | 9 | 480 |
| white-glass | 775 | 4.995 | 542 | 78 | 155 | 10 | 532 |
| **Total** | **15515** | **100** | **10859** | **1552** | **3104** | **200** | **10659** |

`clothes` é maioritária (34,32%) e `brown-glass` minoritária (3,91%): razão
5325/607 = 8,77:1. A accuracy pode ocultar falhas nas classes pequenas. Não se
aplicou oversampling, undersampling, pesos por classe ou qualquer balanceamento.

| Característica | Mínimo | Média | Mediana | Percentil 95 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Largura (px) | 51 | 349,822 | 400 | 512 | 888 |
| Altura (px) | 100 | 351,706 | 384 | 534 | 936 |
| Largura/altura | 0,1275 | 1,0842 | 1 | 1,6514 | 5,02 |

Há 892 resoluções distintas. As mais frequentes são 400×533 (2590 imagens),
512×384 (2358), 225×225 (2004) e 400×534 (1077). Há imagens bastante alongadas;
a futura escolha de resize/crop deverá ser registada e avaliada só com validação.

Todos os nomes terminam em `.jpg`, mas os formatos reais são 15 481 JPEG/RGB e
34 PNG/paleta (`P`). Essa discrepância extensão/conteúdo não significa corrupção:
`Pillow.verify()` e descodificação completa passaram em todas as imagens. Não
foram detetadas imagens ilegíveis. As implementações existentes convertem para RGB.

As tabelas, histogramas, gráfico de distribuição e exemplos por classe estão no
início do notebook. Exemplos usam `default_rng(42)` e caminhos ordenados, sem
modificar o estado aleatório global dos modelos. Gráficos e miniaturas são
exportados para `artifacts/eda/`, ignorado pelo Git. Só os gráficos estatísticos
ficam embebidos no notebook; miniaturas do dataset são outputs locais, não versionados.

## Duplicados e conflitos de rótulo

- **SHA-256 binário:** 18 grupos, 37 imagens envolvidas e 19 cópias redundantes
  (`Σ(n−1)`). São duplicados confirmados dos bytes, não apenas hashes visuais iguais.
- **pHash:** 234 pares candidatos adicionais, envolvendo 331 imagens; 55 pares
  têm classes diferentes. Os candidatos não são automaticamente duplicados confirmados.
- **Conflitos confirmados:** três grupos de bytes idênticos têm rótulos diferentes:
  `metal/metal459.jpg` ↔ `white-glass/white-glass412.jpg`;
  `plastic/plastic159.jpg` ↔ `white-glass/white-glass129.jpg`;
  `plastic/plastic556.jpg` ↔ `white-glass/white-glass761.jpg`.

O pHash usa grayscale 32×32 após orientação EXIF, DCT-II ortonormal, coeficientes
8×8 e mediana excluindo DC para obter 64 bits. Distância de Hamming ≤4.
Cinco blocos disjuntos (13/13/13/13/12 bits) indexam os hashes: dois hashes que
diferem em até quatro bits partilham pelo menos um bloco. Calcula-se a distância
completa só entre candidatos do índice e excluem-se pares já confirmados por SHA-256.
Não foi necessária uma nova dependência; reutilizaram-se Pillow, NumPy e SciPy.

Há falsos positivos/negativos possíveis e nenhuma garantia de detetar todos os
recortes, rotações ou fotografias do mesmo objeto. O limiar 4 é uma escolha fixa
deste protocolo, não foi otimizado com resultados de modelos. A galeria inicial
contém exemplos visualmente muito semelhantes; a revisão completa dos 234 pares
continua pendente. Nenhuma imagem/rótulo foi removido, movido ou modificado.

### Três exclusões humanas confirmadas

O anotador humano reviu três grupos de duplicados exatos (SHA-256) com rótulos contraditórios e
decidiu excluir as cópias incorretamente rotuladas: `plastic/plastic159.jpg` (mantida
`white-glass/white-glass129.jpg`), `metal/metal459.jpg` (mantida `white-glass/white-glass412.jpg`)
e `white-glass/white-glass761.jpg` (mantida `plastic/plastic556.jpg`). A decisão está em
`data/manifests/excluded_images.csv` (caminho, classe original, motivo, cópia mantida, proveniência).
Nenhum rótulo Kaggle foi alterado e a cache Kaggle não foi modificada; a política anterior de
correção de rótulos foi substituída (o histórico permanece no Git). `inventory.csv` mantém as 15 515
imagens originais para a EDA; o dataset efetivo tem 15 512 imagens (metal 768, plastic 864,
white-glass 774; restantes classes inalteradas). Os splits não foram refeitos: os três ids estavam em
`train`/`unlabeled` e foram removidos; os hashes anteriores ficam em `protocol.json`
(`pre_exclusion_manifest_sha256`). Efetivos: treino 10856, validação 1552, teste 3104, pool inicial
200 (inalterado, nenhuma exclusão lhe pertencia) e unlabeled 10656.

`GarbageDataset` omite os caminhos de `excluded_images.csv` e as duas experiências deverão partilhar
`load_effective_inventory()`. Não percorrer diretamente as pastas Kaggle no futuro treino YOLO. O treino e a migração dos ciclos AL continuam
fora desta tarefa. Os arquivos de features sem o SHA-256 atual de `excluded_images.csv`
são recusados explicitamente pelos carregadores do notebook; não são apagados.
Reiniciar o kernel para não reutilizar variáveis/modelos históricos em memória.

A secção dos três grupos exatos usa o inventário original (pré-exclusão) e indica, por imagem,
a classe Kaggle e se foi excluída ou mantida, precedida pela tabela de exclusões. A interface dos 55 pares foi removida;
os 234 candidatos originais ficam preservados, sem recomputação ou novas decisões.

**Política conservadora de prevenção de leakage:** formar componentes conexas
pela união de grupos SHA-256 e de pares pHash candidatos. Toda a componente fica
no mesmo conjunto, mesmo que contenha classes distintas. O maior grupo tem 14
imagens. Isto evita fuga pelos pares identificados sem declarar candidatos como
confirmados; pode manter imagens distintas juntas. Rever esta política e os
conflitos adicionais antes do treino. Os três conflitos acima estão resolvidos
por decisão humana; os grupos e splits desta tarefa não mudam. Uma revisão futura que altere grupos exige gerar novamente
e congelar todos os manifestos para ambas as experiências.

## Divisão reproduzível

1. Ordenar inventário pelo caminho relativo POSIX; IDs são SHA-256 desse caminho
   UTF-8. O SHA-256 dos bytes é uma coluna diferente, usada para identificar cópias.
2. Agrupar conforme a política anterior.
3. Executar `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`:
   folds 0–1 → teste; fold 2 → validação; folds 3–9 → treino.
   Trata-se de alocação estratificada por grupos, não de avaliação cross-validation.
   [Documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
4. Resultado: treino 10859 (69,990%), validação 1552 (10,003%) e teste 3104 (20,006%).
   Cada imagem aparece uma única vez, e nenhum grupo exato/candidato atravessa splits.
5. Quotas iniciais proporcionais às classes **do treino**, com arredondamento por
   maiores restos e desempate alfabético. Selecionar com `default_rng(42)` apenas
   de componentes singleton: 200 imagens, representando as 12 classes. Isto evita
   aliases das imagens inicialmente conhecidas no pool não rotulado.
6. Restantes 10659 imagens de treino → unlabeled pool. Validação/teste nunca entram
   nos pools. Os testes admitem até 1 ponto percentual de desvio global e até 2
   pontos percentuais na fração de cada classe destinada a cada split; preservam-se
   grupos inteiros em vez de forçar os números a partir imagens duplicadas.

`data/manifests/inventory.csv` é o registo auditável: ID, caminho relativo à raiz
`garbage_classification/`, classe Kaggle (`label`, nunca alterada), SHA-256, pHash, dimensões, formato,
integridade e grupo. `train.csv`, `validation.csv`, `test.csv` e `unlabeled.csv`
contêm apenas IDs. `labeled_initial.csv` acrescenta rótulo e proveniência.
`visual_candidates.csv` contém todos os pares. `protocol.json` regista método,
seed, contagens e checksums dos CSVs. `.gitattributes` fixa LF nos manifestos para
que os checksums sobrevivam a checkouts Windows/POSIX. Não há caminhos da cache
nos manifestos e não se copiam imagens para pastas de treino/validação/teste.

Os cálculos pesados são guardados em `artifacts/eda/image_cache.json`, separados
dos manifestos versionáveis. A cache usa handle/schema e, por ficheiro, caminho,
tamanho e `mtime_ns`; não deteta alterações que falsifiquem deliberadamente
tamanho e timestamp. Apagar a cache força novo hashing e descodificação completa.
Os hashes de conteúdo permanecem nos manifestos para auditoria. O `uv.lock`
existente fixa as versões utilizadas; não foram adicionadas dependências.

## Fronteiras dos rótulos e custos

O inventário completo e os seus caminhos (que incluem nomes de classes) são
privilegiados: auditoria/preparação/oráculo/avaliação. **Não entregar essa tabela
ao aprendiz.** `load_pool_views()` devolve rótulos conhecidos do pool inicial e
apenas IDs opacos para unlabeled. `load_pool_image()` limita o acesso ao treino e
devolve pixels RGB sem caminho/rótulo/metadados; não aceita IDs de validação/teste.
Esta separação é uma fronteira de API e uma convenção experimental; não impede
um programa malicioso com acesso ao disco de consultar o ground truth.

Os 200 rótulos iniciais vêm do Kaggle, com `label_source=initial_simulation_oracle`:
constituem supervisão simulada, não 200 anotações humanas realizadas. O futuro
oráculo só revela rótulos para queries selecionadas e contabilizadas. A futura
anotação humana deve guardar o rótulo efetivamente escolhido e a sua proveniência,
distinguindo-o do ground truth. Não consultar o ground truth do unlabeled pool
para treinar, escolher imagens ou calcular incerteza. Este ciclo ainda não existe.

Validação é independente e fica disponível para hiperparâmetros, early stopping,
checkpoints, comparação de configurações e monitorização do AL. Os seus 1552
rótulos Kaggle são supervisão conhecida pelo avaliador e devem ser contabilizados
separadamente do orçamento de treino/queries ao reportar custo de anotação total.
Os 3104 rótulos de teste ficam reservados à avaliação final, após congelar decisões.

## Métricas e seleção

**Primária: Macro-F1**, média não ponderada do F1 das 12 classes fixas. Usar a lista
completa de classes e `zero_division=0`, sem omitir classes sem previsão.
[Definição e parâmetros no scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

**Eficiência: AULC** da Macro-F1 de validação em função de `B`, número acumulado
de imagens rotuladas, incluindo as 200 iniciais. Integrar pela regra trapezoidal
em orçamentos/intervalo comuns e reportar também `AULC / (B_max − 200)`.
Não extrapolar curvas incompletas; comparar no intervalo comum predefinido.
O número de queries humanas e o número de imagens que recebem rótulo podem
diferir caso se propaguem anotações por grupos; essa política ainda precisa de decisão.

**Secundárias:** accuracy, precision/recall/F1 por classe, matriz de confusão,
tempo de treino e inferência, memória e custo computacional. Distinguir warmup,
extração de features e treino, registando hardware/software. Usar as mesmas
divisões, pool inicial e orçamentos nas experiências A e B, sem exigir modelos iguais.

Random Sampling, Least Confidence, Margin Sampling e Entropy Sampling são
estratégias de query, não métricas. Selecionar principalmente por Macro-F1 e
eficiência de anotação de validação; nunca por curvas de teste repetidas.
Transforms/normalizações aprendidas usam só treino rotulado. Validação/teste
não participam em gradient updates, ajuste de scaler ou aquisição de amostras.

## Protocolo da Experiência A (Active Learning com features pré-extraídas)

**Pergunta:** conseguem as estratégias de AL aproximar-se do desempenho com todo o treino usando
muito menos imagens anotadas do que Random Sampling?

- **Dados:** 15 512 imagens efetivas (após as 3 exclusões), treino 10 856, validação 1 552, teste 3 104;
  splits e exclusões inalterados. Features MobileNetV2 (1280-d) em `notebooks/garbage_features.npz`,
  alinhadas por ID; o arquivo é recusado se o hash das exclusões não corresponder ou se tiver linhas a mais/menos.
- **Classificador (igual para todas as estratégias):** `LogisticRegression(max_iter=1000, random_state=42)`,
  sem scaler nem ajuste por estratégia. Retreinado de raiz a cada iteração com todas as imagens rotuladas.
- **Estratégias:** Random, Least Confidence, Margin, Entropy; implementações originais (desempate estável).
  O Random usa `default_rng([seed, iteração])` em vez do estado global, o que garante retoma exata.
- **Seeds:** `[42, 123, 456, 789, 2026]`. Por seed, as 4 estratégias partilham o mesmo conjunto inicial
  (200 imagens) e o mesmo pool não rotulado (treino − inicial). Seed 42: `labeled_initial.csv` existente.
  Outras seeds: mesma regra do manifesto (alocação proporcional por classe, maior resto, grupos singleton,
  `default_rng(seed)`), calculada em memória a partir de `inventory.csv`; nenhum manifesto é alterado.
  A seleção inicial usa rótulos de treino (simulação); as consultas só usam probabilidades do modelo.
- **Orçamento:** uma única execução contínua 200 → 5 000 por (seed, estratégia), batch fixo de **100**
  (48 aquisições, 49 treinos). Os checkpoints `[200, 500, 1000, 1500, 2000, 3000, 5000]` (nº total de
  imagens rotuladas) são todos múltiplos de 100 e, portanto, atingidos exatamente. Escolha de batch
  igual para as 4 estratégias; não foi otimizada nem ajustada com resultados.
- **Métrica:** Macro-F1 `f1_score(average="macro", zero_division=0)` na **validação**, avaliada em cada iteração.
  O teste só é registado com `--with-test`, depois de congelado o protocolo, e é apenas descritivo.
- **AULC:** `AULC = (1/(5000−200)) · ∫ Macro-F1(n) dn` (regra dos trapézios sobre as 49 iterações),
  por seed; reportada como média ± DP (ddof=1) entre seeds. Não é ROC-AUC.
- **Limiares exploratórios:** primeiro `n` (granularidade do batch) com Macro-F1 de validação ≥ 90% e ≥ 92%,
  por estratégia e seed. Se não for atingido regista-se «não atingido», nunca 5 000.
- **Comparação com Random:** diferença de AULC emparelhada por seed (média, DP, nº de seeds melhores).
  É descritiva: com 5 seeds o menor p-valor exato bilateral do Wilcoxon é 0,0625, pelo que não se declara
  significância estatística. As seeds variam o conjunto inicial e (Random) as consultas, não os splits.
- **Full Training Baseline:** mesmo classificador com as 10 856 imagens de treino, treinado uma vez
  (determinístico) e calculado pelo código; nunca introduzido manualmente.
- **Resultados:** `artifacts/active_learning/<nome>/` (ignorado pelo Git): `config.json` (configuração,
  SHA-256 das features, dos manifestos, das exclusões e dos conjuntos iniciais, versões), `curves.jsonl`
  (seed, estratégia, iteração, nº rotulado, split, Macro-F1, tempos, IDs adquiridos) e `full_baseline.json`.
  Os gráficos e tabelas do notebook só leem estes ficheiros. A retoma valida configuração e hashes.

Execução no notebook (todo o código está em `notebooks/01_waste_classification_hitl.ipynb`):
`RUN_EXPERIMENT_A = True` (secção 8.4) corre o protocolo completo, retomável (20 execuções).

Limitações: as consultas operam por imagem, sem tratar duplicados quase exatos do pool não rotulado
(política de grupos ainda pendente); o aprendiz é linear sobre features fixas (conclusões não se transferem
automaticamente para a Experiência B); a incerteza dos pontos individuais é limitada por 5 seeds.

## Problemas históricos preservados e decisões pendentes

- Baseline original: split 80/20 sem validação independente nem grupos.
- Pool de 2% e interface de 150 imagens: partem do dataset inteiro, incluindo
  imagens que o novo protocolo reserva a validação/teste; não são pools conformes.
- Simulação histórica (substituída na Experiência A pelo protocolo acima): split aleatório não estratificado e consulta do teste a cada iteração;
  pode induzir leakage de seleção se usado para escolher estratégia/orçamento.
- Dependências entre células: `garbage_features.npz`; variáveis globais `X`, `y`,
  `filenames`, pools e modelos; funções de query redefinidas; `os.listdir` não
  garante a mesma ordem entre sistemas. Futuro alinhamento de features deve usar IDs.
- Primeiro `HITLLabeler`: callback ignora o `label_id` recebido; não foi corrigido.
- Falta migrar explicitamente os modelos/AL para os manifestos comuns. Não apresentar
  resultados das células históricas como experiências realizadas com este protocolo.
- Os três conflitos exatos foram resolvidos por exclusão humana (acima). Não excluir
  outras imagens nem tirar novas conclusões sobre os candidatos visuais nesta tarefa.
- Experiência A: orçamento, batch, seeds e validação por iteração estão fixados acima. Pendentes para a
  Experiência B: hiperparâmetros/early stopping/checkpoints e critério prévio de desempate Macro-F1/AULC.
- Decidir consultas por imagem ou grupo, propagação de rótulos e custo humano,
  separando rótulos simulados, humanos e de validação. Não implementar AL nesta tarefa.

## Reproduzir e testar

No notebook, `REFRESH_MANIFESTS = True` (secção 6.2, `refresh_effective_dataset()`) aplica `excluded_images.csv` (idempotente) e atualiza checksums e estatísticas
nos manifestos existentes. Não executa nova divisão, hashing ou cálculo de candidatos.
Os splits foram criados originalmente com os rótulos Kaggle; esta curadoria posterior
preserva os seus IDs e apenas atualiza as contagens. A função recusa alterações
inesperadas aos ficheiros de pertença dos conjuntos/pools.
As figuras estão nas secções 4–5 do notebook.
