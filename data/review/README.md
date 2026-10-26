# Revisão humana do dataset Kaggle v1

## Abrir a revisão

```powershell
uv run --locked jupyter lab notebooks/02_dataset_review.ipynb
```

Executar as duas células de código do notebook: a primeira mostra a fila e os
estados; a segunda apresenta a interface. O notebook científico original não é
alterado. Não há treino, inferência de modelos, novo processamento EDA, alteração
de rótulos ou regeneração de splits.

## Fila prioritária

| Prioridade | Casos | Correspondência |
| --- | ---: | --- |
| 1 | 3 | Grupo exato SHA-256 com classes diferentes |
| 2 | 55 | Par candidato pHash com classes diferentes |
| 3 | 15 | Restantes grupos exatos SHA-256 |
| 4 | 179 | Restantes pares candidatos pHash |
| **Total** | **252** | **18 grupos exatos + 234 pares visuais** |

Um grupo exato é apresentado como um único caso, incluindo todas as suas imagens;
não é expandido em pares que inflacionariam a contagem. A lista completa fica em
`cases.json`, com os hashes dos manifestos que a originaram. Cada caso inclui ID,
prioridade, tipo, distância pHash (ou null para SHA-256), classes originais, IDs,
caminhos relativos, SHA-256 de cada imagem e os splits já atribuídos.

O ID do caso é SHA-256 de `[dataset com versão, tipo, IDs ordenados dos membros]`,
com prefixo `exact-` ou `visual-`. Não depende da posição na fila, da ordem dos
ficheiros ou do estado da revisão. A interface mostra o identificador completo.

Os três casos prioritários são:

| Imagem A | Imagem B | Split existente |
| --- | --- | --- |
| `metal/metal459.jpg` | `white-glass/white-glass412.jpg` | treino |
| `plastic/plastic159.jpg` | `white-glass/white-glass129.jpg` | treino |
| `plastic/plastic556.jpg` | `white-glass/white-glass761.jpg` | treino |

A identidade dos bytes confirma cópias exatas, mas não identifica automaticamente
a classe correta. Para pHash, distância pequena é apenas evidência de semelhança:
não equivale a uma decisão humana nem determina se o rótulo está errado.

## Registar decisões

1. Comparar as imagens lado a lado e consultar classe, caminho, tipo/distância e
   split. A interface verifica o SHA-256 apenas das imagens abertas contra o valor
   já registado; não recalcula hashes de todo o dataset. Os originais são lidos,
   nunca regravados. As miniaturas são criadas em memória.
2. Preencher revisor, decisão e justificação; clicar **Acrescentar decisão**.
   Mudar opções ou abrir um caso não grava nada. Revisor/justificação vazios são
   rejeitados. Não usar múltiplas sessões concorrentes de revisão.
3. Para rever uma decisão, desativar **Apenas pendentes** e acrescentar um novo
   evento com a razão da revisão. Nunca editar, substituir ou truncar o histórico.
4. Reexecutar a célula da tabela para atualizar o resumo após novas decisões.

| Código no histórico | Estado apresentado | Interpretação |
| --- | --- | --- |
| `pending` | Pendente de revisão | Ainda não avaliado ou nota de trabalho; sem evento, é o estado inicial. |
| `confirmed_duplicate` | Duplicado confirmado | A inspeção confirma duplicação; a política de retenção continua por decidir. |
| `different_images` | Imagens diferentes | Candidato visual considerado falso positivo na inspeção. |
| `incorrect_label` | Rótulo incorreto | Indicar na justificação os ficheiros afetados, o problema e a classe proposta, se conhecida. |
| `ambiguous` | Caso ambíguo | Indicar a dúvida e o que falta para decidir. |

O registo `decisions.jsonl` começa vazio e é versionável. Cada linha contém
`event_id`, `case_id`, `decision`, `justification`, `reviewer`, `reviewed_at_utc` e
`previous_event_id`. A data é UTC com timezone. O estado atual é o último evento
do caso; todos os anteriores continuam visíveis na interface. O histórico é
validado ao ler; JSON inválido, casos desconhecidos, campos em falta ou cadeias
incoerentes produzem um erro explícito, sem ignorar ou reescrever decisões.
Não foram criadas decisões humanas pela preparação nem pelos testes: estes
escrevem exclusivamente em diretórios temporários.

Uma decisão é uma anotação auditável sobre o caso, **não** uma alteração do dataset.
É possível que uma imagem seja duplicada e tenha um rótulo incorreto; a evidência
de correspondência permanece no caso, e a justificação deve explicitar o conflito
mesmo que se registe uma única categoria de decisão no evento.

## Preservação e congelamento futuro

- `cases.json` é um snapshot determinístico dos manifestos EDA existentes. A
  preparação pode ser repetida, mas não sobrescreve a fila nem o histórico. Se os
  manifestos mudarem, o processo recusa substituir a revisão existente; uma nova
  versão deverá ser criada deliberadamente, mantendo a anterior.
- Imagens continuam exclusivamente na cache Kaggle; os caminhos são relativos à
  raiz `garbage_classification/`. O histórico não contém caminhos absolutos.
- Antes de versionar o notebook, limpar outputs e estado dos widgets para não
  incluir imagens do dataset. A fila e o histórico são metadados versionáveis.
- Rever primeiro P1 e P2, depois os restantes casos. Casos ambíguos e propostas
  de rótulo incorreto precisam de adjudicação antes de congelar o protocolo.
  Confirmar uma cópia em P1 não resolve qual classe usar: documentar essa decisão.
- Depois da revisão, decidir numa tarefa separada a retenção/exclusão, conflitos
  de rótulo e eventuais alterações aos grupos/splits. Aplicar mudanças por metadados
  ou por uma camada de curadoria reproduzível, preservando todos os originais.
- Inspecionar qualidade nos splits antes do congelamento não autoriza escolher
  modelos ou estratégias por desempenho de teste. Não há previsões na revisão.

## Preparação e testes

```powershell
uv run --locked python notebooks/dataset_review.py
uv run --locked python -m unittest discover -s notebooks -p test_dataset_review.py -v
```

A preparação lê `inventory.csv`, `visual_candidates.csv` e os manifestos de splits,
verifica os seus checksums existentes e reutiliza os grupos SHA-256 registados.
Não chama `scan_dataset`, `visual_candidates` nem `write_manifests`.
Os testes cobrem cobertura/prioridades, IDs independentes da ordem, preservação
do histórico, recusa de snapshots alterados, validação das decisões e imagens,
filtros da interface e gravação apenas por ação explícita.
