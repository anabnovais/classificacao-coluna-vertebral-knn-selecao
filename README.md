# Classificação de registros biomecânicos da coluna vertebral com KNN

## Objetivo

Este projeto estima, de forma reprodutível, o desempenho do K vizinhos mais próximos (KNN) na tarefa binária da base Vertebral Column: separar registro normal de registro anormal. O protocolo é um hold-out estratificado por classe, repetido com sementes fixas. O conjunto de teste não escolhe hiperparâmetro, variante nem limiar.

O único classificador é `KNeighborsClassifier`. `DummyClassifier(strategy="most_frequent")` aparece só como linha de base trivial, que prevê sempre a classe mais frequente do treino.

## A base

**Fonte**: Data Poisoning against Differentially-Private Learners:  
Attacks and Defenses. [https://archive.ics.uci.edu/dataset/212/vertebral-column](https://archive.ics.uci.edu/dataset/212/vertebral-column). 

**Citação da base (bibtex):** 
```@misc{vertebral_column_212,
  author       = {Barreto, Guilherme and Neto, Ajalmar},
  title        = {{Vertebral Column}},
  year         = {2005},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5K89B}
}
```

**Citação do artigo (bibtex):** 
```
@inproceedings{Ma2019DataPA,
  title={Data Poisoning against Differentially-Private Learners: Attacks and Defenses},
  author={Yu-Zhe Ma and Xiao-Jin Zhu and Justin Hsu},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:85498668}
}
```

São 310 registros e 6 características numéricas, nesta ordem:


| Coluna | Nome no código             |
| ------ | -------------------------- |
| 1      | `pelvic_incidence`         |
| 2      | `pelvic_tilt`              |
| 3      | `lumbar_lordosis_angle`    |
| 4      | `sacral_slope`             |
| 5      | `pelvic_radius`            |
| 6      | `degree_spondylolisthesis` |


A UCI organiza duas tarefas. Na de três classes há 100 normais, 60 hérnias de disco e 150 espondilolisteses. Na tarefa binária, hérnia e espondilolistese formam uma única classe anormal: 100 normais e 210 anormais.

O repositório contém `vertebral_column.csv`, gravado por `baixar-base-de-dados.py` a partir do `ucimlrepo`. Esse CSV tem cabeçalho, separador vírgula e a coluna `class` com os rótulos `Normal`, `Hernia` e `Spondylolisthesis`. Não há valores ausentes.

Codificação usada em todo o código:

- anormal = 1, classe positiva (`AB`, `Abnormal`, `Hernia`, `Spondylolisthesis`, e também `DH` e `SL` se o arquivo vier com as siglas);
- normal = 0 (`NO`, `Normal`).

O experimento lê o CSV com a precisão completa. Os arquivos originais da base não são alterados. A normalização não acontece na leitura.

## Por que apenas o KNN

O short paper compara um único algoritmo com uma linha de base trivial. O KNN é um classificador de vizinhança: a previsão de um ponto novo é o voto dos `k` vizinhos mais próximos do treino. A escala dos atributos muda quem é vizinho, e por isso a normalização entra como pergunta do experimento, não como um passo escondido. Não há outro classificador no código.

Os valores de `k` são ímpares. Na votação uniforme e binária, um `k` ímpar impede empate exato de votos.

## Metodologia



### Hold-out por classe

A divisão está em `src/splitter.py`.

Para cada classe, com `n` amostras e proporções `train_ratio`, `val_ratio` e `test_ratio` (padrão 0,6 / 0,2 / 0,2):

- `n_train = round(0.6 * n)`
- `n_val = round(0.2 * n)`
- `n_test = n - n_train - n_val`

`round` é o do Python 3: no meio, arredonda para o inteiro par. Nesta base os produtos caem em inteiros, então essa regra de meio não muda os tamanhos. Uma classe precisa colocar pelo menos uma amostra em cada partição. Se o arredondamento deixar alguma parte com zero, o divisor lança `ValueError` e diz o tamanho obtido.

Exemplo com 90 amostras da classe A e 10 da classe B: A vira 54/18/18 e B vira 6/2/2. O treino final tem 60 linhas, a validação 20 e o teste 20.

Nesta base: 100 normais viram 60/20/20 e 210 anormais viram 126/42/42. Totais: treino 186, validação 62, teste 62. A proporção de anormais em cada partição fica igual à da base (210/310), porque as duas classes são cortadas pelas mesmas frações exatas.

O embaralhamento de uma rodada usa um `numpy.random.Generator` criado com `numpy.random.default_rng(semente)`. As classes são percorridas do rótulo menor para o maior, e os índices de cada uma são permutados com esse mesmo gerador. Treino, validação e teste não se sobrepõem, e a união é o conjunto inteiro. A mesma semente devolve os mesmos índices.

### Repetição por sementes

Uma divisão só pode, por acaso, deixar no treino as amostras mais fáceis de separar. O protocolo refaz o embaralhamento e o corte por classe `n_rounds` vezes. O padrão é 5. A rodada `i`, com `i` começando em 0, usa a semente `base_seed + i`. Com `base_seed = 42`, as sementes são 42, 43, 44, 45 e 46. Sementes diferentes produzem divisões diferentes.

Todas as variantes de uma rodada compartilham a mesma divisão. As rodadas são comparáveis entre si porque a lista de sementes é a mesma para todos os experimentos.

### Protocolo de cada rodada

1. Dividir.
2. Para cada configuração da grade, treinar só no treino e medir a validação.
3. Escolher a configuração pela regra abaixo. O teste não entra nessa conta.
4. Avaliar a escolhida no teste uma vez, sem re-treinar com a validação.
5. Quando há normalização, ela é um `StandardScaler` dentro de um `Pipeline`. O `fit` do pipeline recebe só o treino, então média e desvio do scaler são os do treino.

A grade com normalização também é medida no teste depois da escolha. Esse segundo uso do teste alimenta a curva de sensibilidade e não volta para a seleção. Na variante sem normalização, o teste é calculado só na configuração já escolhida.

A linha de base não passa pelo scaler: `most_frequent` ignora os atributos, e a normalização não mudaria a previsão.

### Regra de seleção

Definida antes da execução, usando só a validação:

1. maior F1 macro;
2. empate: maior recall da classe anormal;
3. novo empate: configuração mais simples, nesta ordem: menor `n_neighbors`, depois `weights="uniform"`, depois `metric="euclidean"`.



### Métricas

Em validação e em teste:

- acurácia;
- recall da classe anormal (`pos_label=1`);
- F1-score macro;
- matriz de confusão com rótulos na ordem `[normal=0, anormal=1]`.



### Agregação entre rodadas

Para cada métrica e cada variante, sejam `a_1 ... a_R` os valores no **teste**:

- `media = (a_1 + a_2 + ... + a_R) / R`
- `desvio` = desvio padrão amostral dos `a_i`, com `ddof=1` (divisor `R - 1`)
- `media_mais_desvio = media + desvio`

As três métricas usam a mesma fórmula. Os CSVs guardam as três colunas. As tabelas em `summary.md` também mostram `media ± desvio`, com quatro casas decimais.

`media_mais_desvio` é a soma pedida. Não é intervalo de confiança e pode passar de 1.

### Número de rodadas e a limitação disso

O padrão é 5 rodadas. Com `R = 5`, o desvio amostral divide a soma dos quadrados por 4. Cinco números estimam mal um desvio: a média e o desvio entre rodadas têm incerteza grande. O valor serve para ver a oscilação deste protocolo, não para cravar um erro padrão populacional. Subir `n_rounds` reduz essa limitação, sem mudar a fórmula.

As rodadas também não são amostras independentes. O mesmo paciente pode cair no teste em mais de uma semente. A matriz somada conta predições, não pessoas distintas.

## Grade e configuração padrão

`KNeighborsClassifier` dentro de `Pipeline`:

- `n_neighbors` em {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}
- `weights` em {`uniform`, `distance`}
- `metric` em {`euclidean`, `manhattan`}

A busca é um laço sobre `itertools.product`. A configuração padrão, usada sem busca, é `n_neighbors=5`, `weights="uniform"`, `metric="euclidean"`, com `StandardScaler`.

O modo `--quick` reduz `k` a {1, 5, 11}. Se `--n-rounds` não for passado junto, usa 2 rodadas.

## O que cada arquivo faz



### `src/data.py`

Lê e valida a base.

- `encode_label(raw)`: converte o rótulo bruto em 0 ou 1. Rótulo fora da lista lança erro.
- `validate_binary_counts(y)`: exige 310 linhas, 100 normais e 210 anormais. Se os totais divergirem, interrompe.
- `load_dataset(path)`: devolve `X` com shape `(310, 6)` e `y` inteiro. Aceita `.csv` e `.dat`. Não normaliza.
- `_label_column(frame)`: acha a coluna de classe no CSV (de preferência `class`).
- `_read_csv(path)`: lê o CSV com as seis colunas na ordem acima.
- `_read_dat(path)`: lê o `.dat` sem cabeçalho, sete campos por linha.

Constantes: `FEATURE_COLUMNS`, `NORMAL_LABELS`, `ABNORMAL_LABELS`, `EXPECTED_N_SAMPLES`, `EXPECTED_N_NORMAL`, `EXPECTED_N_ABNORMAL`.

### `src/splitter.py`

- `class_split_sizes(n, ...)`: aplica o arredondamento e recusa classe pequena demais.
- `stratified_holdout_indices(y, seed, ...)`: embaralha cada classe com um `Generator` e devolve os índices de treino, validação e teste.
- `repeated_stratified_holdout(y, n_rounds, base_seed, ...)`: devolve uma lista de divisões. A rodada `i` usa `base_seed + i`.
- `ensure_partition(...)`: falha se houver sobreposição ou se a união não for o conjunto inteiro.
- `_validate_ratios(...)`: exige proporções positivas que somem 1.



### `src/metrics.py`

- `compute_metrics(y_true, y_pred)`: acurácia, recall anormal, F1 macro e matriz 2×2.
- `aggregate_values(values)`: calcula `media`, `desvio` (`ddof=1`) e `media_mais_desvio`. Exige pelo menos dois valores.
- `format_mean_pm(media, desvio)`: monta o texto `media ± desvio`.



### `src/models.py`

- `make_knn_pipeline(...)`: monta o pipeline. Com `scale=True`, o primeiro passo é o `StandardScaler`.
- `make_baseline()`: devolve o `DummyClassifier` da classe mais frequente.
- `iter_grid(...)`: gera as combinações da grade.
- `select_best_configuration(candidates)`: aplica a regra de seleção. Campos de teste no dicionário são ignorados.
- `_selection_key(candidate)`: chave usada por essa escolha.
- `config_to_string(...)`: descreve uma configuração em uma linha, para o log.
- `find_default_configuration(candidates)`: acha k=5, peso uniforme e distância euclidiana dentro da grade já treinada.

Constantes da grade: `FULL_NEIGHBORS`, `QUICK_NEIGHBORS`, `WEIGHTS_GRID`, `METRICS_GRID`.

### `src/experiments.py`

- `run_experiments(...)`: laço das rodadas. Imprime progresso e tempo. Na base real, confere 60/20/20 e 126/42/42 em toda rodada.
- `run_round(...)`: executa linha de base, KNN padrão, KNN ajustado e KNN ajustado sem normalização na mesma divisão.
- `_fit_grid(...)`: treina cada configuração no treino e mede só a validação.
- `_record(...)`: monta a linha de `per_round`, inclusive as quatro células da matriz de teste.
- `_grid_row(...)`: monta a linha bruta da sensibilidade.
- `_progress_line(...)`: texto de uma variante no terminal.
- `_assert_known_vertebral_split(...)`: trava os tamanhos 60/20/20 e 126/42/42 quando a base e as proporções são as do protocolo.



### `src/reporting.py`

- `write_all_results(...)`: cria a pasta de saída e grava tabelas, figuras e `config.json`.
- `build_summary(...)`: agrega o teste de cada variante.
- `build_grid_summary(...)`: agrega cada célula da grade entre as rodadas.
- `summed_confusion(...)`: soma as matrizes de teste do KNN ajustado.
- `row_normalize(...)`: divide cada linha da matriz pelo total da linha.
- `write_summary_markdown(...)`: tabela `média ± desvio`, k escolhido por rodada, frequência de k e a matriz somada.
- `plot_confusion_matrix(...)`: figura com a matriz somada e a normalizada por linha.
- `_draw_matrix(...)`: desenha um painel da matriz.
- `plot_f1_boxplot(...)`: boxplot do F1 macro no teste, com um ponto por rodada.
- `plot_k_sensitivity(...)`: F1 macro médio contra k. Um painel é a validação; o outro é o teste, marcado como descritivo.
- `write_config(...)`: grava sementes, proporções, regra, fórmula e o SHA-256 do arquivo de dados.
- `library_versions()`: versões de Python, NumPy, pandas, scikit-learn e matplotlib.



### `src/main.py`

- `build_parser()`: argumentos `--data-path`, `--n-rounds`, `--seed`, `--output-dir` e `--quick`.
- `main()`: lê a base, roda o protocolo e grava a pasta de saída. Devolve código 1 se a base for recusada ou se `--n-rounds` for menor que 2.



### `tests/`

- `test_splitter.py`: tamanhos 90/10 e 100/210, partições disjuntas, determinismo, sementes 42 a 46, erro de classe pequena, leitura do CSV e do `.dat`.
- `test_metrics.py`: agregação `[0.8, 0.9, 1.0]`, desempate por recall e por menor k, desempate seguinte por peso e distância, seleção que ignora o teste, e scaler ajustado só no treino.



## Experimentos

Todos usam as mesmas sementes.

1. **Linha de base trivial.** `DummyClassifier(strategy="most_frequent")`. A classe anormal é majoritária (210/310 ≈ 67,7%), então a acurácia no teste fica perto disso. O F1 macro fica mais baixo, porque a classe normal não é prevista.
2. **KNN padrão.** k=5, peso uniforme, distância euclidiana, com `StandardScaler`. Sem busca. Medido na validação e no teste.
3. **KNN ajustado.** Grade completa, com `StandardScaler`. A escolha usa a validação. O modelo já treinado no treino é medido no teste.
4. **Efeito da normalização.** O KNN ajustado com scaler contra o KNN ajustado pela mesma grade e pela mesma regra, sem scaler.
5. **Sensibilidade.** Para cada combinação da grade com scaler, métricas médias na validação e no teste. A figura traça o F1 macro médio contra k, com uma linha para cada par peso/distância. Não serve para escolher a configuração.
6. **Repetição.** É o laço de `n_rounds` (padrão 5). Os experimentos acima são agregados com a fórmula da seção de agregação.
7. **Avaliação final.** Matriz de confusão do KNN ajustado somada sobre as rodadas de teste, e a mesma matriz normalizada por linha. `summary.md` lista o k de cada rodada e a frequência de cada k.



## Arquivos em `results/`

O código cria a pasta. Nada disso é escrito à mão.

- `per_round.csv`: uma linha por (experimento, rodada), com semente, configuração, métricas de validação, métricas de teste e as quatro células da matriz de teste (`test_cm_00` a `test_cm_11`, ordem normal/anormal).
- `summary.csv`: uma linha por variante. Para acurácia, recall anormal e F1 macro, as colunas `media`, `desvio`, `media_mais_desvio` e `media_pm_desvio`. A coluna `split` vale `test`.
- `summary.md`: a tabela no formato `media ± desvio`, as configurações escolhidas e a matriz somada.
- `grid_sensitivity.csv`: para cada `n_neighbors`, `weights` e `metric` da grade com scaler, `media`, `desvio` e `media_mais_desvio` na validação e no teste.
- `confusion_matrix_knn.png`: matriz somada e matriz normalizada por linha.
- `boxplot_f1_macro.png`: F1 macro no teste da linha de base, do KNN padrão, do KNN ajustado e do KNN sem normalização, com os pontos das rodadas.
- `k_sensitivity.png`: curva de F1 macro médio contra k.
- `config.json`: sementes, proporções, grade, regra, fórmula de agregação, tempo e versões.



## Decisões de projeto

- A base do repositório é o CSV de três classes. A tarefa binária segue a definição da UCI: hérnia e espondilolistese viram anormal. Os totais fecham em 100 e 210, e os rótulos coincidem com o `column_2C.dat` oficial. O CSV é usado com todas as casas decimais; o `.dat` oficial arredonda a duas casas.
- O desvio é o amostral (`ddof=1`), como pedido. Com 5 rodadas o divisor é 4.
- A figura de k tem dois painéis. O da validação é o que a seleção vê. O do teste está marcado como descritivo.
- A linha de base não usa scaler, porque a estratégia não olha os atributos.
- `--n-rounds` menor que 2 é recusado: com um único número o desvio amostral não existe.
- Figuras no backend `Agg`, estilo padrão do matplotlib, títulos em português.



## Limitações

A amostra tem 310 registros. A validação tem cerca de 20 normais, então o F1 macro que escolhe o k é ruidoso: o k vencedor pode mudar de uma semente para outra. Com 5 rodadas, média e desvio entre elas são estimativas pouco precisas, e as rodadas reutilizam pacientes.

O resultado descreve este protocolo nesta base. Não é ferramenta clínica, não substitui exame e não foi avaliado em outro serviço ou em outro período.
## Atividade de seleção de atributos

Código em `src/feature_selection.py`, `src/selection_experiments.py`, `src/selection_reporting.py` e `src/main_selecao.py`. Reaproveita a base, o hold-out 60/20/20 por classe com as mesmas sementes (42 a 46), as métricas e a agregação `media ± desvio` do restante do projeto.

- **Classificador único**: KNN padrão (k=5, pesos uniformes, distância euclidiana, `StandardScaler` no `Pipeline`) em todos os experimentos.
- **Referência**: KNN com os 6 atributos.
- **Filtro**: `SelectKBest(f_classif)`; `k` escolhido por F1 macro em validação cruzada (5 dobras) dentro do treino.
- **Wrapper**: seleção sequencial para frente com o próprio KNN, guiada por F1 macro em validação cruzada dentro do treino; fica o tamanho de maior F1.
- **Embutido**: `RandomForestClassifier` (200 árvores) com `SelectFromModel(threshold="mean")`.
- Em empate de F1, vence o conjunto com menos atributos.
- A seleção só enxerga o treino. Cada modelo é avaliado no teste uma vez por rodada. A partição de validação não é usada nesta atividade.
- Saídas em `results_selecao/`: acurácia, recall anormal e F1 macro, número de atributos e tempo de seleção (média ± desvio), atributos escolhidos por rodada e frequência de cada atributo.

```powershell
python -m src.main_selecao --data-path ".\vertebral_column.csv" --output-dir results_selecao
```
