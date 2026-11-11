# Seleção de atributos: resumo do teste

Classificador único: KNN padrão (k=5, pesos uniform, distância euclidean) com `StandardScaler` dentro do `Pipeline`. A seleção usa só o treino; cada modelo é avaliado no teste uma única vez por rodada.

Agregação entre 5 rodadas: `media ± desvio` (desvio amostral, `ddof=1`). Colunas `media`, `desvio` e `media_mais_desvio` estão em `summary_selecao.csv`.

| Experimento | Acurácia | Recall anormal | F1 macro | Nº de atributos | Tempo de seleção (ms) |
| --- | --- | --- | --- | --- | --- |
| Referência (todos) | 0.8032 ± 0.0350 | 0.8333 ± 0.0292 | 0.7794 ± 0.0410 | 6.00 ± 0.00 | 0.0 ± 0.0 |
| Filtro (ANOVA F) | 0.7774 ± 0.0575 | 0.8095 ± 0.0842 | 0.7521 ± 0.0568 | 3.20 ± 1.48 | 191.0 ± 23.7 |
| Wrapper (forward) | 0.7935 ± 0.0310 | 0.7952 ± 0.0494 | 0.7755 ± 0.0295 | 3.40 ± 0.55 | 524.6 ± 22.2 |
| Embutido (Random Forest) | 0.7839 ± 0.0436 | 0.8190 ± 0.0398 | 0.7569 ± 0.0533 | 1.80 ± 0.45 | 218.6 ± 4.9 |

O tempo de seleção é só o da busca (a referência não seleciona, tempo 0). O treino do KNN está em `fit_seconds` de `per_round_selecao.csv`.

## Atributos escolhidos por rodada

| Experimento | Rodada | Semente | Nº | Atributos |
| --- | --- | --- | --- | --- |
| Filtro (ANOVA F) | 1 | 42 | 5 | pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle, pelvic_radius, degree_spondylolisthesis |
| Filtro (ANOVA F) | 2 | 43 | 3 | pelvic_incidence, pelvic_tilt, degree_spondylolisthesis |
| Filtro (ANOVA F) | 3 | 44 | 3 | pelvic_incidence, pelvic_tilt, degree_spondylolisthesis |
| Filtro (ANOVA F) | 4 | 45 | 1 | degree_spondylolisthesis |
| Filtro (ANOVA F) | 5 | 46 | 4 | pelvic_incidence, pelvic_tilt, pelvic_radius, degree_spondylolisthesis |
| Wrapper (forward) | 1 | 42 | 4 | pelvic_tilt, lumbar_lordosis_angle, pelvic_radius, degree_spondylolisthesis |
| Wrapper (forward) | 2 | 43 | 3 | lumbar_lordosis_angle, sacral_slope, degree_spondylolisthesis |
| Wrapper (forward) | 3 | 44 | 4 | pelvic_tilt, sacral_slope, pelvic_radius, degree_spondylolisthesis |
| Wrapper (forward) | 4 | 45 | 3 | lumbar_lordosis_angle, pelvic_radius, degree_spondylolisthesis |
| Wrapper (forward) | 5 | 46 | 3 | lumbar_lordosis_angle, sacral_slope, degree_spondylolisthesis |
| Embutido (Random Forest) | 1 | 42 | 2 | pelvic_radius, degree_spondylolisthesis |
| Embutido (Random Forest) | 2 | 43 | 2 | pelvic_radius, degree_spondylolisthesis |
| Embutido (Random Forest) | 3 | 44 | 1 | degree_spondylolisthesis |
| Embutido (Random Forest) | 4 | 45 | 2 | pelvic_radius, degree_spondylolisthesis |
| Embutido (Random Forest) | 5 | 46 | 2 | pelvic_radius, degree_spondylolisthesis |

## Frequência de cada atributo (em 5 rodadas)

| Atributo | Referência (todos) | Filtro (ANOVA F) | Wrapper (forward) | Embutido (Random Forest) |
| --- | --- | --- | --- | --- |
| `pelvic_incidence` | 5 | 4 | 0 | 0 |
| `pelvic_tilt` | 5 | 4 | 2 | 0 |
| `lumbar_lordosis_angle` | 5 | 1 | 4 | 0 |
| `sacral_slope` | 5 | 0 | 3 | 0 |
| `pelvic_radius` | 5 | 2 | 3 | 4 |
| `degree_spondylolisthesis` | 5 | 5 | 5 | 5 |

## Matrizes de confusão somadas (teste)

Soma das predições de teste das rodadas; ordem `[normal, anormal]`. Não é contagem de pacientes distintos.

**Referência (todos)**

```
[[ 74  26]
 [ 35 175]]
```

**Filtro (ANOVA F)**

```
[[ 71  29]
 [ 40 170]]
```

**Wrapper (forward)**

```
[[ 79  21]
 [ 43 167]]
```

**Embutido (Random Forest)**

```
[[ 71  29]
 [ 38 172]]
```
