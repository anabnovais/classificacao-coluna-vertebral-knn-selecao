# Resumo do teste

Agregação das métricas de **teste** entre as rodadas. A validação escolhe a configuração; o teste não entra nessa escolha.

Fórmulas, com `a_1 ... a_R` os valores da métrica no teste:

- `media = (a_1 + ... + a_R) / R`
- `desvio` = desvio padrão amostral (`ddof=1`, divisor `R - 1`). Nesta execução, R = 5 e o divisor é 4.
- `media_mais_desvio = media + desvio`

Com poucas rodadas, média e desvio são estimativas ruidosas. `media_mais_desvio` pode passar de 1, porque não é um intervalo de confiança.

| Experimento | Acurácia | Recall anormal | F1 macro |
| --- | --- | --- | --- |
| Linha de base | 0.6774 ± 0.0000 | 1.0000 ± 0.0000 | 0.4038 ± 0.0000 |
| KNN padrão | 0.8032 ± 0.0350 | 0.8333 ± 0.0292 | 0.7794 ± 0.0410 |
| KNN ajustado | 0.8258 ± 0.0239 | 0.8619 ± 0.0310 | 0.8015 ± 0.0334 |
| KNN sem normalização | 0.8258 ± 0.0072 | 0.8333 ± 0.0292 | 0.8079 ± 0.0080 |

Colunas numéricas correspondentes estão em `summary.csv`: `media`, `desvio` e `media_mais_desvio` de cada métrica.

## Configurações escolhidas do KNN ajustado

Regra: maior F1 macro na validação; empate no recall da classe anormal; novo empate no menor k, depois peso uniforme, depois distância euclidiana.

| Rodada | Semente | k | Pesos | Distância | F1 macro (validação) | F1 macro (teste) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 42 | 13 | distance | euclidean | 0.8691 | 0.8082 |
| 2 | 43 | 15 | uniform | manhattan | 0.8725 | 0.8155 |
| 3 | 44 | 1 | uniform | euclidean | 0.8524 | 0.7943 |
| 4 | 45 | 3 | uniform | manhattan | 0.8560 | 0.8398 |
| 5 | 46 | 21 | uniform | euclidean | 0.9262 | 0.7496 |

### Frequência de k

| k | Rodadas |
| --- | --- |
| 1 | 1 |
| 3 | 1 |
| 13 | 1 |
| 15 | 1 |
| 21 | 1 |

## Matriz de confusão somada (KNN ajustado, teste)

Cada célula soma as predições de teste das rodadas. Não é uma contagem de pacientes distintos: o mesmo registro pode entrar no teste em mais de uma rodada.

Ordem dos rótulos: linha e coluna `[normal, anormal]`.

```
[[ 75  25]
 [ 29 181]]
```

Normalizada por linha (proporção dentro da classe verdadeira):

```
[[0.75   0.25  ]
 [0.1381 0.8619]]
```
