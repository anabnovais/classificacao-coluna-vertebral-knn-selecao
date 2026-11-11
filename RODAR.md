# Como rodar

Passo a passo no Windows, com PowerShell. Os comandos assumem que o terminal está na pasta do projeto, a que contém `src`, `tests` e `vertebral_column.csv`.

## 1. Pré-requisitos

- Python 3.12. Este projeto foi executado com Python 3.12.10.
- O arquivo `vertebral_column.csv` na pasta do projeto, ou um `column_2C.dat` no formato da UCI (6 números e o rótulo `AB` ou `NO`, separados por espaço, sem cabeçalho).

Para ver a versão instalada:

```powershell
python --version
```

Se `python` não for encontrado, tente `py -3.12 --version`.

## 2. Criar e ativar o ambiente virtual

Na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

O prompt passa a mostrar `(.venv)`. Se o PowerShell bloquear o script de ativação:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Essa mudança vale só para a janela atual. Outra opção, sem ativar, é chamar o interpretador direto: `.\.venv\Scripts\python.exe`.

Se a pasta `.venv` já existir, não é preciso criá-la de novo. Basta ativar.

## 3. Instalar as dependências

Com o ambiente ativado:

```powershell
python -m pip install -r requirements.txt
```

O arquivo fixa as versões de `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib` e `pytest`.

## 4. Rodar os testes

Ainda na pasta do projeto:

```powershell
python -m pytest
```

A saída esperada termina com todos os testes passando. Os testes cobrem o hold-out 90/10 e o caso 100/210, a ausência de vazamento do scaler e a regra de desempate da grade.

## 5. Rodar o modo rápido

O modo `--quick` usa 2 rodadas e a grade `k ∈ {1, 5, 11}`. A pasta abaixo é separada, para não apagar um resultado completo já gravado em `results`.

```powershell
python -m src.main --data-path ".\vertebral_column.csv" --quick --output-dir results_quick
```

Se quiser mais rodadas e mesmo assim a grade curta, passe `--n-rounds` junto com `--quick`. Sem `--n-rounds`, o modo rápido usa 2.

## 6. Rodar o experimento completo

```powershell
python -m src.main --data-path ".\vertebral_column.csv" --n-rounds 5 --seed 42 --output-dir results
```

`--n-rounds 5` e `--seed 42` são os padrões e podem ser omitidos:

```powershell
python -m src.main --data-path ".\vertebral_column.csv" --output-dir results
```

As sementes usadas são 42, 43, 44, 45 e 46. O terminal mostra cada rodada, a configuração escolhida e o tempo. Se o caminho for um `column_2C.dat`, troque apenas `--data-path`.

Caminho com espaços precisa de aspas:

```powershell
python -m src.main --data-path "C:\pasta com espaco\column_2C.dat" --output-dir results
```

## 7. Onde ver os resultados

Tudo fica na pasta pedida em `--output-dir` (no comando completo, `results`):

| Arquivo | Conteúdo |
| --- | --- |
| `per_round.csv` | Uma linha por experimento e rodada |
| `summary.csv` | Média, desvio e média + desvio no teste |
| `summary.md` | A mesma tabela no formato média ± desvio, os k escolhidos e a matriz somada |
| `grid_sensitivity.csv` | Métricas médias de cada combinação da grade |
| `confusion_matrix_knn.png` | Matriz de confusão do KNN ajustado |
| `boxplot_f1_macro.png` | F1 macro por variante, com os pontos das rodadas |
| `k_sensitivity.png` | F1 macro médio em função de k |
| `config.json` | Sementes, proporções e versões das bibliotecas |

## 8. Problemas comuns

**O caminho do arquivo tem espaços e o programa não acha a base.** Coloque o caminho inteiro entre aspas duplas, como no exemplo acima.

**`ModuleNotFoundError: No module named 'src'`.** O comando foi dado fora da pasta do projeto, ou o arquivo foi executado como `python src\main.py`. Entre na pasta do projeto e use `python -m src.main`.

**`ModuleNotFoundError: No module named 'sklearn'` (ou `matplotlib`, ou `pytest`).** O ambiente virtual não está ativado, ou `requirements.txt` não foi instalado. Ative `.venv` e rode `python -m pip install -r requirements.txt`. Confira com `python -c "import sklearn; print(sklearn.__version__)"`.

**O comando `python` abre a loja da Microsoft ou diz que não foi encontrado.** Instale o Python 3.12 e marque a opção de colocar o Python no PATH, ou chame `py -3.12` no lugar de `python` na criação do ambiente: `py -3.12 -m venv .venv`.

**`FileNotFoundError` ao ler a base.** Confira o nome do arquivo. Neste repositório o arquivo presente é `vertebral_column.csv`, não `column_2C.dat`.

**A base foi recusada com a mensagem de 100 normais e 210 anormais.** O leitor só segue se, depois da codificação binária, existirem exatamente esses totais. Não altere o CSV original para contornar isso.

**`Activate.ps1` não é reconhecido.** Você não está na pasta do projeto, ou o ambiente ainda não foi criado com `python -m venv .venv`.

**Acentos estranhos no terminal.** Os arquivos de `results/` são UTF-8. Se o PowerShell mostrar caracteres quebrados no progresso, rode antes:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
