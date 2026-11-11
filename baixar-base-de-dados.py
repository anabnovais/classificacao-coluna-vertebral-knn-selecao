
from ucimlrepo import fetch_ucirepo

# Buscar a base de dados
vertebral_column = fetch_ucirepo(id=212)

# Obter os atributos e a variável-alvo
X = vertebral_column.data.features
y = vertebral_column.data.targets

# Combinar os dados
import pandas as pd

dados = pd.concat([X, y], axis=1)

# Salvar em CSV
dados.to_csv("vertebral_column.csv", index=False, encoding="utf-8")

print("Base de dados salva com sucesso!")
print(f"Dimensões: {dados.shape}")
print(dados.head())
