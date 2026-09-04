# Aula prática: Scikit-learn com Palmer Penguins

Aplicação didática que treina um modelo de classificação com scikit-learn e
exibe uma interface interativa com Streamlit para identificar espécies de
pinguins a partir de medidas biométricas.

## O que o projeto ensina

- carregamento e exploração de uma base pública;
- definição de variáveis de entrada e variável-alvo;
- tratamento de dados ausentes;
- separação entre treino e teste;
- treinamento de um `RandomForestClassifier`;
- avaliação com acurácia, matriz de confusão, precisão, recall e F1-score;
- realização de novas previsões em uma interface gráfica.

## Preparação do ambiente

É necessário ter Python 3.10 ou superior.

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Execução

```bash
streamlit run app.py
```

O terminal mostrará o endereço local da aplicação, normalmente
`http://localhost:8501`.

## Base de dados

O projeto usa a base pública Palmer Penguins, distribuída no pacote Python
`palmerpenguins`. Ela reúne medidas de três espécies observadas no arquipélago
Palmer, na Antártica. A base é carregada diretamente pelo código; não é
necessário baixar um CSV separadamente.

## Estrutura

```text
aula_scikit_learn_pinguins/
├── app.py
├── README.md
├── requirements.txt
└── roteiro_aula.md
```
