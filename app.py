import pandas as pd
import streamlit as st
from palmerpenguins import load_penguins
from typing import Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Laboratório de Machine Learning",
    page_icon="🐧",
    layout="wide",
)

FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]


def get_slider_defaults():
    """Retorna valores padrão para os controles na UI (sliders)."""
    return {
        "bill_length": 45.0,
        "bill_depth": 17.0,
        "flipper_length": 200.0,
        "body_mass": 4000.0,
    }

FEATURE_LABELS = {
    "bill_length_mm": "Comprimento do bico (mm)",
    "bill_depth_mm": "Profundidade do bico (mm)",
    "flipper_length_mm": "Comprimento da nadadeira (mm)",
    "body_mass_g": "Massa corporal (g)",
}


@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega a base pública Palmer Penguins.

    Returns:
        pd.DataFrame: DataFrame com os dados dos pinguins.
    """
    return load_penguins()


@st.cache_resource
def treinar_modelo(
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: Optional[int],
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Prepara os dados, treina o modelo e devolve resultados da avaliação.

    Returns a tuple with the trained pipeline, training and test sets,
    labels and predictions.
    """
    dados = carregar_dados()
    X = dados[FEATURES]
    y = dados["species"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preparacao = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("preencher_ausentes", SimpleImputer(strategy="median")),
                        ("padronizar", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preparacao", preparacao),
            ("modelo", modelo),
        ]
    )

    pipeline.fit(X_treino, y_treino)
    previsoes = pipeline.predict(X_teste)

    return pipeline, X_treino, X_teste, y_treino, y_teste, previsoes


dados = carregar_dados()

st.title("🐧 Laboratório de Machine Learning")
st.write(
    "Aprenda como um modelo identifica espécies de pinguins a partir de "
    "medidas corporais reais da base pública Palmer Penguins."
)

with st.sidebar:
    st.header("Configuração do experimento")
    st.markdown("Use os controles abaixo para ajustar o experimento e visualizar resultados.")
    test_size = st.slider(
        "Percentual reservado para teste",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )
    n_estimators = st.slider(
        "Número de árvores",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
    )
    limitar_profundidade = st.checkbox("Limitar profundidade das árvores")
    max_depth = None
    if limitar_profundidade:
        max_depth = st.slider("Profundidade máxima", 1, 15, 5)
    random_state = st.number_input(
        "Semente aleatória",
        min_value=0,
        max_value=9999,
        value=42,
    )

pipeline, X_treino, X_teste, y_treino, y_teste, previsoes = treinar_modelo(
    test_size,
    int(random_state),
    n_estimators,
    max_depth,
)

aba_dados, aba_modelo, aba_previsao = st.tabs(
    ["1. Conhecer os dados", "2. Treinar e avaliar", "3. Fazer previsão"]
)

with aba_dados:
    st.header("Conhecendo a base")
    total_linhas, total_colunas = dados.shape
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", total_linhas)
    col2.metric("Variáveis", total_colunas)
    col3.metric("Espécies", dados["species"].nunique())

    st.subheader("Primeiros registros")
    st.dataframe(dados.head(10), width="stretch")

    grafico1, grafico2 = st.columns(2)
    with grafico1:
        st.subheader("Quantidade por espécie")
        st.bar_chart(dados["species"].value_counts())
    with grafico2:
        st.subheader("Massa corporal média")
        massa_media = dados.groupby("species")["body_mass_g"].mean()
        st.bar_chart(massa_media)

    st.subheader("Relação entre bico e espécie")
    dados_grafico = dados.dropna(
        subset=["bill_length_mm", "bill_depth_mm", "species"]
    )
    st.scatter_chart(
        dados_grafico,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species",
    )

    with st.expander("Ver valores ausentes"):
        st.dataframe(
            dados.isna().sum().rename("quantidade ausente").to_frame(),
            width="stretch",
        )

with aba_modelo:
    st.header("Treinamento e avaliação")
    st.write(
        "O algoritmo usado é o **Random Forest**, um conjunto de árvores de "
        "decisão que votam na classe mais provável."
    )

    acuracia = accuracy_score(y_teste, previsoes)
    col1, col2, col3 = st.columns(3)
    col1.metric("Dados de treino", len(X_treino))
    col2.metric("Dados de teste", len(X_teste))
    col3.metric("Acurácia", f"{acuracia:.1%}")

    st.subheader("Matriz de confusão")
    classes = pipeline.named_steps["modelo"].classes_
    matriz = confusion_matrix(y_teste, previsoes, labels=classes)
    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real: {classe}" for classe in classes],
        columns=[f"Previsto: {classe}" for classe in classes],
    )
    st.dataframe(matriz_df, width="stretch")

    st.subheader("Relatório de classificação")
    relatorio = classification_report(
        y_teste,
        previsoes,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(relatorio).transpose(), width="stretch")

    st.subheader("Importância das características")
    importancias = pd.Series(
        pipeline.named_steps["modelo"].feature_importances_,
        index=[FEATURE_LABELS[item] for item in FEATURES],
    ).sort_values(ascending=False)
    st.bar_chart(importancias)

with aba_previsao:
    st.header("Identifique um novo pinguim")
    st.write("Altere as medidas e peça ao modelo para estimar a espécie.")

    col1, col2 = st.columns(2)
    with col1:
        bill_length = st.slider(
            "Comprimento do bico (mm)", 30.0, 65.0, 45.0, 0.1
        )
        bill_depth = st.slider(
            "Profundidade do bico (mm)", 12.0, 25.0, 17.0, 0.1
        )
    with col2:
        flipper_length = st.slider(
            "Comprimento da nadadeira (mm)", 160.0, 240.0, 200.0, 1.0
        )
        body_mass = st.slider(
            "Massa corporal (g)", 2500.0, 6500.0, 4000.0, 50.0
        )

    novo_pinguim = pd.DataFrame(
        [
            {
                "bill_length_mm": bill_length,
                "bill_depth_mm": bill_depth,
                "flipper_length_mm": flipper_length,
                "body_mass_g": body_mass,
            }
        ]
    )

    if st.button("Identificar espécie", type="primary", width="stretch"):
        especie = pipeline.predict(novo_pinguim)[0]
        probabilidades = pipeline.predict_proba(novo_pinguim)[0]
        confianca = probabilidades.max()

        st.success(f"Espécie prevista: **{especie}**")
        st.metric("Confiança do modelo", f"{confianca:.1%}")

        tabela_probabilidades = pd.DataFrame(
            {
                "Espécie": pipeline.named_steps["modelo"].classes_,
                "Probabilidade": probabilidades,
            }
        ).set_index("Espécie")
        st.bar_chart(tabela_probabilidades)

st.divider()
st.caption(
    "Projeto educacional com Scikit-learn, Streamlit e a base pública "
    "Palmer Penguins. Uma previsão é uma estimativa estatística, não uma certeza."
)
