import pandas as pd
import numpy as np
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import warnings

# config inicial
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # limpa o output do optuna

# carregamento
try:
    df = pd.read_csv('insurance.csv')
    print("Dataset carregado")
except FileNotFoundError:
    # dataset dummy apenas para o código funcionar 
    data = {
        'age': np.random.randint(18, 65, 1000),
        'sex': np.random.choice(['male', 'female'], 1000),
        'bmi': np.random.normal(30, 5, 1000),
        'children': np.random.randint(0, 5, 1000),
        'smoker': np.random.choice(['yes', 'no'], 1000),
        'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], 1000),
        'charges': np.random.normal(13000, 12000, 1000) # Alvo
    }
    df = pd.DataFrame(data)
    print("Dataset dummy carregado")

# separação X e y
X = df.drop('charges', axis=1)
y = df['charges']

# divisão treino/teste
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessamento
# definindo colunas
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# pipelines para transformação
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' evita multicolinearidade
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# modelos e tuning
# dict com os modelos e espaços de busca
modelos_config = {
    "Ridge": {
        "model": Ridge(random_state=42),
        "params": {
            "regressor__alpha": optuna.distributions.FloatDistribution(0.1, 10.0, log=True)
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "regressor__n_estimators": optuna.distributions.IntDistribution(50, 300),
            "regressor__max_depth": optuna.distributions.IntDistribution(3, 15),
            "regressor__min_samples_split": optuna.distributions.IntDistribution(2, 10)
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        "params": {
            "regressor__n_estimators": optuna.distributions.IntDistribution(100, 500),
            "regressor__max_depth": optuna.distributions.IntDistribution(3, 10),
            "regressor__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
            "regressor__subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
            "regressor__colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0)
        }
    }
}

melhores_modelos = {}
resultados_selection = []

print("começando otimização e seleção dos modelos")

for nome, config in modelos_config.items():
    print(f"Otimizando: {nome}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', config["model"])
    ])
    
    # OptunaSearchCV
    search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=config["params"],
        cv=3,              # CV interno rápido para seleção
        n_trials=20,       # numero de trials 
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train_full, y_train_full)
    
    melhores_modelos[nome] = search.best_estimator_
    resultados_selection.append({
        "Modelo": nome,
        "Melhor R2 (CV Interno)": search.best_score_,
        "Melhores parametros": search.best_params_
    })

# exibir ranking dos modelos
df_results = pd.DataFrame(resultados_selection).sort_values(by="Melhor R2 (CV Interno)", ascending=False)
print("\nRanking:")
print(df_results[["Modelo", "Melhor R2 (CV Interno)"]])

# selecionar melhor modedlo 
melhor_nome = df_results.iloc[0]["Modelo"]
# pega o objeto base do modelo
melhor_pipeline_base = modelos_config[melhor_nome]["model"] 
# pega o espaço de busca
melhor_params_space = modelos_config[melhor_nome]["params"] 

print(f"\nModelo Campeão Selecionado para Nested CV: {melhor_nome}")

# avaliacao nested CV
# nested CV para verificar se o score não foi "sorte" ou overfitting nos dados de validação.

# divide em treino/teste
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# roda dentro do outer, optuna
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

outer_scores = []

print(f"\n--- Iniciando Nested CV (5 Folds) para {melhor_nome} ---")
print("Isso pode demorar um pouco...")

# loop manual do Nested CV para controle total
X_array = X.values # facilita indexação no loop
y_array = y.values

# recria a pipeline
pipeline_final = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', melhor_pipeline_base) # usa o modelo que melhor desempenhou
])

# usa OptunaSearchCV dentro do cross_val_score, passando uma instância dele
optuna_search_final = OptunaSearchCV(
    estimator=pipeline_final,
    param_distributions=melhor_params_space,
    cv=inner_cv,
    n_trials=20, # numero de trials
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# executa o nested CV
# cross_val_score faz o loop externo, OptunaSearchCV faz o interno
scores_nested = cross_val_score(optuna_search_final, X, y, cv=outer_cv, scoring='r2', n_jobs=-1)

print("\nResultados do Nested CV:")
for i, score in enumerate(scores_nested):
    print(f"Fold {i+1}: R2 = {score:.4f}")

mean_r2 = np.mean(scores_nested)
std_r2 = np.std(scores_nested)

print("="*60)
print(f"DESEMPENHO FINAL GENERALIZADO ({melhor_nome}):")
print(f"R² Médio: {mean_r2:.4f} ± {std_r2:.4f}")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# usando o modelo com todos os dados
pipeline_final.fit(X, y)

# predicao para o grafico de dispersao
y_pred_cv = cross_val_predict(pipeline_final, X, y, cv=5, n_jobs=-1)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Avaliação Final do Modelo: {melhor_nome}', fontsize=20, weight='bold')

# grafico 1
ax1 = axes[0, 0]
sns.boxplot(y=scores_nested, ax=ax1, color='lightblue', width=0.3)
sns.stripplot(y=scores_nested, ax=ax1, color='red', size=8, jitter=True)
ax1.set_title('Estabilidade da Validação Cruzada ($R^2$)', fontsize=14)
ax1.set_ylabel('Score $R^2$')
ax1.set_xlabel('Folds (5 iterações)')
# valor medio
ax1.text(0.35, mean_r2, f'Média: {mean_r2:.3f}', fontsize=12, fontweight='bold', color='darkblue')

# grafico 2
ax2 = axes[0, 1]
sns.scatterplot(x=y, y=y_pred_cv, ax=ax2, alpha=0.5, edgecolor=None, color='teal')
# linha de referencia perfeita (y = x)
min_val, max_val = min(y.min(), y_pred_cv.min()), max(y.max(), y_pred_cv.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Previsão Perfeita')
ax2.set_title('Valores Reais vs. Preditos (Out-of-Fold)', fontsize=14)
ax2.set_xlabel('Valor Real (Charges)')
ax2.set_ylabel('Valor Predito pelo Modelo')
ax2.legend()

# grafico 3
residuos = y - y_pred_cv
ax3 = axes[1, 0]
sns.histplot(residuos, kde=True, ax=ax3, color='purple', bins=30)
ax3.axvline(x=0, color='red', linestyle='--')
ax3.set_title('Distribuição dos Erros (Resíduos)', fontsize=14)
ax3.set_xlabel('Erro (Real - Predito)')
ax3.set_ylabel('Frequência')

# grafico 4
ax4 = axes[1, 1]

# extrai o modelo do pipeline
modelo_final = pipeline_final.named_steps['regressor']
preprocessor_final = pipeline_final.named_steps['preprocessor']

# tentar obter nomes das features 
try:
    feature_names = preprocessor_final.get_feature_names_out()
    
    # Random Forest e XGBoost têm atributo feature_importances_
    if hasattr(modelo_final, 'feature_importances_'):
        importances = modelo_final.feature_importances_
        
        # dataFrame para plotar
        df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_importance = df_importance.sort_values(by='Importance', ascending=False).head(10) # Top 10
        
        sns.barplot(x='Importance', y='Feature', data=df_importance, ax=ax4, palette='viridis')
        ax4.set_title('Top 10 Variáveis Mais Importantes', fontsize=14)
        ax4.set_xlabel('Importância Relativa')
        ax4.set_ylabel('')
    else:
        ax4.text(0.5, 0.5, 'Modelo não possui feature_importances_', ha='center')
        
except Exception as e:
    ax4.text(0.5, 0.5, f'Não foi possível extrair features:\n{e}', ha='center')
    print("Erro ao extrair features:", e)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # ajuste do titulo
plt.show()

# salvar a figura 
fig.savefig('resultados_modelo_final.png', dpi=300)
print("Gráfico salvo como 'resultados_modelo_final.png'")