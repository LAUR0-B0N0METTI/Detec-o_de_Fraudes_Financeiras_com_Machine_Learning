```markdown
# 🕵️ Sistema de Detecção de Fraudes Financeiras com Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.10.1-red)

Solução completa para identificação de transações fraudulentas utilizando técnicas avançadas de machine learning, com tratamento especial para dados desbalanceados.

## 📌 Visão Geral

Este projeto aborda o desafio real de detectar fraudes em transações financeiras, onde apenas **1.8% dos dados** representam casos fraudulentos. Desenvolvemos uma solução robusta que:

- Combina **Random Forest** e **XGBoost** para máxima eficácia
- Utiliza **SMOTE otimizado** para tratamento de desbalanceamento
- Implementa **ajuste dinâmico de thresholds** para diferentes cenários
- Oferece métricas completas incluindo **AUC-PR** e **F1-Score**

## 📊 Resultados Destacados

| Modelo       | Precision (Fraude) | Recall (Fraude) | F1-Score | AUC-ROC | AUC-PR |
|--------------|--------------------|-----------------|----------|---------|--------|
| Random Forest| 0.30               | 0.73            | 0.42     | 0.9679  | 0.5051 |
| XGBoost      | 0.33               | 0.79            | 0.46     | 0.9767  | 0.6541 |

![Curvas de Avaliação](https://i.imgur.com/JtQm3rP.png)



## ⚙️ Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### 1. Geração de Dados Sintéticos
```python
python src/generate_data.py --samples 50000 --features 12 --fraud_ratio 0.018
```

### 2. Treinamento dos Modelos
```bash
python src/train.py \
  --model all \          # Treina ambos modelos
  --smote_strategy 0.15 \
  --test_size 0.3
```

### 3. Avaliação
```bash
python src/evaluate.py \
  --model_path models/xgboost_model.pkl \
  --threshold optimal   # Pode ser '0.5' ou um valor específico
```

## 🔍 Detalhes Técnicos

### Pré-processamento Avançado
```python
# SMOTE com ajuste fino
smote = SMOTE(
    sampling_strategy=0.15,
    k_neighbors=5,
    random_state=42
)

# Pipeline completo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])
```

### Arquitetura dos Modelos
**Random Forest Otimizado:**
```python
RandomForestClassifier(
    class_weight={0:1, 1:8},
    max_depth=7,
    min_samples_leaf=20,
    n_estimators=150
)
```

**XGBoost com Regularização:**
```python
XGBClassifier(
    scale_pos_weight=7,
    max_depth=3,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    eval_metric='aucpr'
)
```

## 📈 Performance em Diferentes Thresholds

| Threshold | Precision | Recall | Transações Flagadas |
|-----------|-----------|--------|---------------------|
| 0.5       | 0.18      | 0.88   | 4.2%                |
| 0.73      | 0.30      | 0.73   | 2.5%                | 
| 0.95      | 0.50      | 0.60   | 1.1%                |

## 🚀 Como Implementar em Produção

1. **Para Sistemas de Alerta:**
```python
# Carregar modelo
model = joblib.load('models/xgboost_model.pkl')

# Prever com threshold ajustado
probs = model.predict_proba(new_transactions)[:,1]
fraud_flags = (probs > 0.73).astype(int)
```

2. **Monitoramento Contínuo:**
```python
# Verificar drift de dados
from scipy import stats
if stats.ks_2samp(train_probs, new_probs).pvalue < 0.01:
    print("ALERTA: Mudança significativa na distribuição!")
```

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie sua branch (`git checkout -b feature/nova-melhorias`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova métrica'`)
4. Push para a branch (`git push origin feature/nova-melhorias`)
5. Abra um Pull Request

## 📜 Licença

Distribuído sob licença MIT. Veja `LICENSE` para mais informações.

---

**Contato:**</br>
[Lauro Bonometti] </br>
lauro.f.bonometti@gmail.com