# Credit Limit Prediction

## 📌 Objetivo

Desenvolver um modelo de regressão capaz de prever o **limite de crédito ideal para clientes**, com base em características financeiras, comportamentais e demográficas.

O objetivo é apoiar decisões de concessão de crédito, equilibrando **risco e rentabilidade**.

---

## 🧠 Problema de Negócio

Instituições financeiras precisam definir **quanto de limite conceder a cada cliente**.

Um limite inadequado pode gerar:

- 📉 Risco elevado de inadimplência (limite alto demais)
- 💸 Perda de oportunidade de receita (limite baixo demais)

Este projeto responde:

- Qual o limite ideal para cada cliente?
- Quais fatores mais influenciam essa decisão?
- Como otimizar concessão de crédito com base em dados?

---

## 📊 Dataset

O dataset contém informações financeiras e demográficas de clientes.

### 📌 Descrição das Variáveis

- **Income**
  Renda do cliente.

- **Limit**
  Limite de crédito concedido ao cliente (**variável alvo**).

- **Rating**
  Score de crédito do cliente.

- **Cards**
  Número de cartões de crédito que o cliente possui.

- **Age**
  Idade do cliente.

- **Education**
  Anos de educação do cliente.

- **Gender**
  Gênero do cliente.

- **Student**
  Indica se o cliente é estudante.

- **Married**
  Indica se o cliente é casado.

---

## ⚙️ Metodologia

### 1. Análise Exploratória (EDA)

- Distribuição das variáveis
- Identificação de outliers
- Relação entre renda, rating e limite
- Análise de correlação

---

### 2. Pré-processamento

- Tratamento de valores faltantes
- Encoding de variáveis categóricas:
  - Gender
  - Student
  - Married

- Normalização (quando necessário)

---

### 3. Engenharia de Features

Exemplos:

- Relação entre renda e rating
- Quantidade de cartões vs limite
- Faixas de renda

---

### 4. Modelagem

Modelos utilizados:

- Regressão Linear (baseline)
- OLS

---

### 5. Avaliação

Métricas utilizadas:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 💡 Insights de Negócio

- Clientes com maior **income** e **rating** tendem a possuir limites mais altos
- Número de cartões influencia diretamente o limite concedido
- Perfis específicos (ex: estudantes) apresentam limites mais baixos
- O modelo pode ser utilizado para padronizar decisões de crédito

---

## 🚀 Aplicações

- Definição automática de limite de crédito
- Ajuste de limites existentes
- Estratégias de concessão mais eficientes
- Redução de risco financeiro

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

---
