# IA para Otimização de Recrutamento – Datathon POSTECH & Decision

Este projeto foi desenvolvido para o desafio Datathon Fase 5, com o objetivo de criar uma **Inteligência Artificial que simula o trabalho de um entrevistador técnico**, auxiliando hunters a priorizar os melhores candidatos com base em seu currículo.

## 📂 Estrutura dos Dados

O projeto utiliza três bases principais em formato JSON:

- `vagas.json` – Informações sobre as vagas, como cliente, requisitos técnicos e idioma.
- `prospects.json` – Situação dos candidatos em cada vaga (ex: contratado, encaminhado, desistente).
- `applicants.json` – Dados dos candidatos, como formação, conhecimentos técnicos e CV completo.

## 🧠 Modelo de Machine Learning

Utilizamos um modelo de classificação `LogisticRegression` treinado com:

- `tecnologias_mencionadas` no CV
- `nível técnico` (derivado da quantidade de tecnologias)
  
Essas features são ponderadas e combinadas para prever a **chance de contratação** com base em padrões históricos da Decision.

## 💻 Aplicativo Streamlit

O app coleta informações via perguntas guiadas, como:

- Cargo atual  
- Experiência técnica  
- Ferramentas dominadas  
- Certificações, inglês e formação  

E então estima a chance de contratação do candidato com base em seus atributos técnicos.

## 🏁 Execução

```bash
# Treinar o modelo
python scripts/train_vectorizer_and_model.py

# Iniciar o app
streamlit run app/app.py
