import streamlit as st
import pandas as pd
import pickle

from data_loader import load_all
from preprocess import preprocess_applicants
from feature_engineering import extract_skills, vectorize_resumes, job_applicant_matrix
from model import train_model

st.title("Recomendador de Candidatos")
data = load_all('../data')
vagas_df = data['vagas']
applicants_df = preprocess_applicants(data['applicants'])

# 1. Selecionar vaga
job_id = st.sidebar.selectbox("Escolha a vaga", vagas_df['job_id'])
vaga = vagas_df[vagas_df['job_id']==job_id].iloc[0]
st.write("**Descrição da vaga:**", vaga['descricao'])

# 2. Carregar modelo / vetorizer (ou treinar no fly)
if st.sidebar.checkbox("Treinar modelo agora"):
    # Exemplo simplificado: label y baseado em history
    pros_df = data['prospects']
    merged = pros_df.merge(applicants_df, on='candidate_id')
    X, vect = vectorize_resumes(merged)
    y = merged['status'].apply(lambda s: 1 if s=='hired' else 0)
    model = train_model(X.toarray(), y)
    st.success("Modelo treinado!")

else:
    model = pickle.load(open('model.pkl','rb'))
    vect = pickle.load(open('vectorizer.pkl','rb'))

# 3. Perguntas específicas para refinamento
st.sidebar.subheader("Refine por Skills")
skills = ["python","aws","docker","kubernetes","terraform"]
chosen = st.sidebar.multiselect("Quais skills são obrigatórias?", skills)

# 4. Cálculo de similaridade
sim_matrix = job_applicant_matrix(vagas_df, applicants_df, vect)
scores = sim_matrix[:, vagas_df.index[vagas_df['job_id']==job_id][0]]

# 5. Filtrar por skills
mask = applicants_df[[f'skill_{s}' for s in chosen]].all(axis=1) if chosen else [True]*len(applicants_df)
result = applicants_df[mask].copy()
result['match_score'] = scores[mask]

st.write("### Top candidatos")
st.dataframe(result.sort_values('match_score', ascending=False)[['candidate_id','nome','match_score']].head(10))

# 6. Explicação de why (features mais relevantes)
st.write("Use as setas ao lado do candidato para ver detalhes do currículo e justificar o score.")
