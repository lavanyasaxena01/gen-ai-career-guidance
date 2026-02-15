%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("ai_job_dataset.csv")
df['required_skills'] = df['required_skills'].str.lower()
df.dropna(inplace=True)

vectorizer = TfidfVectorizer(max_features=50)
skill_matrix = vectorizer.fit_transform(df['required_skills'])

st.title("🚀 CareerSense AI")

student_skills = st.text_input("Enter your skills (comma separated)")

if st.button("Find Career Recommendations"):

    student_vector = vectorizer.transform([student_skills.lower()])
    similarity = cosine_similarity(student_vector, skill_matrix)

    df['similarity_score'] = similarity[0]

    results = df.sort_values(by='similarity_score', ascending=False).head(5)

    st.dataframe(results[['job_title', 'salary_usd']])
