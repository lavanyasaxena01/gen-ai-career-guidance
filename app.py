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

import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_ai_roadmap(career, skills):
    prompt = f"""
    Create a 3-year structured roadmap to become a {career}.
    Include:
    - Beginner phase
    - Intermediate phase
    - Advanced phase
    - Tools to learn
    - Certifications
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
roadmap = generate_ai_roadmap(row['job_title'], row['required_skills'])
st.markdown(roadmap)
