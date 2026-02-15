import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("ai_job_dataset.csv")
df['required_skills'] = df['required_skills'].str.lower()

# -----------------------------
# TF-IDF Setup
# -----------------------------
vectorizer = TfidfVectorizer(max_features=50)
skill_matrix = vectorizer.fit_transform(df['required_skills'])

# -----------------------------
# Structured Roadmap Generator
# -----------------------------
def generate_structured_roadmap():

    return {
        "Beginner": [
            ("Python Basics", "https://www.w3schools.com/python/"),
            ("Intro to ML", "https://www.coursera.org/learn/machine-learning"),
            ("Git & GitHub", "https://www.freecodecamp.org/news/git-and-github-for-beginners/")
        ],
        "Intermediate": [
            ("Build Projects", "https://www.kaggle.com/learn"),
            ("Data Structures", "https://www.geeksforgeeks.org/data-structures/"),
            ("Scikit-Learn", "https://scikit-learn.org/stable/")
        ],
        "Advanced": [
            ("Deep Learning", "https://www.deeplearning.ai/"),
            ("System Design", "https://github.com/donnemartin/system-design-primer"),
            ("MLOps", "https://mlops.community/")
        ]
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚀 CareerSense AI")

student_skills = st.text_input(
    "Enter your skills (comma separated)",
    key="skills_input"
)

if st.button("Find Career Recommendations"):

    if student_skills.strip() == "":
        st.warning("Please enter at least one skill.")
    else:
        student_vector = vectorizer.transform([student_skills.lower()])
        similarity = cosine_similarity(student_vector, skill_matrix)

        df['similarity_score'] = similarity[0]

        results = df.sort_values(
            by='similarity_score',
            ascending=False
        ).head(3)

        for index, row in results.iterrows():

            st.subheader(row['job_title'])
            st.write("💰 Salary:", row['salary_usd'])

            roadmap = generate_structured_roadmap()

            for phase, resources in roadmap.items():

                with st.expander(f"{phase} Phase"):

                    for title, link in resources:
                        st.markdown(f"- [{title}]({link})")

            st.divider()
