import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("ai_job_dataset.csv")
df['required_skills'] = df['required_skills'].str.lower()

# -------------------------------
# TF-IDF Setup
# -------------------------------
vectorizer = TfidfVectorizer(max_features=50)
skill_matrix = vectorizer.fit_transform(df['required_skills'])

# -------------------------------
# Roadmap Generator (Simple Version)
# -------------------------------
def generate_roadmap(career, skills):
    return f"""
### 📍 Roadmap for {career}

🟢 **Beginner Phase**
- Learn fundamentals
- Study: {skills}
- Complete online courses

🟡 **Intermediate Phase**
- Build real-world projects
- Contribute to GitHub
- Apply for internships

🔵 **Advanced Phase**
- Specialize in advanced tools
- Prepare for technical interviews
- Target high-paying companies
"""

# -------------------------------
# Streamlit UI
# -------------------------------
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

            roadmap = generate_roadmap(
                row['job_title'],
                row['required_skills']
            )

            st.markdown(roadmap)
            st.divider()
