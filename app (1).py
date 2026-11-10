
import streamlit as st
import re
import io

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer")
st.write("Analyze your resume against a job description using NLP.")


skills_database = [
    "python", "java", "c++", "machine learning",
    "deep learning", "sql", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "nlp", "data analysis", "excel",
    "power bi", "tableau", "flask", "fastapi",
    "html", "css", "javascript", "react",
    "docker", "git"
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in skills_database:
        pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'

        if re.search(pattern, text):
            found.append(skill)

    return found


def analyze_resume(resume_text, job_description):

    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_description)

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform([
        clean_resume,
        clean_job
    ])

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0] * 100

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched = [
        skill for skill in job_skills
        if skill in resume_skills
    ]

    missing = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    skill_score = (
        len(matched) / len(job_skills) * 100
        if job_skills else 0
    )

    return similarity, skill_score, matched, missing


uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the Job Description",
    height=200
)


if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        try:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

            resume_text = ""

            for page in reader.pages:
                resume_text += page.extract_text() or ""

            if not resume_text.strip():
                st.error("No readable text found in this PDF.")
            else:
                similarity, skill_score, matched, missing = (
                    analyze_resume(resume_text, job_description)
                )

                st.subheader("Analysis Results")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Resume Similarity",
                    f"{similarity:.2f}%"
                )

                col2.metric(
                    "Skill Match",
                    f"{skill_score:.2f}%"
                )

                st.subheader("Matched Skills")
                st.write(matched)

                st.subheader("Missing Skills")
                st.write(missing)

                st.info(
                    "These scores are text and keyword similarity "
                    "indicators, not hiring probabilities."
                )

        except Exception as e:
            st.error(f"Analysis failed: {e}")
