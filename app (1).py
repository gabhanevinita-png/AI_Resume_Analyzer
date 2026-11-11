
import streamlit as st
import pandas as pd
import re
import io

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer")
st.write(
    "Analyze your resume against a job description "
    "using NLP and machine learning."
)

st.divider()


# ---------------- SKILL DATABASE ----------------

skills_database = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "deep neural networks"],
    "sql": ["sql", "structured query language"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch", "torch"],
    "nlp": ["nlp", "natural language processing"],
    "data analysis": ["data analysis", "data analytics"],
    "excel": ["excel", "microsoft excel"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs"],
    "docker": ["docker"],
    "git": ["git"],
    "github": ["github"]
}


# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- SKILL EXTRACTION ----------------

def extract_skills(text):

    text = text.lower()
    found = []

    for skill, variations in skills_database.items():

        for variation in variations:

            pattern = (
                r'(?<!\w)' +
                re.escape(variation) +
                r'(?!\w)'
            )

            if re.search(pattern, text):
                found.append(skill)
                break

    return found


# ---------------- RESUME ANALYSIS ----------------

def analyze_resume(resume_text, job_description):

    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_description)

    if not clean_resume or not clean_job:
        raise ValueError("Resume or job description has no readable text.")

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

    return similarity, skill_score, matched, missing, job_skills


# ---------------- USER INPUT ----------------

st.header("📄 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume in PDF format",
    type=["pdf"]
)

st.header("💼 Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=220,
    placeholder="Paste the complete job description..."
)


# ---------------- ANALYZE BUTTON ----------------

if st.button("🚀 Analyze Resume", type="primary"):

    if uploaded_file is None:

        st.warning("Please upload your resume PDF.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        try:

            reader = PdfReader(
                io.BytesIO(uploaded_file.getvalue())
            )

            resume_text = ""

            for page in reader.pages:
                resume_text += page.extract_text() or ""

            if not resume_text.strip():

                st.error(
                    "No readable text found. "
                    "This may be a scanned PDF."
                )

            else:

                (
                    similarity,
                    skill_score,
                    matched,
                    missing,
                    job_skills
                ) = analyze_resume(
                    resume_text,
                    job_description
                )

                # ---------------- RESULTS ----------------

                st.divider()
                st.header("📊 Analysis Results")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Resume Similarity",
                    f"{similarity:.2f}%"
                )

                col2.metric(
                    "Skill Match",
                    f"{skill_score:.2f}%"
                )

                # ---------------- SKILL RESULTS ----------------

                st.subheader("✅ Matched Skills")

                if matched:
                    st.write(", ".join(matched))
                else:
                    st.info("No matching skills found.")

                st.subheader("❌ Missing Skills")

                if missing:
                    st.write(", ".join(missing))
                else:
                    st.success("No missing skills found!")

                # ---------------- DASHBOARD ----------------

                st.divider()
                st.subheader("📈 Skill Match Dashboard")

                chart_data = pd.DataFrame({
                    "Skill Status": [
                        "Matched Skills",
                        "Missing Skills"
                    ],
                    "Number of Skills": [
                        len(matched),
                        len(missing)
                    ]
                })

                st.bar_chart(
                    chart_data,
                    x="Skill Status",
                    y="Number of Skills"
                )

                st.write(
                    f"Total required skills: {len(job_skills)}"
                )

                st.write(
                    f"Total matched skills: {len(matched)}"
                )

                st.write(
                    f"Total missing skills: {len(missing)}"
                )

                st.progress(
                    int(skill_score)
                )

                # ---------------- DISCLAIMER ----------------

                st.info(
                    "Resume similarity and skill match are "
                    "text-based indicators. They are not "
                    "hiring probabilities or guarantees."
                )

        except Exception as e:

            st.error(f"Analysis failed: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Built with Python, NLP, Scikit-learn and Streamlit | "
    "AI Resume Analyzer"
)
