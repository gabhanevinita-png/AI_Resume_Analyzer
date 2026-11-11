# 🤖 AI Resume Analyzer

An NLP-powered web application that analyzes resumes against job descriptions using TF-IDF, cosine similarity and automated skill extraction.

## 🚀 Live Demo

[Try AI Resume Analyzer](https://airesumeanalyzer-lffnrpjfdang9srqr46bux.streamlit.app/)

## ✨ Features

- PDF resume upload and text extraction
- Resume and job description similarity scoring
- Automated technical skill extraction
- Improved skill matching with keyword variations
- Identification of matched and missing skills
- Interactive skill-match dashboard
- Visual skill-match progress indicator
- Input validation and error handling

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- PyPDF
- TF-IDF Vectorization
- Cosine Similarity
- Regular Expressions (Regex)

## ⚙️ How It Works

1. Upload a resume in PDF format.
2. Paste the target job description.
3. Extract resume text using PyPDF.
4. Convert text into TF-IDF vectors.
5. Calculate cosine similarity.
6. Extract and compare technical skills.
7. Display matched skills, missing skills and visual results.

## 📂 Project Structure

```text
AI-Resume-Analyzer/
│
├── app.py
├── requirements.txt
└── README.md
```

## 💻 Run Locally

Clone the repository:

```bash
git clone 
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## ⚠️ Limitations

- Scanned image-based PDFs may require OCR.
- Skill matching depends on the configured skill dictionary.
- Similarity scores are text-based indicators, not hiring probabilities.
- The application does not independently verify candidate proficiency.

## 👨‍💻 Author

**Vignesh Gabhane**

Computer Science Engineering (AI & ML)

- GitHub: https://github.com/gabhanevinita-png
- LinkedIn: https://www.linkedin.com/in/vignesh-gabhane-8a521b3a6?utm_source=share_via&utm_content=profile&utm_medium=member_android

---

Built as an AI/ML portfolio project to explore NLP and practical machine learning applications.
