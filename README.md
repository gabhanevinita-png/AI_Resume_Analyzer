# AI Resume Analyzer

An NLP-based application that analyzes a candidate's resume against a job description.

## Features
- PDF resume text extraction
- Text preprocessing
- TF-IDF vectorization
- Cosine similarity
- Skill extraction
- Matched and missing skill identification
- Resume analysis report

## Technologies
- Python
- Streamlit
- Scikit-learn
- PyPDF
- Pandas

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone  https://github.com/gabhanevinita-png/AI_Resume_Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```text
AI-Resume-Analyzer/
├── app.py
├── requirements.txt
└── README.md
```

## How It Works

The application extracts text from a PDF resume, compares it with a job description using TF-IDF and cosine similarity, and identifies matched and missing skills using keyword matching.

**Note:** Similarity and skill scores are indicators, not hiring probabilities.

## Author

Vignesh Gabhane

Computer Science Engineering – Artificial Intelligence and Machine Learning

[GitHub](https://github.com/gabhanevinita-png) 
