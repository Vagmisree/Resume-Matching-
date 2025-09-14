import os
import json
import csv
import logging
import pdfplumber
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load API key securely from environment
api_key = "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"
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment variables.")

client = OpenAI(api_key=api_key)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utilities ---

def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception as e:
        logging.error(f"Error reading PDF {file_path}: {e}")
        return ""

def chat(prompt, temperature=0):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return ""

def extract_sections(text):
    if not text.strip():
        return {"skills": "", "experience": "", "education": ""}

    prompt = f"""
Extract the following sections from the text in JSON format with keys 'skills', 'experience', and 'education'.

Text:
{text}

Only return JSON. Do not add explanations or markdown. Use empty string if any section is missing.

Expected format:
{{
  "skills": "...",
  "experience": "...",
  "education": "..."
}}
"""
    try:
        return json.loads(chat(prompt))
    except Exception as e:
        logging.warning(f"Failed to extract JSON sections: {e}")
        return {"skills": "", "experience": "", "education": ""}

def expand_abbreviations(text):
    if not text.strip():
        return ""
    prompt = f"Expand all abbreviations and acronyms in this text:\n{text}\nReturn only the expanded text."
    return chat(prompt, temperature=0.2)

def extract_skill_keywords(text):
    if not text.strip():
        return ""
    prompt = f"""
Extract a clean, comma-separated list of technical skills, tools, programming languages, frameworks, or certifications from this job description:

"{text}"

Only return comma-separated keywords like: Java, Spring, Hibernate, SQL, Docker, etc.
"""
    return chat(prompt)

def expand_skills(text):
    if not text.strip():
        return ""
    prompt = f"""
Given the following list of skills, technologies, or frameworks, expand it by including related or commonly associated ones.

Input: {text}

Return an expanded, comma-separated list only. Do not add explanations.
"""
    return chat(prompt, temperature=0.2)

def compute_cosine_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = model.encode([text1.lower().strip()])[0].reshape(1, -1)
    emb2 = model.encode([text2.lower().strip()])[0].reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

def flatten_section(section_data):
    return " ".join(str(v) for v in section_data.values()) if isinstance(section_data, dict) else str(section_data)

def check_requirement(resume_section, jd_section, section_name, min_score=0.2):
    guidance = ""
    if section_name.lower() == "education":
        guidance = """
Score from 0 to 1:
- 1.0: Degree & field match
- 0.8: Related field
- 0.5: Technical but different stream
- 0.2: Unrelated
- 0.0: No info
"""
    elif section_name.lower() == "experience":
        guidance = """
Score from 0 to 1:
- 1.0: Direct match
- 0.5: Technical but different
- 0.2: Some background
- 0.0: No match
"""
    else:
        guidance = """
Score from 0 to 1:
- 1.0: Strong overlap
- 0.7: Partial overlap
- 0.3: 1 common skill
- 0.1: Totally different
- 0.0: No skills
"""

    prompt = f"""
You are a job-matching evaluator. Based on the job description and candidate's {section_name}, give a score 0 to 1.

{guidance}

Job {section_name}: {jd_section}
Candidate {section_name}: {resume_section}

Return only JSON like: {{"score": 0.7}}
"""
    try:
        result = chat(prompt)
        return float(json.loads(result)["score"])
    except Exception as e:
        logging.warning(f"GPT scoring failed for {section_name}: {e}")
        return min_score

# --- Main Process ---

def process(resume_dir, jd_dir, output_csv):
    weights = {"skills": 0.8, "experience": 0.1, "education": 0.1}
    results = [("Resume", "JD", "Skills", "Experience", "Education", "Match %")]

    for resume_file in os.listdir(resume_dir):
        if not resume_file.lower().endswith(".pdf"):
            continue
        resume_path = os.path.join(resume_dir, resume_file)
        resume_text = read_pdf(resume_path)
        if not resume_text:
            logging.warning(f"Skipping empty resume: {resume_file}")
            continue
        resume_data = extract_sections(resume_text)

        for jd_file in os.listdir(jd_dir):
            if not jd_file.lower().endswith(".pdf"):
                continue
            jd_path = os.path.join(jd_dir, jd_file)
            jd_text = read_pdf(jd_path)
            if not jd_text:
                logging.warning(f"Skipping empty JD: {jd_file}")
                continue
            jd_data = extract_sections(jd_text)

            # Skills
            res_skills = flatten_section(resume_data.get("skills", ""))
            jd_skills = flatten_section(jd_data.get("skills", ""))
            jd_keywords = extract_skill_keywords(jd_skills)
            expanded_res_skills = expand_skills(res_skills)
            expanded_jd_skills = expand_skills(jd_keywords)
            skill_score = compute_cosine_similarity(expanded_res_skills, expanded_jd_skills)

            # Experience
            res_exp = expand_abbreviations(flatten_section(resume_data.get("experience", "")))
            jd_exp = expand_abbreviations(flatten_section(jd_data.get("experience", "")))
            exp_score = check_requirement(res_exp, jd_exp, "experience")

            # Education
            res_edu = expand_abbreviations(flatten_section(resume_data.get("education", "")))
            jd_edu = expand_abbreviations(flatten_section(jd_data.get("education", "")))
            edu_score = check_requirement(res_edu, jd_edu, "education")

            final_score = (
                weights["skills"] * skill_score +
                weights["experience"] * exp_score +
                weights["education"] * edu_score
            )

            logging.info(f"Matched {resume_file} to {jd_file}: {round(final_score * 100, 2)}%")

            results.append((
                resume_file,
                jd_file,
                round(skill_score, 2),
                round(exp_score, 2),
                round(edu_score, 2),
                round(final_score * 100, 2)
            ))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    logging.info(f"✅ Results saved to {output_csv}")

# --- Entry Point ---
if __name__ == "__main__":
    process("Resumes", "JDs", "results.csv")
