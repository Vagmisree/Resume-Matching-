import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# --- Initialization ---
st.set_page_config(page_title="TalentAlign AI", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Read PDF ---
def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

# --- Extract structured sections from Resume/JD ---
def extract_sections(text):
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
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

# --- Auto-classify JD if not structured ---
def classify_jd_sections(jd_text):
    prompt = f"""
The following job description is unstructured. Extract and return content under 'skills', 'experience', and 'education'.

Return only JSON like:
{{
  "skills": "...",
  "experience": "...",
  "education": "..."
}}

Text:
{jd_text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        st.text("🧠 Auto-classified JD Sections:\n" + raw)
        return json.loads(raw)
    except Exception as e:
        st.error(f"JD classification failed: {e}")
        return {"skills": "", "experience": "", "education": ""}

# --- Expand abbreviations using GPT ---
def expand_abbreviations(text):
    if not text.strip(): return ""
    prompt = f"Expand all abbreviations and acronyms in this text:\n{text}\nReturn only the expanded text."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# --- Expand skills semantically using GPT ---
def expand_skills(text):
    if not text.strip():
        return ""
    prompt = f"""
Given the following list of skills, technologies, or frameworks, expand it by including related or commonly associated ones.

Input: {text}

Return an expanded, comma-separated list only. Do not add explanations.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠️ Failed to expand skills: {e}")
        return text

# --- Extract skill keywords from JD if unstructured ---
def extract_skill_keywords(text):
    if not text.strip():
        return ""
    prompt = f"""
Extract a clean, comma-separated list of technical skills, tools, programming languages, frameworks, or certifications from this job description:

"{text}"

Only return comma-separated keywords like: Java, Spring, Hibernate, SQL, Docker, etc.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠️ Failed to extract skills: {e}")
        return text

# --- Cosine similarity for skills ---
def compute_cosine_similarity(text1, text2):
    emb1 = model.encode([text1])[0].reshape(1, -1)
    emb2 = model.encode([text2])[0].reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

# --- Requirement satisfaction GPT scoring ---
def check_requirement(resume_section, jd_section, section_name, min_score=0.2):
    # Check if education section is missing in JD
    if section_name.lower() == "education" and not jd_section.strip():
        st.info("📘 JD does not specify education requirements. Assuming education match is perfect.")
        return 1.0

    if section_name.lower() == "education":
        guidance = """
Score from 0 to 1:
- 1.0: Degree & field match (e.g., BTech in CS vs BS in CS , all allied branches)
- 0.8: Related field (e.g., IT, CSBS)
- 0.5: Technical but different stream (e.g., ECE, EE)
- 0.2: Unrelated or vague education
- 0.0: No degree or info provided
"""
    elif section_name.lower() == "experience":
        guidance = """
Score from 0 to 1:
- 1.0: Direct match (same domain, years, tech stack and the JD does not specify minimum experience then rate it as 1)
- 0.5: Technical experience but not exact stack
- 0.2: Some tech or software background
- 0.0: No relevant experience at all
"""
    else:
        guidance = """
Score from 0 to 1:
- 1.0: Strong overlap of skills
- 0.7: Partial overlap (2–3 skills)
- 0.3: 1 common skill
- 0.1: Totally different
- 0.0: No skills provided
"""

    prompt = f"""
You are a job-matching evaluator. Based on the job description and candidate's {section_name}, give a score 0 to 1 showing how well it fits.

{guidance}

Job {section_name}: {jd_section}
Candidate {section_name}: {resume_section}

Return only JSON like: {{"score": 0.7}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        st.text(f"Raw GPT Response for {section_name}:\n" + raw)
        score = float(json.loads(raw)["score"])
        return max(score, min_score)
    except Exception as e:
        st.warning(f"⚠️ Failed to parse GPT score for '{section_name}': {e}. Using fallback {min_score}.")
        return min_score


# --- Flatten section values ---
def flatten_section(section_data):
    return " ".join(str(v) for v in section_data.values()) if isinstance(section_data, dict) else str(section_data)

# --- UI ---
st.title("🧠 TalentAlign AI: Resume & JD Matching")

resume_file = st.file_uploader("📄 Upload Resume PDF", type="pdf")
jd_file = st.file_uploader("📃 Upload Job Description PDF", type="pdf")

if resume_file and jd_file:
    resume_text = read_pdf(resume_file)
    jd_text = read_pdf(jd_file)

    st.subheader("🔍 Extracting Sections...")
    resume_data = extract_sections(resume_text)

    jd_data = extract_sections(jd_text)
    if not jd_data or not any(jd_data.values()):
        jd_data = classify_jd_sections(jd_text)

    st.markdown("### 🧑‍💼 Resume Sections")
    st.json(resume_data)

    st.markdown("### 📋 Job Description Sections")
    st.json(jd_data)

    st.subheader("📊 Similarity & Satisfaction Scores")
    scores = {}
    weights = {"skills": 0.8, "experience": 0.1, "education": 0.1}

    # Skills: cosine similarity with GPT expansion
    skill_res_raw = flatten_section(resume_data.get("skills", ""))
    skill_jd_raw = flatten_section(jd_data.get("skills", ""))
    skill_jd_keywords = extract_skill_keywords(skill_jd_raw)
    expanded_res_skills = expand_skills(skill_res_raw)
    expanded_jd_skills = expand_skills(skill_jd_keywords)

    if expanded_res_skills and expanded_jd_skills:
        sim_score = compute_cosine_similarity(expanded_res_skills, expanded_jd_skills)
        scores["skills"] = sim_score
        st.success(f"Skills Similarity (expanded): {sim_score:.2f}")
        st.caption(f"🔎 Expanded Resume Skills: {expanded_res_skills}")
        st.caption(f"🔎 Expanded JD Skills: {expanded_jd_skills}")
    else:
        st.warning("⚠️ Missing 'skills' content in one of the files.")

    # Experience
    exp_res = expand_abbreviations(flatten_section(resume_data.get("experience", "")))
    exp_jd = expand_abbreviations(flatten_section(jd_data.get("experience", "")))
    if exp_res and exp_jd:
        exp_score = check_requirement(exp_res, exp_jd, "experience")
        scores["experience"] = exp_score
        st.success(f"Experience Satisfaction: {exp_score:.2f}")
    else:
        st.warning("⚠️ Missing 'experience' content in one of the files.")

    # Education
    edu_res = expand_abbreviations(flatten_section(resume_data.get("education", "")))
    edu_jd = expand_abbreviations(flatten_section(jd_data.get("education", "")))
    if edu_res and edu_jd:
        edu_score = check_requirement(edu_res, edu_jd, "education")
        scores["education"] = edu_score
        st.success(f"Education Satisfaction: {edu_score:.2f}")
    else:
        st.warning("⚠️ Missing 'education' content in one of the files.")

    # --- Final Score ---
    final_score = sum(weights[s] * scores.get(s, 0) for s in weights)
    st.subheader("📌 Final Resume-JD Match Score")
    st.success(f"✅ Weighted Match: **{final_score * 100:.2f}%**")
else:
    st.info("⬆️ Please upload both Resume and Job Description PDFs to begin.")
