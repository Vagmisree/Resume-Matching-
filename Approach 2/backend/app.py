from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from openai import OpenAI, AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import asyncio
import io
import traceback
import os
from dotenv import load_dotenv
import httpx
import re
import string
from collections import OrderedDict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import uuid

# --- Load Environment Variables ---
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Client Setup (fixed models) ---
OPENAI_MODEL_FIXED = "gpt-3.5-turbo-1106"
GPT_TEMPERATURE = 0
GPT_SEED = 42

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Core models loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load a core model: {e}")
    client, async_client, embedding_model = None, None, None

QDRANT_COLLECTION_NAME = "RAHUL_jd_collection"

# --- Email Configuration ---
# IMPORTANT: Add these to your .env file
# NOTE: SENDER_PASSWORD must be a 16-digit App Password from Google, not your regular password.
SENDER_EMAIL ="23b01a12a0@svecw.edu.in"
SENDER_PASSWORD ="qelk mobn tlrb drsu" # <-- This password is likely incorrect. Replace with your 16-digit App Password.

# --- In-memory async-safe GPT cache ---
_gpt_cache = {}
_gpt_cache_lock = asyncio.Lock()

async def cached_async_gpt_call(prompt: str, model: str = OPENAI_MODEL_FIXED, response_format=None):
    """
    Async wrapper that caches GPT responses for identical prompts during the server runtime.
    Uses async_client when available, falls back to sync client otherwise.
    """
    key = f"{model}|{GPT_TEMPERATURE}|{GPT_SEED}|{prompt}"
    async with _gpt_cache_lock:
        if key in _gpt_cache:
            return _gpt_cache[key]

    try:
        if async_client is not None:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": GPT_TEMPERATURE,
                "seed": GPT_SEED
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = await async_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()
        else:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": GPT_TEMPERATURE,
                "seed": GPT_SEED
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()

        async with _gpt_cache_lock:
            _gpt_cache[key] = content
        return content
    except Exception as e:
        print(f"⚠️ GPT call failed for prompt (len={len(prompt)}): {e}")
        return ""

# --- Qdrant Search Workaround ---
class ScoredPoint:
    def __init__(self, id, version, score, payload, vector=None, **kwargs):
        self.id = id
        self.version = version
        self.score = score
        self.payload = payload
        self.vector = vector

async def search_qdrant_with_httpx(base_url, api_key, collection_name, vector, limit=10):
    search_url = f"{base_url.rstrip('/')}/collections/{collection_name}/points/search"
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    payload = {"vector": vector, "limit": limit, "with_payload": True}
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            print(f"INFO: Performing direct HTTPX search to {search_url}...")
            response = await http_client.post(search_url, headers=headers, json=payload)
            response.raise_for_status()
            search_results = response.json().get('result', [])
            return [ScoredPoint(**point) for point in search_results]
        except Exception as e:
            print(f"❌ An error occurred during Qdrant search: {e}")
# --- Helper Functions ---
async def qdrant_op_with_httpx(method, endpoint, payload=None):
    """
    Performs an HTTP operation (GET, POST, PUT, DELETE) to Qdrant using httpx.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set in environment variables.")
    url = f"{qdrant_url.rstrip('/')}{endpoint}"
    headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, json=payload)
            elif method.upper() == "PUT":
                response = await client.put(url, headers=headers, json=payload)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers, json=payload)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Qdrant HTTPX operation failed: {e}")
            return {}


# --- Helper Functions ---
def read_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def flatten_section(data):
    if isinstance(data, dict):
        return " ".join(str(v) for v in data.values())
    return str(data) if data is not None else ""

def normalize_text_for_embedding(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.translate(str.maketrans("", "", string.punctuation.replace(",", "")))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_skills(skills_str: str) -> str:
    if not skills_str:
        return ""
    parts = [s.strip().lower() for s in re.split(r'[,\n;]+', skills_str) if s.strip()]
    unique = sorted(set(parts))
    return ", ".join(unique)

def get_embedding(text):
    if embedding_model is None:
        return None
    try:
        norm = normalize_text_for_embedding(text)
        return embedding_model.encode(norm, convert_to_tensor=False).tolist()
    except Exception as e:
        print(f"Failed to generate SentenceTransformer embedding: {e}")
        return None

# --- Section extraction (cached) ---
async def extract_sections_async(text, doc_type="Resume"):
    prompt = f"""
Analyze the following {doc_type} text and extract the content for 'skills', 'experience', and 'education'.
Return the extracted information in a clean JSON format. If a section is not found, its value should be an empty string.
Text:
{text}
Expected format: {{"skills": "...", "experience": "...", "education": "..."}}
"""
    try:
        raw = await cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED, response_format={"type": "json_object"})
        parsed = json.loads(raw) if raw else {"skills": "", "experience": "", "education": ""}
        parsed["skills"] = normalize_skills(parsed.get("skills", ""))
        parsed["experience"] = parsed.get("experience", "") or ""
        parsed["education"] = parsed.get("education", "") or ""
        return parsed
    except Exception as e:
        print(f"AI failed to extract sections for {doc_type}: {e}")
        return {"skills": "", "experience": "", "education": ""}

def summarize_resume_for_search(resume_data):
    prompt = f"""
Create a job board search query string from the following resume data.
The query should contain the most prominent job title and the top 5-10 skills and technologies.

Skills: {flatten_section(resume_data.get('skills', ''))}
Experience: {flatten_section(resume_data.get('experience', ''))}

Return ONLY the query string. Do not add any explanation, preamble, or markdown formatting.
Example Output: Front-End Developer with React, Node.js, and JavaScript
Example Output: Intern with Python, Machine Learning, and MongoDB
"""
    try:
        # NOTE: This runs asyncio in a separate event loop, which is suitable for a sync function.
        raw = asyncio.run(cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED))
        summary = raw.strip()
        print(f"Generated Search Query: {summary}")
        return summary
    except Exception as e:
        print(f"Could not summarize resume for search, using broader query. Error: {e}")
        return f"Skills: {resume_data.get('skills', '')}. Experience: {resume_data.get('experience', '')}"

# --- Email Sending Function ---
def send_email(to_email, matched_jobs):
    """
    Sends an email with the matched job results.
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("⚠️ Email credentials (SENDER_EMAIL, SENDER_PASSWORD) not set. Skipping email.")
        return False
        
    try:
        subject = "Your AI-Powered Job Matches from TalentAlign"

        body_html = """
<html>
<head>
  <style>
    body {{
      font-family: Arial, sans-serif;
      line-height: 1.6;
      color: #333;
      background-color: #f4f6f8;
      margin: 0;
      padding: 0;
    }}
    .container {{
      max-width: 600px;
      margin: 30px auto;
      background-color: #ffffff;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      border: 1px solid #e0e0e0;
    }}
    .header {{
      background: linear-gradient(90deg, #E53935, #D32F2F);
      color: #ffffff;
      padding: 20px;
      text-align: center;
    }}
    .header h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: bold;
    }}
    .header p {{
      margin: 5px 0 0;
      font-size: 14px;
      opacity: 0.9;
    }}
    .content {{
      padding: 20px;
    }}
    .job {{
      background-color: #fafafa;
      border: 1px solid #eee;
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 15px;
    }}
    .job h3 {{
      margin: 0 0 8px 0;
      color: #1E88E5;
      font-size: 18px;
    }}
    .score {{
      display: inline-block;
      padding: 4px 10px;
      background-color: #E8F5E9;
      color: #2E7D32;
      font-weight: bold;
      border-radius: 20px;
      font-size: 14px;
    }}
    .footer {{
      background-color: #f8f9fa;
      text-align: center;
      padding: 15px;
      font-size: 12px;
      color: #777;
      border-top: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>TalentAlign AI</h1>
      <p>Your Personalized Job Match Results</p>
    </div>
    <div class="content">
      <p>Dear User,</p>
      <p>Based on your uploaded resume, here are the top job opportunities we've matched for you:</p>
"""

        for i, job in enumerate(matched_jobs, 1):
            title = job['payload'].get('title', 'No Title')
            score = job['score']
            body_html += f"""
      <div class="job">
        <h3>{i}. {title}</h3>
        <span class="score">{(score * 100):.1f}% Match</span>
      </div>
    """

        body_html += """
      <p>We’re constantly improving our matches to help you find the best opportunities faster.</p>
    </div>
    <div class="footer">
      <p>&copy; 2025 TalentAlign AI. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
"""


        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body_html, 'html'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        
        print(f"✅ Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        traceback.print_exc()
        return False

# --- SCORING LOGIC (remains the same) ---
async def expand_abbreviations_async(text):
    if not text or not text.strip():
        return ""
    prompt = f"Expand all abbreviations and acronyms in this text:\n{text}\nReturn only the expanded text."
    return await cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED)

async def expand_skills_async(text):
    if not text or not text.strip():
        return ""
    prompt = f"""
Given the following list of skills, technologies, or frameworks, expand it by including related or commonly associated ones.
Input: {text}

IMPORTANT: Normalize skill names to their base forms:
- Use "php" instead of "php7", "php8"
- Use "css" instead of "css3", "css4"  
- Use "html" instead of "html5"
- Use "javascript" instead of "js", "es6", "es7"
- Use "react" instead of "reactjs", "react.js"
- Use "node" instead of "nodejs", "node.js"
- Use "python" instead of "python3"
- Use "java" instead of "java8", "java11", "java17"
- Use "sql" for database skills (mysql, postgresql, oracle)
- Use "git" for version control (github, gitlab)

Return an expanded, comma-separated list only. Do not add explanations.
"""
    return await cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED)

async def extract_skill_keywords_async(text):
    if not text or not text.strip():
        return ""
    prompt = f"""
Extract a clean, comma-separated list of technical skills, tools, programming languages, frameworks, or certifications from this job description:
"{text}"

IMPORTANT: Normalize skill names to their base forms:
- Use "php" instead of "php7", "php8"
- Use "css" instead of "css3", "css4"  
- Use "html" instead of "html5"
- Use "javascript" instead of "js", "es6", "es7"
- Use "react" instead of "reactjs", "react.js"
- Use "node" instead of "nodejs", "node.js"
- Use "python" instead of "python3"
- Use "java" instead of "java8", "java11", "java17"
- Use "sql" for database skills (mysql, postgresql, oracle)
- Use "git" for version control (github, gitlab)

Only return comma-separated keywords like: Java, Spring, Hibernate, SQL, Docker, etc.
"""
    return await cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED)

async def identify_missing_skills_async(resume_skills, job_skills):
    """
    Identify skills that are required in the job description but missing from the resume.
    Returns a list of missing skills with their importance level.
    """
    if not job_skills or not job_skills.strip():
        return []
    
    if not resume_skills or not resume_skills.strip():
        # If no resume skills, all job skills are missing
        job_skill_list = [skill.strip() for skill in job_skills.split(',') if skill.strip()]
        return [{"skill": skill, "importance": "high", "category": "technical"} for skill in job_skill_list]
    
    prompt = f"""
You are a skill matching expert. Compare the candidate's skills with the job requirements and identify ALL missing skills.

IMPORTANT: Consider skill variations and versions as equivalent:
- "php" and "php7", "php8" are the same skill
- "css" and "css3", "css4" are the same skill  
- "html" and "html5" are the same skill
- "javascript" and "js", "es6", "es7" are the same skill
- "react" and "reactjs", "react.js" are the same skill
- "node" and "nodejs", "node.js" are the same skill
- "python" and "python3" are the same skill
- "java" and "java8", "java11", "java17" are the same skill
- "sql" and "mysql", "postgresql", "oracle" are database skills
- "git" and "github", "gitlab" are version control skills

Candidate's Skills: {resume_skills}
Job Requirements: {job_skills}

You MUST return a JSON array with the key "missingSkills" containing ALL missing skills.
Example format:
{{
  "missingSkills": [
    {{
      "skill": "skill name",
      "importance": "high/medium/low",
      "category": "technical/soft/certification/tool"
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. ALWAYS return an array under "missingSkills" key
2. Include ALL missing skills, not just one
3. Only include skills that are clearly required for the job but missing from the candidate's profile
4. Do NOT include skills that have equivalent variations already present in the candidate's skills
5. Focus ONLY on technical skills: programming languages, frameworks, libraries, tools, databases, platforms
6. DO NOT include concepts, practices, or methodologies like "responsive design", "well-documented code", "reusable code", "clean code", "agile", "scrum", etc.
7. If no skills are missing, return: {{"missingSkills": []}}
"""
    
    try:
        raw = await cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED, response_format={"type": "json_object"})
        print(f"🔍 Raw GPT response: {raw}")
        result = json.loads(raw) if raw else {}
        print(f"🔍 Parsed result: {result}")
        
        # Handle both array and object responses
        if isinstance(result, dict) and "missing_skills" in result:
            print(f"🔍 Found missing_skills in dict: {result['missing_skills']}")
            return result["missing_skills"]
        elif isinstance(result, dict) and "missingSkills" in result:
            print(f"🔍 Found missingSkills in dict: {result['missingSkills']}")
            return result["missingSkills"]
        elif isinstance(result, list):
            print(f"🔍 Found list result: {result}")
            return result
        elif isinstance(result, dict) and "skill" in result:
            # Handle single skill object response - retry with more explicit prompt
            print(f"🔍 Found single skill object, retrying with explicit prompt: {result}")
            
            # Retry with a more explicit prompt
            retry_prompt = f"""
The previous response was incorrect. You returned a single skill object instead of an array.

Candidate's Skills: {resume_skills}
Job Requirements: {job_skills}

You MUST return a JSON object with this EXACT structure:
{{
  "missingSkills": [
    {{
      "skill": "skill name",
      "importance": "high/medium/low", 
      "category": "technical/soft/certification/tool"
    }}
  ]
}}

IMPORTANT: Only include actual technical skills (programming languages, frameworks, tools, databases).
DO NOT include concepts like "responsive design", "well-documented code", "reusable code", etc.

Return ALL missing technical skills as an array under "missingSkills" key.
"""
            
            retry_raw = await cached_async_gpt_call(retry_prompt, model=OPENAI_MODEL_FIXED, response_format={"type": "json_object"})
            print(f"🔍 Retry response: {retry_raw}")
            retry_result = json.loads(retry_raw) if retry_raw else {}
            
            if isinstance(retry_result, dict) and "missingSkills" in retry_result:
                print(f"🔍 Retry successful: {retry_result['missingSkills']}")
                return retry_result["missingSkills"]
            else:
                print(f"🔍 Retry failed, using original single skill: {result}")
                return [result]
        else:
            print(f"🔍 No valid result found, returning empty list")
            return []
    except Exception as e:
        print(f"⚠️ Failed to identify missing skills: {e}")
        print(f"⚠️ Raw response was: {raw}")
        return []



def compute_skill_similarity(text1, text2):
    if not text1 or not text2 or embedding_model is None:
        return 0.0
    try:
        emb1 = embedding_model.encode(normalize_text_for_embedding(text1)).reshape(1, -1)
        emb2 = embedding_model.encode(normalize_text_for_embedding(text2)).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0][0])
    except Exception as e:
        print(f"Failed skill similarity: {e}")
        return 0.0

async def check_requirement_async(resume_section, jd_section, section_name, min_score=0.2):
    if section_name.lower() == "education" and not jd_section.strip():
        print("📘 JD does not specify education requirements. Assuming education match is perfect.")
        return 1.0

    guidance = ""
    if section_name.lower() == "education":
        guidance = """
    - **1.0**: Degree and Major are an exact or very close match. (e.g., JD requires 'BS in Computer Science', candidate has 'B.Tech in Computer Science & Engineering').
    - **0.8**: Degree level matches, but the major is a related technical field. (e.g., JD requires 'Computer Science', candidate has 'Electrical Engineering' or 'Information Technology').
    - **0.5**: Degree is in a different but still quantitative or scientific field. (e.g., 'B.S. in Physics' or 'Mechanical Engineering').
    - **0.2**: A degree is present but in a completely unrelated, non-technical field. (e.g., 'Bachelor of Arts in History').
    - **0.0**: No degree is listed or information is insufficient.
    """
    elif section_name.lower() == "experience":
        guidance = """
- **1.0**: Candidate's experience is a near-perfect match for the role's primary duties, domain, and technologies. (e.g., JD requires a 'Java Backend Developer with Spring and AWS', and the candidate has extensive experience in exactly that stack).
- **0.8**: Candidate's core technologies and domain align well, but might be missing a secondary tool or have slightly less experience in one area. (e.g., Strong Java/Spring experience, but used Azure instead of AWS).
- **0.5**: Candidate has relevant software development experience but in a different technology stack or domain. (e.g., A '.NET developer' applying for a 'Java' role).
- **0.2**: Candidate has some professional experience in a technical field, but it's not directly related to software development. (e.g., A 'QA Tester' applying for a 'Developer' role).
- **0.0**: No relevant professional experience is listed.
"""
    else:
        guidance = "Score from 0 to 1 based on general overlap."

    prompt = f"""
You are a meticulous evaluation agent. Compare the Candidate's {section_name} against the Job's {section_name} and provide a score.
Follow these steps:
1.  Analyze Job Requirement: Briefly state the key requirement from the "Job {section_name}" text.
2.  Analyze Candidate Profile: Briefly state the key qualification from the "Candidate {section_name}" text.
3.  Compare and Score: Compare based on the rubric below. Explain your reasoning.
4.  Provide Output: Return a single JSON object with your final "score" (a float from 0.0 to 1.0) and a "reason".

Scoring Rubric:
{guidance}

Job {section_name}:
{jd_section}
Candidate {section_name}:
{resume_section}
Return ONLY the JSON object.
"""
    try:
        raw = await cached_async_gpt_call(prompt, model=OPENAI_MODEL_FIXED, response_format={"type": "json_object"})
        result = json.loads(raw) if raw else {}
        score = float(result.get("score", min_score))
        print(f"INFO: GPT evaluation for '{section_name}': Score={score}")
        return max(score, min_score)
    except Exception as e:
        print(f"⚠️ Failed to parse GPT score for '{section_name}': {e}. Using fallback {min_score}.")
        return min_score

async def score_jd_in_parallel(resume_data, jd_text):
    jd_data = await extract_sections_async(jd_text, "Job Description")
    
    skill_res_raw = flatten_section(resume_data.get("skills", ""))
    skill_jd_raw = flatten_section(jd_data.get("skills", ""))
    
    skill_jd_keywords = await extract_skill_keywords_async(skill_jd_raw)
    expanded_res_skills = await expand_skills_async(skill_res_raw)
    expanded_jd_skills = await expand_skills_async(skill_jd_keywords)
    
    skill_score = compute_skill_similarity(expanded_res_skills, expanded_jd_skills)
    
    exp_res = flatten_section(resume_data.get("experience", ""))
    exp_jd = flatten_section(jd_data.get("experience", ""))
    edu_res = flatten_section(resume_data.get("education", ""))
    edu_jd = flatten_section(jd_data.get("education", ""))
    
    exp_res_expanded = await expand_abbreviations_async(exp_res)
    exp_jd_expanded = await expand_abbreviations_async(exp_jd)
    edu_res_expanded = await expand_abbreviations_async(edu_res)
    edu_jd_expanded = await expand_abbreviations_async(edu_jd)
    
    exp_score_gpt = await check_requirement_async(exp_res_expanded, exp_jd_expanded, "experience")
    edu_score_gpt = await check_requirement_async(edu_res_expanded, edu_jd_expanded, "education")
    
    try:
        exp_embed_sim = 0.0
        edu_embed_sim = 0.0
        if exp_res_expanded.strip() and exp_jd_expanded.strip() and embedding_model is not None:
            exp_emb_res = embedding_model.encode(normalize_text_for_embedding(exp_res_expanded)).reshape(1, -1)
            exp_emb_jd = embedding_model.encode(normalize_text_for_embedding(exp_jd_expanded)).reshape(1, -1)
            exp_embed_sim = float(cosine_similarity(exp_emb_res, exp_emb_jd)[0][0])
        if edu_res_expanded.strip() and edu_jd_expanded.strip() and embedding_model is not None:
            edu_emb_res = embedding_model.encode(normalize_text_for_embedding(edu_res_expanded)).reshape(1, -1)
            edu_emb_jd = embedding_model.encode(normalize_text_for_embedding(edu_jd_expanded)).reshape(1, -1)
            edu_embed_sim = float(cosine_similarity(edu_emb_res, edu_emb_jd)[0][0])
    except Exception as e:
        print(f"Embedding sim error in score_jd_in_parallel: {e}")
        exp_embed_sim = 0.0
        edu_embed_sim = 0.0
    
    exp_score = 0.7 * exp_score_gpt + 0.3 * exp_embed_sim
    edu_score = 0.7 * edu_score_gpt + 0.3 * edu_embed_sim
    
    scores = {"skills": skill_score, "experience": exp_score, "education": edu_score}
    
    base_weights = {"skills": 0.8, "experience": 0.1, "education": 0.1}
    adjusted_weights = base_weights.copy()
    
    if not edu_jd.strip():
        adjusted_weights["skills"] += adjusted_weights.pop("education", 0)
    if not exp_jd.strip():
        adjusted_weights["skills"] += adjusted_weights.pop("experience", 0)
    
    # Identify missing skills
    missing_skills = await identify_missing_skills_async(skill_res_raw, skill_jd_keywords)
    
    # Debug logging
    print(f"🔍 Resume skills: {skill_res_raw}")
    print(f"🔍 Job skills: {skill_jd_keywords}")
    print(f"🔍 Missing skills found: {len(missing_skills)}")
    if missing_skills:
        print(f"🔍 Missing skills: {missing_skills}")
        
    return {
        "scores": scores, 
        "weights": adjusted_weights,
        "missing_skills": missing_skills,
        "job_skills": skill_jd_keywords,
        "resume_skills": skill_res_raw
    }

# --- SCORING LOGIC (ADD THIS NEW HELPER FUNCTION) ---

async def score_resume_against_preprocessed_jd(resume_text, resume_filename, jd_data, jd_preprocessed):
    """
    Analyzes and scores a single resume against a JD that has already been processed.
    This is an efficient version for bulk analysis.
    """
    try:
        # 1. Extract sections from the current resume
        resume_data = await extract_sections_async(resume_text, "Resume")
        if not any(resume_data.values()):
            print(f"⚠️ Could not extract content from {resume_filename}, skipping.")
            return None # Skip this resume if empty

        # 2. Prepare resume data for scoring
        skill_res_raw = flatten_section(resume_data.get("skills", ""))
        expanded_res_skills = await expand_skills_async(skill_res_raw)
        
        exp_res = flatten_section(resume_data.get("experience", ""))
        edu_res = flatten_section(resume_data.get("education", ""))
        exp_res_expanded = await expand_abbreviations_async(exp_res)
        edu_res_expanded = await expand_abbreviations_async(edu_res)
        
        # 3. Compute scores using pre-processed JD data
        skill_score = compute_skill_similarity(expanded_res_skills, jd_preprocessed["expanded_jd_skills"])
        exp_score_gpt = await check_requirement_async(exp_res_expanded, jd_preprocessed["exp_jd_expanded"], "experience")
        edu_score_gpt = await check_requirement_async(edu_res_expanded, jd_preprocessed["edu_jd_expanded"], "education")
        
        # 3b. Compute embedding similarity for experience and education
        exp_embed_sim = 0.0
        edu_embed_sim = 0.0
        if exp_res_expanded.strip() and jd_preprocessed["exp_jd_expanded"].strip() and embedding_model is not None:
            exp_emb_res = embedding_model.encode(normalize_text_for_embedding(exp_res_expanded)).reshape(1, -1)
            exp_emb_jd = embedding_model.encode(normalize_text_for_embedding(jd_preprocessed["exp_jd_expanded"])).reshape(1, -1)
            exp_embed_sim = float(cosine_similarity(exp_emb_res, exp_emb_jd)[0][0])
        
        if edu_res_expanded.strip() and jd_preprocessed["edu_jd_expanded"].strip() and embedding_model is not None:
            edu_emb_res = embedding_model.encode(normalize_text_for_embedding(edu_res_expanded)).reshape(1, -1)
            edu_emb_jd = embedding_model.encode(normalize_text_for_embedding(jd_preprocessed["edu_jd_expanded"])).reshape(1, -1)
            edu_embed_sim = float(cosine_similarity(edu_emb_res, edu_emb_jd)[0][0])

        # 4. Calculate final weighted scores
        exp_score = 0.7 * exp_score_gpt + 0.3 * exp_embed_sim
        edu_score = 0.7 * edu_score_gpt + 0.3 * edu_embed_sim
        
        scores = {"skills": skill_score, "experience": exp_score, "education": edu_score}
        
        # Adjust weights based on what the JD requires
        base_weights = {"skills": 0.8, "experience": 0.1, "education": 0.1}
        adjusted_weights = base_weights.copy()
        if not jd_preprocessed["exp_jd_expanded"].strip():
            adjusted_weights["skills"] += adjusted_weights.pop("experience", 0)
        if not jd_preprocessed["edu_jd_expanded"].strip():
            adjusted_weights["skills"] += adjusted_weights.pop("education", 0)
        
        final_score = sum(adjusted_weights[key] * scores.get(key, 0) for key in adjusted_weights)
        
        # 5. Format the output for this single resume
        return {
            "score": final_score,
            "payload": {
                "file_name": resume_filename,
                "extracted_profile": resume_data
            },
            "scores_breakdown": scores,
            "weights_used": adjusted_weights
        }
    except Exception as e:
        print(f"❌ Error processing resume {resume_filename}: {e}")
        traceback.print_exc()
        return None

# --- MAIN API ENDPOINT (ANALYSIS) ---
@app.route('/api/analyze-resume', methods=['POST'])
async def analyze_resume():
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not all([qdrant_url, qdrant_api_key, client, async_client, embedding_model]):
            return jsonify({'error': 'A server-side client or configuration is missing.'}), 503

        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        file_bytes = request.files['resume'].read()
        resume_text = read_pdf(file_bytes)
        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        resume_data = await extract_sections_async(resume_text, "Resume")
        
        if not resume_data or not any(resume_data.values()):
            return jsonify({
                'error': 'Failed to extract meaningful content from the resume. The PDF might be an image, empty, or have an unusual format. Please try a different file.'
            }), 422
        
        query_text = summarize_resume_for_search(resume_data)
        query_vector = get_embedding(query_text)
        
        if query_vector is None:
            return jsonify({'error': 'Could not create query vector from resume text'}), 500
        
        initial_hits = await search_qdrant_with_httpx(
            base_url=qdrant_url, api_key=qdrant_api_key,
            collection_name=QDRANT_COLLECTION_NAME, vector=query_vector, limit=10
        )
        
        if not initial_hits:
            return jsonify({'candidate_profile': resume_data, 'job_matches': []})
            
        scoring_tasks = [
            score_jd_in_parallel(
                resume_data, 
                hit.payload.get("full_text", hit.payload.get("content_snippet", ""))
            ) for hit in initial_hits
        ]
        all_scores_and_data = await asyncio.gather(*scoring_tasks)
        
        job_matches = []
        all_missing_skills = []
        
        for i, result_data in enumerate(all_scores_and_data):
            scores = result_data["scores"]
            adjusted_weights = result_data["weights"]
            final_score = sum(adjusted_weights[key] * scores.get(key, 0) for key in adjusted_weights)
            original_payload = initial_hits[i].payload
            
            # Collect missing skills for this job
            missing_skills = result_data.get("missing_skills", [])
            all_missing_skills.extend(missing_skills)
            
            job_matches.append({
                "score": final_score,
                "payload": {
                    "title": original_payload.get("file_name", "No Title"),
                    "description": (
                        original_payload.get("job_description")
                        or original_payload.get("full_text")
                        or original_payload.get("content_snippet", "No Description")
                    ).strip()
                },
                "scores_breakdown": scores,
                "weights_used": adjusted_weights,
                "missing_skills": missing_skills,
                "job_skills": result_data.get("job_skills", ""),
                "resume_skills": result_data.get("resume_skills", "")
            })
        
        job_matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = job_matches[:5]
        
        # Aggregate and deduplicate missing skills across all jobs
        unique_missing_skills = {}
        for skill in all_missing_skills:
            skill_name = skill.get("skill", "").lower().strip()
            if skill_name and skill_name not in unique_missing_skills:
                unique_missing_skills[skill_name] = skill
        
        aggregated_missing_skills = list(unique_missing_skills.values())
        
        # Debug logging for final response
        print(f"🔍 Final response - Job matches: {len(top_matches)}")
        for i, job in enumerate(top_matches):
            print(f"🔍 Job {i+1}: {job.get('payload', {}).get('title', 'No title')}")
            print(f"🔍 Job {i+1} missing skills: {len(job.get('missing_skills', []))}")
            if job.get('missing_skills'):
                print(f"🔍 Job {i+1} missing skills details: {job['missing_skills']}")
        
        return jsonify({
            'candidate_profile': resume_data,
            'job_matches': top_matches,
            'missing_skills_summary': aggregated_missing_skills
        })
        
    except Exception as e:
        print("An error occurred during the /api/analyze-resume request:")
        traceback.print_exc()
        return jsonify({'error': f'A critical server error occurred: {str(e)}'}), 500


    
@app.route('/api/jds', methods=['GET'])
async def get_all_jds():
    try:
        scroll_payload = {"limit": 1000, "with_payload": True, "with_vectors": False}
        scroll_result = await qdrant_op_with_httpx(
            'POST', f'/collections/{QDRANT_COLLECTION_NAME}/points/scroll', scroll_payload
        )
        points = scroll_result.get('result', {}).get('points', [])
        return jsonify(points), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to fetch JDs: {str(e)}'}), 500

@app.route('/api/upload-jd', methods=['POST'])
async def upload_jd():
    if not all([embedding_model, os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")]):
        return jsonify({'error': 'Server clients or configuration not initialized'}), 503
    if 'jd' not in request.files:
        return jsonify({'error': 'No JD file uploaded'}), 400

    try:
        jd_file = request.files['jd']
        file_name = jd_file.filename
        file_bytes = jd_file.read()
        jd_text = read_pdf(file_bytes)
        if not jd_text:
            return jsonify({'error': 'Could not extract text from the JD PDF'}), 400

        vector = get_embedding(jd_text)
        if vector is None:
            return jsonify({'error': 'Failed to create embedding for the JD'}), 500

        # --- DUPLICATE CHECK ---
        # 1. Search for potential duplicates before uploading.
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        search_results = await search_qdrant_with_httpx(
            base_url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=QDRANT_COLLECTION_NAME,
            vector=vector,
            limit=1  # We only need the top match to check for a duplicate.
        )

        # 2. If a very similar document exists, block the upload.
        DUPLICATE_THRESHOLD = 0.995  # Use a high threshold to only catch near-identical duplicates.
        if search_results and search_results[0].score > DUPLICATE_THRESHOLD:
            existing_jd = search_results[0]
            print(f"⚠️ Duplicate JD detected. New upload '{file_name}' is highly similar to existing JD '{existing_jd.payload.get('file_name', 'N/A')}' (ID: {existing_jd.id}) with score {existing_jd.score}.")
            return jsonify({
                'message': 'This job description already exists in the database.',
                'status': 'duplicate_found',
                'existing_jd': {
                    'id': existing_jd.id,
                    'file_name': existing_jd.payload.get('file_name'),
                    'similarity_score': existing_jd.score
                }
            }), 409  # HTTP 409 Conflict is the appropriate status code for this situation.
        # --- END DUPLICATE CHECK ---

        # 3. If no duplicate is found, proceed with the upload.
        point_id = str(uuid.uuid4())
        payload = {"file_name": file_name, "full_text": jd_text}

        upsert_payload = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
        await qdrant_op_with_httpx(
            'PUT', f'/collections/{QDRANT_COLLECTION_NAME}/points?wait=true', upsert_payload
        )

        print(f"✅ Successfully uploaded new JD: {file_name} with ID: {point_id}")
        return jsonify({'id': point_id, 'payload': payload, 'status': 'created'}), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'A critical error occurred during JD upload: {str(e)}'}), 500

@app.route('/api/delete-jd/<jd_id>', methods=['DELETE'])
async def delete_jd(jd_id):
    try:
        delete_payload = {"points": [jd_id]}
        await qdrant_op_with_httpx(
            'POST', f'/collections/{QDRANT_COLLECTION_NAME}/points/delete?wait=true', delete_payload
        )
        print(f"✅ Successfully deleted JD with ID: {jd_id}")
        return jsonify({'message': f'JD with id {jd_id} deleted successfully'}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to delete JD: {str(e)}'}), 500
    
# --- NEW ENDPOINT FOR SENDING EMAIL RESULTS ---
@app.route('/api/analyze-missing-skills', methods=['POST'])
async def analyze_missing_skills():
    """
    Analyze missing skills for a specific resume against a specific job description.
    This endpoint allows for detailed skill gap analysis.
    """
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        if 'job_description' not in request.form:
            return jsonify({'error': 'No job description provided'}), 400
        
        # Read resume
        file_bytes = request.files['resume'].read()
        resume_text = read_pdf(file_bytes)
        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Extract resume sections
        resume_data = await extract_sections_async(resume_text, "Resume")
        if not resume_data or not any(resume_data.values()):
            return jsonify({'error': 'Failed to extract meaningful content from the resume'}), 422
        
        # Get job description
        jd_text = request.form['job_description']
        jd_data = await extract_sections_async(jd_text, "Job Description")
        
        # Extract skills
        resume_skills = flatten_section(resume_data.get("skills", ""))
        jd_skills_raw = flatten_section(jd_data.get("skills", ""))
        jd_skills_keywords = await extract_skill_keywords_async(jd_skills_raw)
        
        # Identify missing skills
        missing_skills = await identify_missing_skills_async(resume_skills, jd_skills_keywords)
        
        # Calculate skill match percentage
        total_required_skills = len([skill.strip() for skill in jd_skills_keywords.split(',') if skill.strip()])
        missing_count = len(missing_skills)
        match_percentage = max(0, ((total_required_skills - missing_count) / total_required_skills * 100)) if total_required_skills > 0 else 0
        
        return jsonify({
            'resume_skills': resume_skills,
            'job_skills': jd_skills_keywords,
            'missing_skills': missing_skills,
            'skill_match_percentage': round(match_percentage, 2),
            'total_required_skills': total_required_skills,
            'missing_skills_count': missing_count,
            'matched_skills_count': total_required_skills - missing_count
        })
        
    except Exception as e:
        print("An error occurred during the /api/analyze-missing-skills request:")
        traceback.print_exc()
        return jsonify({'error': f'A critical server error occurred: {str(e)}'}), 500

@app.route('/api/send-results', methods=['POST'])
def send_results_email():
    """
    This endpoint receives the analysis results from the frontend
    and triggers the email sending function.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        to_email = data.get('email')
        job_matches = data.get('job_matches')

        if not to_email or not job_matches:
            return jsonify({'error': 'Missing email or job_matches in request'}), 400
        
        # Call the existing send_email helper function
        success = send_email(to_email=to_email, matched_jobs=job_matches)

        if success:
            print(f"✅ Successfully queued email to {to_email}")
            return jsonify({'email_sent': True}), 200
        else:
            print(f"❌ Server-side function failed to send email to {to_email}")
            return jsonify({'error': 'The server failed to send the email.', 'email_sent': False}), 500

    except Exception as e:
        print(f"A critical error occurred in /api/send-results: {e}")
        traceback.print_exc()
        return jsonify({'error': f'A critical server error occurred: {str(e)}'}), 500

# --- NEW ENDPOINT FOR BULK RESUME ANALYSIS ---
@app.route('/api/bulk-analyze-resumes', methods=['POST'])
async def bulk_analyze_resumes():
    try:
        # 1. Validate file uploads
        if 'jd' not in request.files or not request.files.getlist('resumes'):
            return jsonify({'error': 'A single "jd" file and at least one "resumes" file are required.'}), 400

        jd_file = request.files['jd']
        resume_files = request.files.getlist('resumes')

        # 2. Process the Job Description (once)
        jd_text = read_pdf(jd_file.read())
        if not jd_text:
            return jsonify({'error': 'Could not extract text from the job description PDF.'}), 400
        
        jd_data = await extract_sections_async(jd_text, "Job Description")
        
        # Pre-process JD sections for efficient scoring
        skill_jd_raw = flatten_section(jd_data.get("skills", ""))
        exp_jd = flatten_section(jd_data.get("experience", ""))
        edu_jd = flatten_section(jd_data.get("education", ""))

        jd_preprocessed = {
            "expanded_jd_skills": await expand_skills_async(await extract_skill_keywords_async(skill_jd_raw)),
            "exp_jd_expanded": await expand_abbreviations_async(exp_jd),
            "edu_jd_expanded": await expand_abbreviations_async(edu_jd)
        }
        
        # 3. Create parallel analysis tasks for each resume
        analysis_tasks = []
        for resume_file in resume_files:
            resume_text = read_pdf(resume_file.read())
            if resume_text:
                task = score_resume_against_preprocessed_jd(
                    resume_text=resume_text,
                    resume_filename=resume_file.filename,
                    jd_data=jd_data,
                    jd_preprocessed=jd_preprocessed
                )
                analysis_tasks.append(task)
        
        # 4. Run all tasks concurrently
        results = await asyncio.gather(*analysis_tasks)
        
        # Filter out any resumes that failed processing
        successful_results = [res for res in results if res is not None]
        
        # 5. Sort results by score (highest first)
        successful_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 6. Format final JSON response
        return jsonify({
            "job_description_payload": {
                "file_name": jd_file.filename,
                "extracted_data": jd_data
            },
            "candidate_matches": successful_results
        })

    except Exception as e:
        print("An error occurred during the /api/bulk-analyze-resumes request:")
        traceback.print_exc()
        return jsonify({'error': f'A critical server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
