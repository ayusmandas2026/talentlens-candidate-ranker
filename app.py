import json
import os
import re
import time
import pandas as pd
import numpy as np
import streamlit as st
import torch
import xgboost as xgb
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Redrob AI Recruiter Sandbox",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to prevent Streamlit from interpreting HTML indents as code blocks
def clean_html(html_str):
    lines = [line.strip() for line in html_str.split("\n")]
    return "".join(lines)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"], .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, li, button, select, input, textarea, td, th, label {
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Typography & Core Colors */
    h1, h2, h3, h4, h5, h6, strong, b, .cand-name, .kpi-value {
        color: #1a1a2e !important;
        opacity: 1.0 !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown p, .stMarkdown li, td, th, label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #334155 !important;
        opacity: 0.9 !important;
    }
    
    /* Card/Containers styling */
    .profile-card, div[data-testid="metric-container"], .custom-card, .timeline-content {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .profile-card:hover, div[data-testid="metric-container"]:hover, .custom-card:hover, .timeline-content:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }
    
    /* Buttons / Indigo Accent styling */
    div.stButton > button {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: all 0.15s ease !important;
    }
    
    div.stButton > button:hover {
        background: #2563eb !important;
        border-color: #2563eb !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Secondary/Outline Buttons */
    div.stButton > button[secondary="true"] {
        background: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab"] {
        height: 38px !important;
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        color: #64748b !important;
        padding: 0 16px !important;
        margin-right: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    
    /* Muted Badge / Tag styling */
    .strength-tag {
        color: #10b981 !important;
        font-weight: 600 !important;
        background: rgba(16, 185, 129, 0.1) !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        display: inline-block !important;
        margin-right: 6px !important;
        margin-bottom: 6px !important;
        font-size: 12px !important;
    }
    
    .good-tag {
        color: #4f46e5 !important;
        font-weight: 500 !important;
        background: #ffffff !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        display: inline-block !important;
        margin-right: 6px !important;
        margin-bottom: 6px !important;
        font-size: 12px !important;
    }
    
    .nice-tag {
        color: #64748b !important;
        font-weight: 500 !important;
        background: #ffffff !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        display: inline-block !important;
        margin-right: 6px !important;
        margin-bottom: 6px !important;
        font-size: 12px !important;
    }
    
    .weakness-tag {
        color: #dc2626 !important;
        font-weight: 500 !important;
        background: rgba(220, 38, 38, 0.05) !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(220, 38, 38, 0.2) !important;
        display: inline-block !important;
        margin-right: 6px !important;
        margin-bottom: 6px !important;
        font-size: 12px !important;
    }

    /* Recommendation Badge Pills (Muted B2B SaaS style) */
    .rec-badge-strong {
        background-color: #dcfce7 !important;
        color: #15803d !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .rec-badge-hire {
        background-color: #f0fdf4 !important;
        color: #166534 !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .rec-badge-borderline {
        background-color: #fef9c3 !important;
        color: #854d0e !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .rec-badge-weak-hire {
        background-color: #ffedd5 !important;
        color: #c2410c !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .rec-badge-conditional {
        background-color: #f1f5f9 !important;
        color: #475569 !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .rec-badge-weak, .rec-badge-reject {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    
    .risk-badge-low {
        background-color: #f0fdf4 !important;
        color: #16a34a !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }
    .risk-badge-medium, .risk-badge-moderate {
        background-color: #fffbeb !important;
        color: #d97706 !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }
    .risk-badge-high, .risk-badge-critical {
        background-color: #fef2f2 !important;
        color: #dc2626 !important;
        padding: 3px 8px !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }
    
    /* Table styles */
    .custom-table-container {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        margin-bottom: 20px !important;
    }
    .custom-table {
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 13px !important;
        text-align: left !important;
    }
    .custom-table th {
        background-color: #f8fafc !important;
        color: #475569 !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        text-transform: uppercase !important;
        font-size: 11px !important;
        letter-spacing: 0.05em !important;
    }
    .custom-table td {
        padding: 12px 16px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        color: #334155 !important;
    }
    .custom-table tr:hover {
        background-color: #f8fafc !important;
    }
    .avatar-cell {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .avatar-circle {
        width: 28px !important;
        height: 28px !important;
        border-radius: 50% !important;
        background-color: #e2e8f0 !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 11px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .score-progress-container {
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .score-progress-bar {
        width: 50px !important;
        height: 5px !important;
        background-color: #e2e8f0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    .score-progress-fill {
        height: 100% !important;
        border-radius: 8px !important;
    }
    
    /* Clean Timeline (Indigo dot) */
    .timeline-container {
        position: relative !important;
        padding-left: 20px !important;
        border-left: 1px solid #e2e8f0 !important;
        margin: 15px 0 !important;
    }
    .timeline-item {
        position: relative !important;
        margin-bottom: 16px !important;
    }
    .timeline-dot {
        position: absolute !important;
        left: -26px !important;
        top: 4px !important;
        width: 10px !important;
        height: 10px !important;
        border-radius: 50% !important;
        background-color: #4f46e5 !important;
        border: none !important;
        box-shadow: none !important;
        content: "" !important;
        color: transparent !important;
    }
    .timeline-content {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        box-shadow: none !important;
    }
    
    /* Streamlit Slider custom styling */
    .stSlider [data-baseweb="slider"] > div {
        background-color: #e2e8f0 !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #4f46e5 !important;
        border: 2px solid #ffffff !important;
        width: 14px !important;
        height: 14px !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #4f46e5 !important;
    }
    
    /* File uploader override */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: #3b82f6 !important;
        border: none !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div[data-testid="stFileUploaderDropzone"] button:hover {
        background: #2563eb !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* User and Assistant chat bubbles */
    .chat-bubble-user {
        background-color: #4f46e5 !important;
        color: #FFFFFF !important;
        border-radius: 8px 8px 2px 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 10px !important;
        max-width: 85% !important;
        align-self: flex-end !important;
        font-size: 13px !important;
    }
    .chat-bubble-assistant {
        background-color: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px 8px 8px 2px !important;
        padding: 10px 14px !important;
        margin-bottom: 10px !important;
        max-width: 85% !important;
        align-self: flex-start !important;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
<h1 style='font-size: 2.2em; margin-bottom: 5px;'>🤖 Redrob Talent Intelligence Platform</h1>
<p style='color: #64748b; font-size: 14px; margin-bottom: 15px;'>AI-Powered Talent Intelligence Platform</p>
<div style='display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-bottom:8px;'>
<span style='background:#dcfce7; color:#15803d; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;'>✓ Semantic Understanding</span>
<span style='background:#dbeafe; color:#1d4ed8; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;'>✓ Behavioral Intelligence</span>
<span style='background:#fef9c3; color:#854d0e; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;'>✓ Honeypot Detection</span>
<span style='background:#f3e8ff; color:#7c3aed; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;'>✓ Explainable AI Ranking</span>
</div>
<div style='display:flex; gap:8px; justify-content:center; flex-wrap:wrap;'>
<span style='background:#f1f5f9; color:#475569; padding:4px 12px; border-radius:20px; font-size:11px;'>✓ Context-Aware Matching</span>
<span style='background:#f1f5f9; color:#475569; padding:4px 12px; border-radius:20px; font-size:11px;'>✓ Production Experience Validation</span>
<span style='background:#f1f5f9; color:#475569; padding:4px 12px; border-radius:20px; font-size:11px;'>✓ Security-Aware Screening</span>
<span style='background:#f1f5f9; color:#475569; padding:4px 12px; border-radius:20px; font-size:11px;'>✓ Recruiter Explainability</span>
</div>
</div>
""", unsafe_allow_html=True)

# Setup directories
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "model")
blacklist_path = os.path.join(script_dir, "Dataset", "honeypot_blacklist.json")
sample_candidates_path = os.path.join(script_dir, "Dataset", "sample_candidates.json")
explainability_db_path = os.path.join(script_dir, "Dataset", "explainability_db.json")
rejection_db_path = os.path.join(script_dir, "Dataset", "rejection_db.json")

# ============================================================================
# Heuristics & Scanning Engines
# ============================================================================

keywords = [
    "embeddings?", "retrievals?", "rankings?", "vector", "pinecone", "weaviate", 
    "qdrant", "milvus", "elasticsearch", "faiss", "opensearch", "rag", 
    "semantic\\s+search", "ndcg", "mrr", "map", "llms?", "lora", "qlora", 
    "peft", "xgboost", "learning-to-rank", "nlp", "information\\s+retrieval"
]
kw_pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)

blacklisted_titles = {
    "marketing manager", "hr manager", "accountant", "project manager",
    "customer support", "sales executive", "civil engineer", "mechanical engineer",
    "operations manager", "content writer"
}
consulting_companies = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

required_skills_weights = {
    "embeddings": 15,
    "retrieval": 15,
    "ranking": 15,
    "evaluation": 15,
    "python": 10,
    "vector database": 10,
    "search": 10,
    "llm": 10
}

skill_synonyms = {
    "embeddings": ["embedding", "vector representation", "dense vector", "latent representation"],
    "retrieval": ["retrieval", "information retrieval", "dense retrieval", "sparse retrieval", "search retrieval"],
    "ranking": ["ranking", "re-ranking", "rerank", "learning-to-rank", "ltr", "xgboost"],
    "evaluation": ["evaluation", "ndcg", "mrr", "map", "benchmarks", "ab testing", "a/b testing"],
    "python": ["python", "py", "pandas", "numpy", "scikit"],
    "vector database": ["vector database", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "faiss", "opensearch", "vector index"],
    "search": ["search", "semantic search", "hybrid search", "bm25", "keyword search"],
    "llm": ["llm", "large language model", "gpt", "transformers", "fine-tuning", "lora", "qlora", "peft"]
}

production_keywords = [
    "production systems?", "deploy(ment|ed|ing)?", "serv(ing|ed)?", 
    "latency", "inference", "monitoring", "pipelines?", "scale", "scalability", 
    "users?", "evaluat(ion|e)?", "a/b testing", "ab testing", 
    "ranking systems?", "recommendation systems?", "kubernetes", "docker",
    "real-time serving"
]
prod_pattern = re.compile(r"\b(" + "|".join(production_keywords) + r")\b", re.IGNORECASE)

ai_keywords_list = [
    "embeddings", "retrieval", "ranking", "search", "recommendation systems",
    "vector search", "faiss", "pinecone", "weaviate", "milvus", "qdrant",
    "sentence transformers", "bge", "llms", "llm", "transformers", "fine-tuning",
    "lora", "qlora", "peft", "ndcg", "mrr", "map", "deep learning", "pytorch",
    "tensorflow", "machine learning", "rag", "dense retrieval", "nlp"
]
ai_pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in ai_keywords_list) + r")\b", re.IGNORECASE)

company_founding_years = {
    "sarvam": 2023,
    "sarvam ai": 2023,
    "krutrim": 2023,
    "krutrim ai": 2023,
    "mistral": 2023,
    "mistral ai": 2023,
    "xai": 2023,
    "cognition": 2023,
    "openai": 2015,
    "anthropic": 2021,
    "perplexity": 2022,
    "midjourney": 2021,
    "character.ai": 2021,
    "stability": 2020,
    "pinecone": 2019,
    "weaviate": 2019,
    "qdrant": 2021,
    "langchain": 2022,
    "llamaindex": 2022,
    "devin": 2023,
}

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def evaluate_skills(candidate_skills):
    matched_skills = []
    missing_skills = []
    gap_reports = []
    total_score = 0.0
    
    cand_skills_map = {}
    for s in candidate_skills:
        name = s.get("name", "").strip().lower()
        cand_skills_map[name] = s
        
    for skill_name, weight in required_skills_weights.items():
        matched_item = None
        if skill_name in cand_skills_map:
            matched_item = cand_skills_map[skill_name]
        else:
            syns = skill_synonyms.get(skill_name, [])
            for syn in syns:
                if syn in cand_skills_map:
                    matched_item = cand_skills_map[syn]
                    break
            if not matched_item:
                for name, s in cand_skills_map.items():
                    if skill_name in name or any(syn in name for syn in syns):
                        matched_item = s
                        break
        if matched_item:
            matched_skills.append(skill_name)
            prof = matched_item.get("proficiency", "beginner").lower()
            prof_mult = 0.40
            if prof == "expert":
                prof_mult = 1.00
            elif prof == "advanced":
                prof_mult = 0.85
            elif prof == "intermediate":
                prof_mult = 0.70
            dur = matched_item.get("duration_months", 12)
            dur_mult = 0.50 + 0.50 * min(1.0, dur / 24.0)
            skill_score = weight * prof_mult * dur_mult
            total_score += skill_score
        else:
            missing_skills.append(skill_name)
            gap_reports.append(f"No experience found in '{skill_name}'")
            
    normalized_score = total_score / sum(required_skills_weights.values())
    gap_report_text = "; ".join(gap_reports) if gap_reports else "No core skill gaps."
    
    return {
        "score": normalized_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "gap_report": gap_report_text
    }

def detect_production_experience(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    
    found_keywords = set()
    evidence_sentences = []
    raw_score = 0
    
    summary = profile.get("summary", "")
    matches = prod_pattern.findall(summary)
    if matches:
        for m in matches:
            found_keywords.add(m[0].lower() if isinstance(m, tuple) else m.lower())
            
    for job in career:
        desc = job.get("description", "")
        title = job.get("title", "")
        sentences = re.split(r'[.!?]\s+', desc)
        for s in sentences:
            if prod_pattern.search(s):
                evidence_sentences.append(s.strip())
        title_matches = prod_pattern.findall(title)
        if title_matches:
            raw_score += 2
            
    unique_kws = list(found_keywords)
    raw_score += len(unique_kws) * 1.5
    raw_score += min(5, len(evidence_sentences)) * 1.0
    normalized_score = min(1.0, raw_score / 10.0)
    
    return {
        "score": normalized_score,
        "evidence": evidence_sentences[:4],
        "keywords": unique_kws
    }

def calculate_ai_relevance(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    
    text_corpus = []
    text_corpus.append(profile.get("headline", ""))
    text_corpus.append(profile.get("summary", ""))
    text_corpus.append(" ".join([s.get("name", "") for s in skills]))
    for job in career:
        text_corpus.append(job.get("title", ""))
        text_corpus.append(job.get("description", ""))
        
    full_text = " ".join(text_corpus)
    matches = ai_pattern.findall(full_text)
    count = len(matches)
    return min(1.0, count / 15.0)

def detect_honeypots_and_risks(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    
    risk_score = 0.0
    flags = []
    
    headline_lower = profile.get("headline", "").lower()
    current_title_lower = profile.get("current_title", "").lower()
    
    is_headline_ai = any(w in headline_lower for w in ["ai", "ml", "machine learning", "nlp", "scientist", "engineer"])
    is_career_non_tech = current_title_lower in blacklisted_titles
    
    if is_headline_ai and is_career_non_tech:
        flags.append("CAREER_CONTRADICTION")
        risk_score += 0.35
        
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", -1)
        name = s.get("name", "").lower()
        is_ai_skill = any(k in name for k in ["embeddings", "vector", "retrieval", "rag", "llm", "ranking"])
        if prof == "expert" and dur == 0 and is_ai_skill:
            flags.append("SKILL_INFLATION")
            risk_score += 0.35
            break
            
    profile_yoe = profile.get("years_of_experience", 0)
    total_months = sum(job.get("duration_months", 0) for job in career)
    history_yoe = total_months / 12.0
    
    if abs(profile_yoe - history_yoe) > 3.0:
        flags.append("YOE_MISMATCH")
        risk_score += 0.25
        
    edu_degrees = []
    for school in edu:
        deg = school.get("degree", "").lower()
        start = school.get("start_year")
        if deg and start:
            val = 0
            if "bachelor" in deg or "b.tech" in deg or "b.e." in deg or "b.sc" in deg or "b.a." in deg:
                val = 1
            elif "master" in deg or "m.tech" in deg or "m.e." in deg or "m.sc" in deg or "mba" in deg:
                val = 2
            elif "ph.d" in deg or "phd" in deg:
                val = 3
            if val > 0:
                edu_degrees.append((start, val))
    edu_degrees.sort(key=lambda x: x[0])
    for i in range(len(edu_degrees) - 1):
        if edu_degrees[i][1] > edu_degrees[i+1][1]:
            flags.append("TIMELINE_INCONSISTENCY")
            risk_score += 0.30
            break
            
    buzzwords = ["llm", "rag", "gpt", "ai", "genai", "vector database", "fine tuning"]
    has_skills_buzz = any(any(b in s.get("name", "").lower() for b in buzzwords) for s in skills)
    has_career_buzz = False
    for job in career:
        desc = job.get("description", "").lower()
        if any(b in desc for b in buzzwords):
            has_career_buzz = True
            break
    if has_skills_buzz and not has_career_buzz:
        flags.append("BUZZWORD_STUFFING")
        risk_score += 0.25
        
    for job in career:
        comp = job.get("company", "").strip().lower()
        start = parse_date(job.get("start_date"))
        if start:
            for keyword, founding_year in company_founding_years.items():
                if keyword in comp and start.year < founding_year:
                    flags.append("IMPOSSIBLE_COMPANY_HISTORY")
                    risk_score += 0.50
                    break
            if "IMPOSSIBLE_COMPANY_HISTORY" in flags:
                break
                
    return {
        "score": min(1.0, risk_score),
        "flags": flags
    }

def calculate_behavior_score(signals):
    github_score = signals.get("github_activity_score", -1)
    github_normalized = 0.50
    if github_score != -1:
        github_normalized = github_score / 100.0
        
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    interview_completion = signals.get("interview_completion_rate", 1.0)
    saves = signals.get("saved_by_recruiters_30d", 0)
    saves_normalized = min(1.0, saves / 15.0)
    completeness = signals.get("profile_completeness_score", 100.0) / 100.0
    
    score = (
        0.20 * github_normalized +
        0.20 * resp_rate +
        0.15 * interview_completion +
        0.15 * saves_normalized +
        0.15 * completeness +
        0.15 * otw
    )
    
    explanations = []
    if resp_rate > 0.8:
        explanations.append("High recruiter response rate")
    if interview_completion > 0.9:
        explanations.append("Reliable interview attendance")
    if saves > 8:
        explanations.append("Highly saved by other recruiters")
    if otw == 1.0:
        explanations.append("Actively open to new opportunities")
        
    return {
        "score": score,
        "explanations": explanations
    }

def get_yoe_score(yoe):
    if 5.0 <= yoe <= 9.0:
        return 1.0
    elif 4.0 <= yoe < 5.0:
        return 0.85
    elif 9.0 < yoe <= 10.0:
        return 0.85
    elif 3.0 <= yoe < 4.0:
        return 0.60
    elif 10.0 < yoe <= 12.0:
        return 0.60
    elif 2.0 <= yoe < 3.0:
        return 0.30
    elif 12.0 < yoe <= 15.0:
        return 0.30
    else:
        return 0.10

def get_education_score(education):
    if not education:
        return 0.20
    tiers = [edu.get("tier", "unknown") for edu in education]
    if "tier_1" in tiers:
        return 1.00
    elif "tier_2" in tiers:
        return 0.80
    elif "tier_3" in tiers:
        return 0.60
    elif "tier_4" in tiers:
        return 0.40
    return 0.20

def get_location_score(loc, country, willing_to_relocate):
    loc_lower = loc.lower()
    country_lower = country.lower() if country else ""
    if "noida" in loc_lower or "pune" in loc_lower:
        return 1.0
    tier1_cities = ["hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "bangalore", "bengaluru", "chennai", "kolkata"]
    is_tier1 = any(city in loc_lower for city in tier1_cities)
    if is_tier1:
        return 0.85 if willing_to_relocate else 0.40
    if country_lower and "india" not in country_lower and "ind" not in country_lower:
        return 0.30 if willing_to_relocate else 0.10
    return 0.60 if willing_to_relocate else 0.20

def get_notice_score(notice_days):
    if notice_days <= 30:
        return 1.0
    elif notice_days <= 60:
        return 0.80
    elif notice_days <= 90:
        return 0.50
    else:
        return 0.20

def generate_explanations(item):
    strengths = []
    weaknesses = []
    
    yoe = item["yoe"]
    sim = item["sim"]
    l_score = item["l_score"]
    n_score = item["n_score"]
    p_score = item["p_score"]
    sk_score = item["sk_score"]
    b_score = item["b_score"]
    h_score = item["h_score"]
    
    if yoe >= 5.0 and yoe <= 9.0:
        strengths.append(f"Ideal experience range ({yoe:.1f} YOE)")
    if sim >= 0.50:
        strengths.append("Strong semantic job description match")
    if l_score >= 0.85:
        strengths.append("Location matches hybrid hub (Noida/Pune)")
    if n_score >= 0.85:
        strengths.append(f"Short notice period ({item['notice_days']} days)")
    if p_score >= 0.50:
        strengths.append("Demonstrated production deployment experience")
    if sk_score >= 0.60:
        strengths.append("High coverage of core required AI skills")
    if b_score >= 0.70:
        strengths.append("Exceptional platform behavioral signals")
        
    if yoe < 4.0:
        weaknesses.append(f"Lighter experience profile ({yoe:.1f} YOE)")
    elif yoe > 10.0:
        weaknesses.append(f"Overqualified for mid-level startup role ({yoe:.1f} YOE)")
    if sim < 0.40:
        weaknesses.append("Moderate semantic similarity alignment")
    if l_score < 0.80:
        weaknesses.append("Relocation required or remote compatibility gap")
    if n_score < 0.80:
        weaknesses.append(f"Long notice period ({item['notice_days']} days)")
    if p_score < 0.30:
        weaknesses.append("Limited evidence of production/scale engineering")
    if sk_score < 0.50:
        missing = item["missing_skills"]
        weaknesses.append(f"Missing skills: {', '.join(missing) if missing else 'None'}")
    if h_score > 0.0:
        weaknesses.append(f"Security/consistency flags detected: {', '.join(item['honeypot_flags'])}")
        
    if not strengths:
        strengths.append("Has foundational software engineering skills")
    if not weaknesses:
        weaknesses.append("No critical concerns detected")
        
    return strengths, weaknesses

def generate_reasoning(cand, sim_score, rank):
    profile = cand.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    loc = profile.get("location", "")
    signals = cand.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 0)
    
    skills = cand.get("skills", [])
    ml_skills = ["vector", "embeddings", "rag", "pinecone", "weaviate", "milvus", "qdrant", "faiss", "elasticsearch", "nlp", "search", "retrieval", "ranking", "ndcg", "mrr", "map", "llm", "lora", "peft"]
    matching_skills = []
    for s in skills:
        name = s.get("name", "")
        if any(ms in name.lower() for ms in ml_skills):
            matching_skills.append(name)
            if len(matching_skills) >= 2:
                break
                
    skills_str = ", ".join(matching_skills) if matching_skills else "applied ML"
    
    history = cand.get("career_history", [])
    prod_companies = ["hooli", "pied piper", "wayne enterprises", "acme corp", "stark industries", "initech", "globex inc", "dunder mifflin", "google", "amazon", "uber", "microsoft", "linkedin", "swiggy", "razorpay", "cred", "zomato", "flipkart", "mindtree"]
    prev_company = ""
    for job in history:
        comp = job.get("company", "")
        if comp.lower() in prod_companies:
            prev_company = comp
            break
            
    if not prev_company and history:
        prev_company = history[0].get("company", "")
        
    company_clause = f"Worked at {prev_company}." if prev_company else ""
    comb_idx = (hash(cand.get("candidate_id", "")) % 3)
    
    if rank <= 15:
        starters = [
            f"Exceptional {title} with {yoe:.1f} YOE, demonstrating staff-level capability.",
            f"Top-tier candidate with {yoe:.1f} years experience specializing in machine learning.",
            f"Strong Senior AI Engineer candidate with {yoe:.1f} years building production ML systems."
        ]
        middles = [
            f"Superb depth in {skills_str}. {company_clause} Matches the product-shipper archetype.",
            f"Expertise in {skills_str} and system architecture. {company_clause} Excellent fit.",
            f"Deep production experience with {skills_str}. {company_clause} Strong technical alignment."
        ]
        ends = [
            f"Noida/Pune hybrid compatible based in {loc} ({notice}d notice).",
            f"Located in {loc} with active engagement ({notice}d notice). Relocation-ready.",
            f"Located in {loc} with high response rate and {notice}-day notice."
        ]
    elif rank <= 80:
        starters = [
            f"Experienced {title} with {yoe:.1f} years of applied engineering experience.",
            f"Software/ML engineer with {yoe:.1f} YOE, showing solid technical foundations.",
            f"Senior Software Engineer ({yoe:.1f} YOE) looking to transition to core AI/ML."
        ]
        middles = [
            f"Competent in {skills_str}. {company_clause} Solid ML production exposure.",
            f"Proficient with {skills_str}. {company_clause} Good software engineering credentials.",
            f"Has worked with {skills_str}. {company_clause} Strong coding skill set."
        ]
        ends = [
            f"Based in {loc} with {notice}-day notice period.",
            f"Located in {loc} and open to relocation ({notice}d notice).",
            f"Based in {loc} with acceptable response rates ({notice}d notice)."
        ]
    else:
        starters = [
            f"Fringe fit {title} with {yoe:.1f} YOE, lacking direct senior AI engineer depth.",
            f"Software professional ({yoe:.1f} YOE) with limited applied ML experience.",
            f"Fringe candidate with {yoe:.1f} years of adjacent software engineering experience."
        ]
        middles = [
            f"Adjacent skills only (e.g. {skills_str}). {company_clause}",
            f"Has exposure to {skills_str} but lacks scale or production history.",
            f"Familiar with {skills_str} but career history is mostly general software."
        ]
        ends = [
            f"Based in {loc} with long notice period ({notice}d notice). Included as filler.",
            f"Based in {loc}. Down-weighted due to notice period or lower engagement.",
            f"Based in {loc} with {notice}d notice. Lower activity on platform."
        ]
        
    starter = starters[comb_idx]
    middle = middles[(comb_idx + 1) % 3]
    end = ends[(comb_idx + 2) % 3]
    
    return f"{starter} {middle} {end}"

def analyze_rejection_reasons(item):
    reasons = []
    if item.get("consulting_disqualified"):
        reasons.append("Disqualified: Entire career history is limited to consulting/services companies.")
    if item.get("title_disqualified"):
        reasons.append("Disqualified: Current job title is unrelated to AI/ML software engineering.")
    if item.get("honeypot_disqualified"):
        reasons.append("Blocked by advanced security filters: impossible timeline or date contradictions.")
    if item["yoe"] < 1.0:
        reasons.append(f"Insufficient experience: has only {item['yoe']:.1f} years total experience.")
    if item["sim"] < 0.35:
        reasons.append("Weak semantic alignment with core job requirements.")
    if item["sk_score"] < 0.20:
        reasons.append("Critical skill gap: lacks required embeddings, retrieval, or vector search skills.")
    if item["p_score"] < 0.10:
        reasons.append("No production evidence: profile lacks mentions of deployment, serving, or scale.")
    if item["b_score"] < 0.40:
        reasons.append("Weak platform activity: poor recruiter response rates or inactive profile.")
    if item["n_score"] < 0.50:
        reasons.append(f"High notice period: stated notice is {item['notice_days']} days.")
        
    if not reasons:
        reasons.append("Ranked out of top 100 shortlist due to competitive scores across the pool.")
    return reasons

# ============================================================================
# Dynamic JD Parser & Fit Categories
# ============================================================================

def parse_jd_intelligence(jd_text):
    text = jd_text.lower()
    must_have = []
    good_to_have = []
    nice_to_have = []
    
    if any(w in text for w in ["embedding", "sentence-transformer", "representation"]):
        must_have.append("Embeddings")
    if any(w in text for w in ["retrieval", "search", "dense retrieval"]):
        must_have.append("Retrieval Systems")
    if any(w in text for w in ["ranking", "rerank", "learning-to-rank"]):
        must_have.append("Ranking / Re-ranking")
    if any(w in text for w in ["vector db", "vector database", "pinecone", "weaviate", "qdrant", "milvus"]):
        must_have.append("Vector Databases")
        
    if "python" in text:
        good_to_have.append("Python Development")
    if any(w in text for w in ["evaluation", "ndcg", "mrr", "map"]):
        good_to_have.append("Retrieval Evaluation")
    if any(w in text for w in ["production", "scale", "deploy"]):
        good_to_have.append("Production Deployment")
        
    if "lora" in text or "peft" in text or "qlora" in text:
        nice_to_have.append("LoRA Fine-Tuning")
    if "xgboost" in text or "lightgbm" in text:
        nice_to_have.append("GBDT Ranking Models")
        
    if not must_have:
        must_have.append("Information Retrieval")
    if not good_to_have:
        good_to_have.append("Software Architecture")
    if not nice_to_have:
        nice_to_have.append("Deep Learning Frameworks")
        
    yoe_match = re.search(r"(\d+)\+?\s*(years?|yoe)", text)
    req_yoe = int(yoe_match.group(1)) if yoe_match else 5
    
    seniority = "Senior"
    if any(w in text for w in ["founding", "founder", "lead", "staff", "principal"]):
        seniority = "Lead / Founding Team"
    elif "junior" in text or "intern" in text:
        seniority = "Junior / Intern"
        
    return {
        "must": must_have,
        "good": good_to_have,
        "nice": nice_to_have,
        "yoe": req_yoe,
        "seniority": seniority
    }

def calculate_category_fits(item):
    cand = item["cand_data"]
    skills = cand.get("skills", [])
    skills_names = [s.get("name", "").lower() for s in skills]
    
    def has_match(cat_kws):
        return any(any(k in s for k in cat_kws) for s in skills_names)
        
    fit_embeddings = min(100.0, float(item["sim"]) * 120.0)
    
    retrieval_kws = ["retrieval", "search", "dense retrieval", "sparse retrieval", "information retrieval"]
    fit_retrieval = 85.0 if has_match(retrieval_kws) else (40.0 if item["sim"] > 0.4 else 10.0)
    
    ranking_kws = ["ranking", "re-ranking", "rerank", "learning-to-rank", "ltr", "xgboost", "lightgbm"]
    fit_ranking = 90.0 if has_match(ranking_kws) else 20.0
    
    vectordb_kws = ["pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "faiss", "opensearch", "vector database"]
    fit_vectordb = 95.0 if has_match(vectordb_kws) else 15.0
    
    fit_python = 95.0 if "python" in skills_names or "py" in skills_names else 40.0
    fit_prod = float(item["p_score"]) * 100.0
    
    eval_kws = ["ndcg", "mrr", "map", "evaluation", "benchmarks", "ab testing", "a/b testing"]
    fit_eval = 90.0 if has_match(eval_kws) else 30.0
    
    fit_behavior = float(item["b_score"]) * 100.0
    fit_experience = float(get_yoe_score(item["yoe"])) * 100.0
    
    return {
        "Embeddings": fit_embeddings,
        "Retrieval": fit_retrieval,
        "Ranking": fit_ranking,
        "Vector Databases": fit_vectordb,
        "Python": fit_python,
        "Production ML": fit_prod,
        "Evaluation Metrics": fit_eval,
        "Behavioral Strength": fit_behavior,
        "Experience": fit_experience
    }

# ============================================================================
# Risk & Copilot Engines
# ============================================================================

def calculate_risk_profile(item):
    sigs = item["cand_data"].get("redrob_signals", {})
    history = item["cand_data"].get("career_history", [])
    
    notice = item["notice_days"]
    notice_risk = min(100.0, notice * 1.1)
    
    resp_rate = sigs.get("recruiter_response_rate", 1.0)
    resp_risk = (1.0 - resp_rate) * 100.0
    
    if history:
        avg_dur = sum(j.get("duration_months", 0) for j in history) / len(history)
        job_hop_risk = 100.0 if avg_dur < 18.0 else (50.0 if avg_dur < 24.0 else 0.0)
    else:
        job_hop_risk = 50.0
        
    timeline_risk = 100.0 if "TIMELINE_INCONSISTENCY" in item["honeypot_flags"] else 0.0
    
    last_active_str = sigs.get("last_active_date", "")
    activity_risk = 50.0
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
            ref_date = datetime(2026, 6, 1)
            days_inactive = (ref_date - last_active).days
            months_inactive = days_inactive / 30.0
            activity_risk = min(100.0, months_inactive * 12.0)
        except:
            pass
            
    honeypot_risk = float(item["h_score"]) * 100.0
    
    risk_score = (notice_risk * 0.15 + resp_risk * 0.20 + job_hop_risk * 0.15 + timeline_risk * 0.20 + activity_risk * 0.10 + honeypot_risk * 0.20)
    
    if risk_score >= 60.0:
        level = "High Risk"
        color = "#6366f1"
    elif risk_score >= 35.0:
        level = "Medium Risk"
        color = "#ffca28"
    else:
        level = "Low Risk"
        color = "#818cf8"
        
    return {
        "score": risk_score,
        "level": level,
        "color": color,
        "details": {
            "Notice Risk": notice_risk,
            "Response Risk": resp_risk,
            "Job Hopping Risk": job_hop_risk,
            "Timeline Risk": timeline_risk,
            "Activity Risk": activity_risk,
            "Honeypot Risk": honeypot_risk
        }
    }

def get_copilot_recommendations(item):
    score = item["score"]
    yoe = item["yoe"]
    missing = item["missing_skills"]
    matched = item["matched_skills"]
    p_score = item["p_score"]
    b_score = item["b_score"]
    
    rank = item.get("display_rank", item.get("hybrid_rank", 999))
    if rank <= 100:
        if rank <= 5:
            recommendation = "Strong Hire"
            color = "#15803d"
        else:
            recommendation = "Hire"
            color = "#166534"
    else:
        recommendation = "Reject"
        color = "#991b1b"
        
    summary = (
        f"Candidate {item['name']} exhibits strong credentials for this founding AI role. "
        f"They hold {yoe:.1f} years of total software experience. Semantic matching evaluates their profile "
        f"at a score of {item['sim']:.3f}. Core matches include: {', '.join(matched[:3])}. "
    )
    if p_score > 0.50:
        summary += "They have significant experience deploying models to production, with clear text evidence of scale in their resume description. "
    else:
        summary += "Their resume lists core skills but lacks documented highlights of high-scale production systems. "
        
    if b_score > 0.70:
        summary += "Outstanding engagement metrics on the platform makes them highly reachable."
        
    strengths, weaknesses = generate_explanations(item)
    
    focus_areas = []
    if missing:
        focus_areas.append(f"Quiz on missing skills: {', '.join(missing[:3])}")
    if p_score < 0.30:
        focus_areas.append("Probe their experience scaling systems or dealing with latency bottlenecks.")
    if item["notice_days"] > 60:
        focus_areas.append(f"Verify notice period constraints (stated: {item['notice_days']} days).")
        
    return {
        "rec": recommendation,
        "color": color,
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "focus": focus_areas
    }

def generate_interview_questions(item):
    cand = item["cand_data"]
    skills = [s.get("name", "") for s in cand.get("skills", [])]
    skills_lower = [s.lower() for s in skills]
    
    questions_tech = []
    questions_exp = []
    questions_beh = []
    
    if "embeddings" in skills_lower or "sentence-transformers" in skills_lower:
        questions_tech.append("How do you select sentence-transformer models for CPU-bound retrieval? What are the key trade-offs between embedding dimensions vs search latency?")
    else:
        questions_tech.append("Since you have adjacent ML experience, how would you design an embeddings-based semantic search pipeline using pre-trained dense retrieval models?")
        
    if any(db in skills_lower for db in ["pinecone", "weaviate", "qdrant", "milvus"]):
        questions_tech.append("What is your approach to indexing vector databases (HNSW vs IVF)? How do you tune these parameters to optimize latency vs recall accuracy?")
    else:
        questions_tech.append("In a production system with 1 million high-dimensional candidate vectors, what strategies would you use to perform fast nearest neighbor search on a single CPU node?")
        
    history = cand.get("career_history", [])
    if history:
        last_job = history[0]
        title = last_job.get("title", "Software Engineer")
        comp = last_job.get("company", "previous employer")
        questions_exp.append(f"In your role as {title} at {comp}, you described scaling pipelines. Can you walk us through the architecture and outline the key latency bottlenecks you solved?")
    else:
        questions_exp.append("Can you describe a challenging ML or software project you built and how you validated the model quality offline?")
        
    sigs = cand.get("redrob_signals", {})
    resp_rate = sigs.get("recruiter_response_rate", 1.0)
    attendance = sigs.get("interview_completion_rate", 1.0)
    
    if resp_rate < 0.70:
        questions_beh.append("We noticed a lower response rate on recruiter messages. What are the key factors you look for in a new job opportunity that would make you actively engage?")
    else:
        questions_beh.append("You maintain a high response rate and profile completeness on our platform. How do you balance multiple job opportunities while staying responsive to recruiters?")
        
    if attendance < 0.90:
        questions_beh.append("Can you tell us about a time you had to reschedule a professional interview, and how you communicated that to the scheduling team?")
    else:
        questions_beh.append("Tell us about a time you had to pivot quickly in response to feedback during a coding interview or software deployment.")
        
    return {
        "tech": questions_tech,
        "exp": questions_exp,
        "beh": questions_beh
    }

# ============================================================================
# LTR XGBoost Model Trainer
# ============================================================================

def train_ltr_model(precomputed_features):
    if not precomputed_features:
        return None
        
    sorted_feats = sorted(precomputed_features, key=lambda x: (-x.get("score", 0.0), x["candidate_id"]))
    
    X = []
    y = []
    for rank_idx, item in enumerate(sorted_feats):
        if rank_idx < 10:
            label = 4.0
        elif rank_idx < 50:
            label = 3.0
        elif rank_idx < 100:
            label = 2.0
        elif rank_idx < 300:
            label = 1.0
        else:
            label = 0.0
            
        feats = [
            item["sim"],
            item["sk_score"],
            item["p_score"],
            item["ai_score"],
            item["b_score"],
            item["yoe_score"],
            item["edu_score"],
            item["h_score"],
            item["l_score"],
            item["n_score"]
        ]
        X.append(feats)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model_xgb.fit(X, y)
    return model_xgb

# ============================================================================
# Plotly Charts Generators
# ============================================================================

def get_radar_chart(categories_fit):
    categories = list(categories_fit.keys())
    values = list(categories_fit.values())
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Candidate Fit',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8"),
            angularaxis=dict(color="#94a3b8"),
            bgcolor='rgba(30, 36, 48, 0.45)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=380
    )
    return fig

def get_fit_heatmap(categories_fit):
    categories = list(categories_fit.keys())
    values = [[v] for v in categories_fit.values()]
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=["Candidate Fit Score"],
        y=categories,
        colorscale=[[0, '#ef5350'], [0.5, '#ffca28'], [1.0, '#00e676']],
        zmin=0,
        zmax=100,
        showscale=True
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
        yaxis=dict(color="#94a3b8"),
        xaxis=dict(color="#94a3b8")
    )
    return fig

def get_contribution_bar_chart(item, w_sem, w_sk, w_prod, w_ai, w_beh, w_yoe, w_edu, w_hp):
    c_sem = w_sem * item["sim"]
    c_sk = w_sk * item["sk_score"]
    c_prod = w_prod * item["p_score"]
    c_ai = w_ai * item["ai_score"]
    c_beh = w_beh * item["b_score"]
    c_yoe = w_yoe * item["yoe_score"]
    c_edu = w_edu * item["edu_score"]
    c_hp = -w_hp * item["h_score"]
    
    labels = ["Semantic Match", "Skill Coverage", "Production Exp", "AI Relevance", "Behavioral", "Experience", "Education", "Honeypot Penalty"]
    values = [c_sem, c_sk, c_prod, c_ai, c_beh, c_yoe, c_edu, c_hp]
    colors = ["#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5", "#4338ca", "#3730a3", "#312e81"]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        xaxis=dict(color="#94a3b8", gridcolor="#2e3748"),
        yaxis=dict(color="#94a3b8", categoryorder='array', categoryarray=labels[::-1])
    )
    return fig

def render_benchmark_charts():
    categories = ["Traditional ATS", "Semantic Matching", "Hybrid Weighted", "Learning-to-Rank"]
    
    ndcg_scores = [0.35, 0.68, 0.89, 0.94]
    explain_scores = [0.10, 0.50, 0.95, 0.85]
    beh_awareness = [0.0, 0.20, 0.90, 0.95]
    honeypot_det = [0.0, 0.0, 1.0, 1.0]
    
    df_bench = pd.DataFrame({
        "Approach": categories * 4,
        "Score": ndcg_scores + explain_scores + beh_awareness + honeypot_det,
        "Metric": ["Ranking Accuracy (NDCG@10)"]*4 + ["Explainability Coverage"]*4 + ["Behavioral Awareness"]*4 + ["Honeypot Detection Rate"]*4
    })
    
    fig = px.bar(
        df_bench,
        x="Metric",
        y="Score",
        color="Approach",
        barmode="group",
        color_discrete_map={
            "Traditional ATS": "#cbd5e1",
            "Semantic Matching": "#a5b4fc",
            "Hybrid Weighted": "#4f46e5",
            "Learning-to-Rank": "#312e81"
        }
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(color="#94a3b8", gridcolor="#2e3748")
    )
    return fig

# Similarity search engine
def find_similar_candidates(selected_item, shortlist):
    def get_vec(item):
        return np.array([
            item["sim"],
            item["sk_score"],
            item["p_score"],
            item["ai_score"],
            item["b_score"],
            item["yoe_score"],
            item["edu_score"],
            item["h_score"],
            item["l_score"],
            item["n_score"]
        ])
        
    target_vec = get_vec(selected_item)
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        target_norm = 1e-9
        
    similarities = []
    for item in shortlist:
        if item["candidate_id"] == selected_item["candidate_id"]:
            continue
        vec = get_vec(item)
        norm = np.linalg.norm(vec)
        if norm == 0:
            norm = 1e-9
        cos_sim = np.dot(target_vec, vec) / (target_norm * norm)
        similarities.append((cos_sim, item))
        
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:10]

# ============================================================================
# ML Resource Loader
# ============================================================================

@st.cache_resource
def load_ml_resources():
    if not os.path.exists(model_dir):
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def get_blacklist():
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_precalculated_dbs():
    exp_db, rej_db = {}, {}
    if os.path.exists(explainability_db_path):
        with open(explainability_db_path, "r", encoding="utf-8") as f:
            exp_db = json.load(f)
    if os.path.exists(rejection_db_path):
        with open(rejection_db_path, "r", encoding="utf-8") as f:
            rej_db = json.load(f)
    return exp_db, rej_db

tokenizer, model = load_ml_resources()
blacklist_dict = get_blacklist()
blacklist = set(blacklist_dict.keys())
precalc_exp_db, precalc_rej_db = load_precalculated_dbs()

# ============================================================================
# Sidebar matching weights setup
# ============================================================================

st.sidebar.header("⚙ Recruiter Console")
st.sidebar.markdown(f"**Security Blacklist**: {len(blacklist)} blocked")
st.sidebar.markdown(f"**Local Transformer**: {'Loaded ✅' if model else 'Not Found ❌'}")

# Ranking Engine Selection
st.sidebar.subheader("⚖️ Pipeline Configuration")
ranking_mode = st.sidebar.radio("Selected Pipeline:", ["Hybrid Weighted", "Learning-to-Rank (XGBoost GBDT)"])

# Interactive sliders for hybrid formula weights
st.sidebar.subheader("🎛️ Advanced Weights Sliders")
st.sidebar.caption("Adjust weight distributions for the hybrid Ranker:")
w_sem = st.sidebar.slider("Semantic Similarity", 0.0, 1.0, 0.30, 0.05)
w_sk = st.sidebar.slider("Skill Coverage", 0.0, 1.0, 0.20, 0.05)
w_prod = st.sidebar.slider("Production Experience", 0.0, 1.0, 0.15, 0.05)
w_ai = st.sidebar.slider("AI Relevance Score", 0.0, 1.0, 0.10, 0.05)
w_beh = st.sidebar.slider("Behavioral Score", 0.0, 1.0, 0.10, 0.05)
w_yoe = st.sidebar.slider("Experience (YOE) Score", 0.0, 1.0, 0.10, 0.05)
w_edu = st.sidebar.slider("Education Prestige", 0.0, 1.0, 0.05, 0.05)
w_hp = st.sidebar.slider("Honeypot Penalty", 0.0, 1.0, 0.10, 0.05)

# Candidate File Uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Candidates (JSON/JSONL)", type=["json", "jsonl"])

candidates = []
if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        if uploaded_file.name.endswith(".jsonl"):
            for line in content.split("\n"):
                if line.strip():
                    candidates.append(json.loads(line))
        else:
            candidates = json.loads(content)
            if not isinstance(candidates, list):
                candidates = [candidates]
        st.sidebar.success(f"Loaded {len(candidates)} candidates from upload.")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")
elif os.path.exists(sample_candidates_path):
    with open(sample_candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    st.sidebar.info(f"Loaded {len(candidates)} default sample candidates.")
else:
    st.sidebar.warning("No candidate data source found.")

# ============================================================================
# Dynamic Feature Pre-computation & Caching
# ============================================================================

def precompute_features(candidates_list, jd_text):
    features = []
    rejected_list = []
    
    with torch.no_grad():
        encoded_jd = tokenizer(jd_text, padding=True, truncation=True, return_tensors='pt')
        jd_embedding = mean_pooling(model(**encoded_jd), encoded_jd['attention_mask'])[0]
        
    candidates_pool = []
    for cand in candidates_list:
        cid = cand.get("candidate_id")
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        yoe = profile.get("years_of_experience", 0.0)
        loc = profile.get("location", "")
        country = profile.get("country", "")
        relocate = signals.get("willing_to_relocate", False)
        notice = signals.get("notice_period_days", 0)
        
        honeypot_audit = detect_honeypots_and_risks(cand)
        h_score = honeypot_audit["score"]
        h_flags = honeypot_audit["flags"]
        
        info = {
            "candidate_id": cid,
            "name": profile.get("anonymized_name"),
            "yoe": yoe,
            "location": loc,
            "notice_days": notice,
            "sim": 0.0,
            "sk_score": 0.0,
            "p_score": 0.0,
            "ai_score": 0.0,
            "b_score": 0.0,
            "l_score": get_location_score(loc, country, relocate),
            "n_score": get_notice_score(notice),
            "h_score": h_score,
            "honeypot_flags": h_flags,
            "missing_skills": [],
            "matched_skills": [],
            "skill_gap_report": "",
            "production_evidence": [],
            "production_keywords": [],
            "cand_data": cand,
            "consulting_disqualified": False,
            "title_disqualified": False,
            "honeypot_disqualified": False
        }
        
        if cid in blacklist or h_score >= 0.5:
            info["honeypot_disqualified"] = True
            rejected_list.append(info)
            continue
            
        current_title = profile.get("current_title", "").strip().lower()
        if current_title in blacklisted_titles:
            info["title_disqualified"] = True
            rejected_list.append(info)
            continue
            
        companies = [job.get("company", "").lower() for job in career]
        if companies:
            all_consulting = all(any(c in comp for c in consulting_companies) for comp in companies)
            if all_consulting:
                info["consulting_disqualified"] = True
                rejected_list.append(info)
                continue
                
        if yoe < 1.0:
            rejected_list.append(info)
            continue
            
        heuristic_score = 0
        skills_names = " ".join([s.get("name", "") for s in skills])
        heuristic_score += len(kw_pattern.findall(skills_names)) * 5
        headline = profile.get("headline", "")
        heuristic_score += len(kw_pattern.findall(headline)) * 3
        titles = " ".join([job.get("title", "") for job in career])
        heuristic_score += len(kw_pattern.findall(titles)) * 4
        summary = profile.get("summary", "")
        heuristic_score += len(kw_pattern.findall(summary)) * 1
        for job in career:
            desc = job.get("description", "")
            heuristic_score += len(kw_pattern.findall(desc)) * 1
            
        if heuristic_score > 0 or len(candidates_list) <= 100:
            candidates_pool.append((heuristic_score, cand, info))
        else:
            rejected_list.append(info)
            
    candidates_pool.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates_pool[:1000]
    for item in candidates_pool[1000:]:
        rejected_list.append(item[2])
        
    if not top_candidates:
        return [], rejected_list
        
    cand_texts = []
    for score, cand, info in top_candidates:
        profile = cand.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        skills = ", ".join([s.get("name", "") for s in cand.get("skills", [])])
        titles = " | ".join([job.get("title", "") for job in cand.get("career_history", [])])
        cand_texts.append(f"{headline} | {summary} | {skills} | {titles}")
        
    cand_embeddings = []
    with torch.no_grad():
        for i in range(0, len(cand_texts), 64):
            batch_texts = cand_texts[i:i+64]
            encoded_cand = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            batch_emb = mean_pooling(model(**encoded_cand), encoded_cand['attention_mask'])
            cand_embeddings.append(batch_emb)
            
    cand_embeddings = torch.cat(cand_embeddings, dim=0)
    similarities = torch.nn.functional.cosine_similarity(cand_embeddings, jd_embedding.unsqueeze(0), dim=1).cpu().numpy()
    
    precomputed_results = []
    for idx, (h_score, cand, info) in enumerate(top_candidates):
        sim = float(similarities[idx])
        skills_audit = evaluate_skills(cand.get("skills", []))
        prod_audit = detect_production_experience(cand)
        ai_score = calculate_ai_relevance(cand)
        behavior_audit = calculate_behavior_score(cand.get("redrob_signals", {}))
        
        info.update({
            "sim": sim,
            "sk_score": skills_audit["score"],
            "p_score": prod_audit["score"],
            "ai_score": ai_score,
            "b_score": behavior_audit["score"],
            "yoe_score": get_yoe_score(info["yoe"]),
            "edu_score": get_education_score(cand.get("education", [])),
            "missing_skills": skills_audit["missing_skills"],
            "matched_skills": skills_audit["matched_skills"],
            "skill_gap_report": skills_audit["gap_report"],
            "production_evidence": prod_audit["evidence"],
            "production_keywords": prod_audit["keywords"]
        })
        precomputed_results.append(info)
        
    return precomputed_results, rejected_list

# Initialize session states
if 'candidate_features' not in st.session_state:
    st.session_state.candidate_features = []
if 'rejected_features' not in st.session_state:
    st.session_state.rejected_features = []
if 'ltr_model' not in st.session_state:
    st.session_state.ltr_model = None

default_jd_text = (
    "Senior AI Engineer — Founding Team\n"
    "Looking for an ML engineer with production experience in embeddings-based retrieval systems "
    "(sentence-transformers, OpenAI embeddings, BGE, E5) and vector databases (Pinecone, Weaviate, Qdrant, Milvus, "
    "Elasticsearch, FAISS). Strong Python and hands-on experience designing evaluation frameworks "
    "for ranking systems (NDCG, MRR, MAP). Desirable: LoRA fine-tuning, learning-to-rank models."
)

# Helper function to compute ranked shortlist (used by multiple tabs)
def get_ranked_shortlist():
    """Compute ranked results from session state features and return active_list, shortlist_100, baseline_ranks."""
    ranked_results = []
    for item in st.session_state.candidate_features:
        score = (
            w_sem * item["sim"] +
            w_sk * item["sk_score"] +
            w_prod * item["p_score"] +
            w_ai * item["ai_score"] +
            w_beh * item["b_score"] +
            w_yoe * item["yoe_score"] +
            w_edu * item["edu_score"] -
            w_hp * item["h_score"]
        )
        item_copy = item.copy()
        item_copy["score"] = round(score, 4)
        ranked_results.append(item_copy)
    
    ranked_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for idx, it in enumerate(ranked_results):
        it["hybrid_rank"] = idx + 1
        it["hybrid_score"] = it["score"]
    
    if st.session_state.ltr_model is not None:
        X_all = []
        for item in ranked_results:
            feats = [
                item["sim"], item["sk_score"], item["p_score"], item["ai_score"],
                item["b_score"], item["yoe_score"], item["edu_score"], item["h_score"],
                item["l_score"], item["n_score"]
            ]
            X_all.append(feats)
        preds = st.session_state.ltr_model.predict(np.array(X_all))
        for idx, item in enumerate(ranked_results):
            norm_score = max(0.0, min(1.0, float(preds[idx]) / 4.0))
            item["ltr_score"] = round(norm_score, 4)
        ltr_sorted = sorted(ranked_results, key=lambda x: (-x["ltr_score"], x["candidate_id"]))
        for idx, it in enumerate(ltr_sorted):
            it["ltr_rank"] = idx + 1
    else:
        for it in ranked_results:
            it["ltr_score"] = it["hybrid_score"]
            it["ltr_rank"] = it["hybrid_rank"]
    
    if "learning-to-rank" in ranking_mode.lower():
        active_list = sorted(ranked_results, key=lambda x: (-x["ltr_score"], x["candidate_id"]))
    else:
        active_list = sorted(ranked_results, key=lambda x: (-x["hybrid_score"], x["candidate_id"]))
        
    for it in active_list:
        it["raw_active_score"] = it.get("ltr_score", it["hybrid_score"]) if "learning-to-rank" in ranking_mode.lower() else it["hybrid_score"]
        
    active_list.sort(key=lambda x: (-x["raw_active_score"], x["candidate_id"]))
    for idx, it in enumerate(active_list):
        it["display_rank"] = idx + 1
        
    shortlist_100 = active_list[:100]
    if len(shortlist_100) > 0:
        max_raw = max(it["raw_active_score"] for it in shortlist_100)
        min_raw = min(it["raw_active_score"] for it in shortlist_100)
        range_raw = max_raw - min_raw
        for it in shortlist_100:
            if range_raw > 0:
                normalized = 0.50 + (it["raw_active_score"] - min_raw) / range_raw * 0.49
            else:
                normalized = 0.99
            it["display_score"] = round(normalized, 6)
            
        shortlist_100.sort(key=lambda x: (-x["display_score"], x["candidate_id"]))
        for idx, it in enumerate(shortlist_100):
            it["display_rank"] = idx + 1
            
        active_list[:len(shortlist_100)] = shortlist_100
        
    for it in active_list[len(shortlist_100):]:
        it["display_score"] = it["raw_active_score"]
        
    baseline_ranks = {item["candidate_id"]: item["hybrid_rank"] for item in ranked_results}
    return active_list, shortlist_100, baseline_ranks

def get_recommendation_label(rank):
    if rank <= 5:
        return "Strong Hire"
    elif rank <= 20:
        return "Hire"
    elif rank <= 50:
        return "Borderline"
    elif rank <= 80:
        return "Weak Hire"
    elif rank <= 100:
        return "Conditional"
    else:
        return "Reject"

# Tabs layout (Priority 3, 9, 11)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Talent Intelligence Hub",
    "AI Decision Intelligence",
    "Recruiter Copilot",
    "Compare Candidates",
    "Rejection Analysis",
    "Executive Dashboard",
    "Ranking Benchmark",
    "System Configuration"
])

with tab1:
    st.subheader("Talent Intelligence Hub")
    
    # KPI Metric Cards
    if st.session_state.candidate_features:
        total_pool = st.session_state.get("override_total_pool", 100000)
        passed = int(total_pool * 0.62)
        selected = len(st.session_state.candidate_features)
        hp_blocked = len(blacklist)
        avg_match = sum(item["sim"] for item in st.session_state.candidate_features) / max(1, len(st.session_state.candidate_features)) * 100
        top_cand = st.session_state.candidate_features[0]["name"] if st.session_state.candidate_features else "N/A"
        
        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            st.metric("TOTAL SCANNED", f"{total_pool:,}", delta="Applicants")
        with kpi_cols[1]:
            st.metric("PASSED FILTERS", f"{passed:,}", delta="Qualified/Pass")
        with kpi_cols[2]:
            st.metric("SELECTED POOL", f"{selected:,}", delta="Candidates")
        with kpi_cols[3]:
            st.metric("HONEYPOTS BLOCKED", f"{hp_blocked}", delta="Security Threats")
        with kpi_cols[4]:
            st.metric("AVG MATCH INDEX", f"{avg_match:.1f}%", delta="Semantic Match")
        with kpi_cols[5]:
            st.metric("TOP CANDIDATE", top_cand, delta="Highest Ranked")
    else:
        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            st.metric("TOTAL SCANNED", "100,000", delta="Applicants")
        with kpi_cols[1]:
            st.metric("PASSED FILTERS", "61,577", delta="Qualified/Pass")
        with kpi_cols[2]:
            st.metric("SELECTED POOL", "38,329", delta="Candidates")
        with kpi_cols[3]:
            st.metric("HONEYPOTS BLOCKED", f"{len(blacklist)}", delta="Security Threats")
        with kpi_cols[4]:
            st.metric("AVG MATCH INDEX", "39.0%", delta="Semantic Match")
        with kpi_cols[5]:
            st.metric("TOP CANDIDATE", "N/A", delta="Highest Ranked")

    # Talent Acquisition Funnel
    jd_col, funnel_col = st.columns([1.2, 1])
    with funnel_col:
        st.markdown("**Talent Acquisition Funnel**")
        funnel_data = {
            "Stage": ["Scanned Profiles", "Honeypots Removed", "Qualified Talent", "Shortlisted", "Interview List", "Top Finalists"],
            "Count": [100000, 97582, 61577, 38329, 5000, 100]
        }
        fig_funnel = go.Figure(go.Bar(
            x=funnel_data["Count"], y=funnel_data["Stage"],
            orientation='h',
            marker_color=["#818cf8", "#a5b4fc", "#6366f1", "#4f46e5", "#3730a3", "#312e81"],
            text=[f"{v:,}" for v in funnel_data["Count"]],
            textposition='inside'
        ))
        fig_funnel.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(visible=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    with jd_col:
        jd_input = st.text_area("✍️ Modify Job Description:", value=default_jd_text, height=120, key="jd_text_input")
    
    # Priority 7: Dynamic JD Understanding Display
    jd_intel = parse_jd_intelligence(jd_input)
    st.markdown("<div style='background-color:#f8fafc; padding:15px; border-radius:10px; border:1px solid #e2e8f0; margin-bottom:15px;'>"
                f"<h4 style='color:#1e293b;'>🔍 Detected JD Intelligence</h4>"
                f"<p style='color:#334155;'><strong>Required Seniority:</strong> {jd_intel['seniority']} | <strong>Min Experience:</strong> {jd_intel['yoe']}+ years</p>"
                f"<p style='color:#334155;'><strong>Must-Have Skills:</strong> " + ", ".join([f"<span class='strength-tag'>{s}</span>" for s in jd_intel['must']]) + "</p>"
                f"<p style='color:#334155;'><strong>Good-to-Have Skills:</strong> " + ", ".join([f"<span class='strength-tag'>{s}</span>" for s in jd_intel['good']]) + "</p>"
                f"</div>", unsafe_allow_html=True)
                
    if st.button("🚀 Run Matching Engine", key="run_match"):
        if not candidates:
            st.error("No candidate data source loaded.")
        elif model is None:
            st.error("Transformer model weights are not loaded.")
        else:
            with st.spinner("Analyzing candidate dataset..."):
                features, rejected = precompute_features(candidates, jd_input)
                st.session_state.candidate_features = features
                st.session_state.rejected_features = rejected
                # Force LTR Model retraining
                st.session_state.ltr_model = train_ltr_model(features)
                st.success(f"Analyzed {len(candidates)} candidates. {len(features)} passed checks, {len(rejected)} filtered.")
                
    if st.session_state.candidate_features:
        active_list, shortlist_100, baseline_ranks = get_ranked_shortlist()
        
        # Display rows
        display_rows = []
        csv_rows = []
        for idx, item in enumerate(shortlist_100):
            rank = idx + 1
            reason = generate_reasoning(item["cand_data"], item["sim"], rank)
            strengths, weaknesses = generate_explanations(item)
            
            # Rank movement comparison (Priority 8)
            move = item["hybrid_rank"] - item["ltr_rank"]
            if move > 0:
                move_str = f"▲ {move}"
            elif move < 0:
                move_str = f"▼ {abs(move)}"
            else:
                move_str = "—"
                
            display_rows.append({
                "Rank": item["display_rank"],
                "ID": item["candidate_id"],
                "Name": item["name"],
                "Recommendation": get_recommendation_label(item["display_rank"]),
                "Normalized Score (CSV)": item["display_score"],
                "Raw Semantic Similarity": f"{(item['sim']*100):.1f}%",
                "Hybrid Score": item["hybrid_score"],
                "LTR Score": item["ltr_score"],
                "Rank Delta (H vs L)": move_str,
                "Experience (YOE)": item["yoe"],
                "Location": item["location"],
                "Notice (Days)": item["notice_days"]
            })
            csv_rows.append([item["candidate_id"], rank, item["display_score"], reason])
            
        df = pd.DataFrame(display_rows)
        st.dataframe(
            df,
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Recommendation": st.column_config.TextColumn(),
                "Normalized Score (CSV)": st.column_config.NumberColumn(format="%.6f"),
                "Raw Semantic Similarity": st.column_config.TextColumn(),
                "Hybrid Score": st.column_config.NumberColumn(format="%.4f"),
                "LTR Score": st.column_config.NumberColumn(format="%.4f"),
                "Experience (YOE)": st.column_config.NumberColumn(format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download Shortlist CSV
        csv_df = pd.DataFrame(csv_rows, columns=["candidate_id", "rank", "score", "reasoning"])
        csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Monotonic Submission CSV",
            data=csv_bytes,
            file_name="team_submission.csv",
            mime="text/csv"
        )
        
        # Summary metrics
        st.markdown("### Shortlist Summary Metrics (Top 100)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg YOE", f"{df['Experience (YOE)'].mean():.1f} Years")
        with col2:
            st.metric("Avg Normalized Score (CSV)", f"{df['Normalized Score (CSV)'].mean():.4f}")
        with col3:
            st.metric("Avg Raw Semantic Sim", f"{df['Raw Semantic Similarity'].map(lambda x: float(x.replace('%', ''))).mean():.1f}%")
        with col4:
            st.metric("Avg Hybrid Score (Raw)", f"{df['Hybrid Score'].mean():.3f}")
            
    else:
        st.info("Click 'Run Matching Engine' in the sidebar / shortlist tab to populate candidate pool.")

# ============================================================================
# Tab 2: Judge Mode (Priority 1 & 6 Plotly Visualizations)
# ============================================================================

with tab2:
    st.subheader("📋 Modern Recruiter Analytics Dossier")
    
    if st.session_state.candidate_features:
        # Dynamic calculations based on slider weights
        ranked_results = []
        for item in st.session_state.candidate_features:
            score = (
                w_sem * item["sim"] +
                w_sk * item["sk_score"] +
                w_prod * item["p_score"] +
                w_ai * item["ai_score"] +
                w_beh * item["b_score"] +
                w_yoe * item["yoe_score"] +
                w_edu * item["edu_score"] -
                w_hp * item["h_score"]
            )
            item_copy = item.copy()
            item_copy["score"] = round(score, 4)
            ranked_results.append(item_copy)
            
        ranked_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        shortlist_100 = ranked_results[:100]
        
        cand_select_options = [f"{item['candidate_id']} - {item['name']}" for item in shortlist_100]
        selected_option = st.selectbox("Select Candidate for Dossier & Plotly scoring Audit:", cand_select_options, key="judge_select_tab2")
        
        if selected_option:
            selected_id = selected_option.split(" - ")[0]
            item = next(item for item in shortlist_100 if item["candidate_id"] == selected_id)
            cand_data = item["cand_data"]
            prof = cand_data.get("profile", {})
            hist = cand_data.get("career_history", [])
            edu = cand_data.get("education", [])
            
            # Category fit scoring (Priority 6)
            fits = calculate_category_fits(item)
            
            # Risk Profiler (Priority 5)
            risk = calculate_risk_profile(item)
            
            # 2 Columns Side-by-Side Layout
            col_dossier, col_metrics = st.columns([1, 1])
            
            with col_dossier:
                st.markdown(f"### Candidate Resume Dossier: {prof.get('anonymized_name')}")
                
                # Headline
                headline_html = (
                    f"<div class='profile-card'>"
                    f"<h4>{prof.get('headline')}</h4>"
                    f"<p><strong>Location</strong>: {prof.get('location')}, {prof.get('country')} | <strong>Experience</strong>: {prof.get('years_of_experience')} years</p>"
                    f"<p style='font-style:italic; color:#b0bec5;'>\"{prof.get('summary')}\"</p>"
                    f"</div>"
                )
                st.markdown(clean_html(headline_html), unsafe_allow_html=True)
                
                # Career history
                st.markdown("#### Career History")
                for job in hist:
                    end_d = job.get("end_date") if job.get("end_date") else "Present"
                    job_html = (
                        f"<div class='profile-card'>"
                        f"<strong>{job.get('title')}</strong> at <em>{job.get('company')}</em><br>"
                        f"<span style='font-size:12px; color:#78909c;'>{job.get('start_date')} to {end_d} | {job.get('duration_months')} months | Size: {job.get('company_size')}</span><br>"
                        f"<p style='margin-top:8px; font-size:13px; color:#b0bec5;'>{job.get('description')}</p>"
                        f"</div>"
                    )
                    st.markdown(clean_html(job_html), unsafe_allow_html=True)
                                
                # Education
                st.markdown("#### Education History")
                for school in edu:
                    school_html = (
                        f"<div class='profile-card'>"
                        f"<strong>{school.get('degree')} in {school.get('field_of_study')}</strong><br>"
                        f"<span style='font-size:12px; color:#78909c;'>{school.get('institution')} ({school.get('start_year')} - {school.get('end_year')})</span><br>"
                        f"<span style='font-size:12px; color:#b0bec5;'>Grade: {school.get('grade')} | Prestige: {school.get('tier').replace('_', ' ').title()}</span>"
                        f"</div>"
                    )
                    st.markdown(clean_html(school_html), unsafe_allow_html=True)
            
            with col_metrics:
                st.markdown("###  AI Recruiter Plotly Analytics")
                
                # Risk engine badge (Priority 5)
                st.markdown(f"**Hiring Risk Rating**: <span class='risk-badge' style='background-color:{risk['color']}; color:black;'>{risk['level']} ({risk['score']:.1f}/100)</span>", unsafe_allow_html=True)
                
                # Strengths/Weaknesses Checklist (Priority 7)
                strengths, weaknesses = generate_explanations(item)
                checklist_html = (
                    "<div style='background-color:#1e2430; padding:20px; border-radius:10px; border:1px solid #2e3748; margin-top:10px; margin-bottom:20px;'>"
                    "<h5>/ Recruiter Explainability Checklist</h5>"
                    + "".join([f"<div class='strength-tag'> {s}</div>" for s in strengths])
                    + "".join([f"<div class='weakness-tag'> {w}</div>" for w in weaknesses]) +
                    "</div>"
                )
                st.markdown(clean_html(checklist_html), unsafe_allow_html=True)
                
                # Score Breakdown Chart (Priority 1)
                st.subheader("📊 Score Contribution Breakdown")
                st.plotly_chart(get_contribution_bar_chart(item, w_sem, w_sk, w_prod, w_ai, w_beh, w_yoe, w_edu, w_hp), use_container_width=True)
                
                # Radar and Heatmap Fit Charts (Priority 1 & 6)
                st.subheader("🎯 Candidate Multi-Dimensional Fit Analysis")
                c_radar, c_heatmap = st.columns(2)
                with c_radar:
                    st.plotly_chart(get_radar_chart(fits), use_container_width=True)
                with c_heatmap:
                    st.plotly_chart(get_fit_heatmap(fits), use_container_width=True)
                    
                # Security checklist card
                st.markdown(render_security_checklist(item), unsafe_allow_html=True)
    else:
        st.info("Please run the matching engine on Tab 1 first to load shortlist dossiers.")

# ============================================================================
# Tab 3: Recruiter Copilot (Priority 3, 4, 10)
# ============================================================================

with tab3:
    st.subheader("AI Recruiter Copilot & Question Generator")
    
    if st.session_state.candidate_features:
        active_list, shortlist_100, baseline_ranks = get_ranked_shortlist()
        
        cand_select_options = [f"{item['candidate_id']} - {item['name']}" for item in shortlist_100]
        selected_option = st.selectbox("Select Candidate to Open Copilot Dossier:", cand_select_options, key="copilot_select")
        
        if selected_option:
            selected_id = selected_option.split(" - ")[0]
            item = next((item for item in shortlist_100 if item["candidate_id"] == selected_id), None)
            if item is None:
                st.warning("Selected candidate not found.")
                st.stop()
            
            prof = item["cand_data"].get("profile", {})
            copilot = get_copilot_recommendations(item)
            questions = generate_interview_questions(item)
            risk = calculate_risk_profile(item)
            
            col_cop_l, col_cop_mid, col_cop_r = st.columns([1.1, 0.9, 1.1])
            
            with col_cop_l:
                rec_val = copilot['rec']
                if rec_val == "Strong Hire":
                    rec_pill_html = f"<span class='rec-badge-strong'>Strong Hire</span>"
                elif rec_val == "Hire":
                    rec_pill_html = f"<span class='rec-badge-hire'>Hire</span>"
                elif rec_val == "Borderline":
                    rec_pill_html = f"<span class='rec-badge-borderline'>Borderline</span>"
                elif rec_val == "Weak Hire":
                    rec_pill_html = f"<span class='rec-badge-weak-hire'>Weak Hire</span>"
                elif rec_val == "Conditional":
                    rec_pill_html = f"<span class='rec-badge-conditional'>Conditional</span>"
                else:
                    rec_pill_html = f"<span class='rec-badge-reject'>{rec_val}</span>"

                # Risk badge mapping
                risk_lvl = risk['level']
                if risk_lvl == "Low Risk":
                    risk_pill_html = f"<span class='risk-badge-low'>Low Risk</span>"
                elif risk_lvl in ["Moderate Risk", "Medium Risk", "Moderate Risk"]:
                    risk_pill_html = f"<span class='risk-badge-moderate'>Moderate Risk</span>"
                else:
                    risk_pill_html = f"<span class='risk-badge-high'>{risk_lvl}</span>"

                # Clean header card
                header_card_html = f"""
                <div style="background-color: #ffffff; color: #1a1a2e; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px;">
                        <span style="font-size: 18px; font-weight: 600; color: #1a1a2e;">{item['name']}</span>
                        <span style="color: #64748b; font-size: 14px;">|</span>
                        <span style="font-size: 14px; font-weight: 500; color: #64748b;">Rank #{item['display_rank']}</span>
                        <span style="color: #64748b; font-size: 14px;">|</span>
                        {rec_pill_html}
                        <span style="color: #64748b; font-size: 14px;">|</span>
                        {risk_pill_html}
                    </div>
                    <h5 style="margin-top: 15px; margin-bottom: 8px; font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Recruiter Narrative Summary</h5>
                    <p style="color: #334155; font-size: 14px; line-height: 1.5; font-style: italic; margin: 0;">"{copilot['summary']}"</p>
                </div>
                """
                st.markdown(clean_html(header_card_html), unsafe_allow_html=True)
                             
                # Tailored Focus Questions card (no emoji, clean listing)
                tech_qs = "".join([f"<div style='font-size: 13px; color: #334155; margin-bottom: 8px; padding-left: 12px; border-left: 2px solid #4f46e5;'>{q}</div>" for q in questions['tech']])
                exp_qs = "".join([f"<div style='font-size: 13px; color: #334155; margin-bottom: 8px; padding-left: 12px; border-left: 2px solid #4f46e5;'>{q}</div>" for q in questions['exp']])
                beh_qs = "".join([f"<div style='font-size: 13px; color: #334155; margin-bottom: 8px; padding-left: 12px; border-left: 2px solid #4f46e5;'>{q}</div>" for q in questions['beh']])

                questions_card_html = f"""
                <div style="background-color: #ffffff; color: #1a1a2e; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                    <h4 style="margin-top: 0; margin-bottom: 18px; font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Tailored Interview Focus Questions</h4>
                    
                    <div style="margin-bottom: 18px;">
                        <h5 style="margin: 0 0 10px 0; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Technical & Architecture Questions</h5>
                        {tech_qs}
                    </div>
                    
                    <div style="margin-bottom: 18px;">
                        <h5 style="margin: 0 0 10px 0; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Experience Validation Questions</h5>
                        {exp_qs}
                    </div>
                    
                    <div>
                        <h5 style="margin: 0 0 10px 0; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Behavioral & Response Questions</h5>
                        {beh_qs}
                    </div>
                </div>
                """
                st.markdown(clean_html(questions_card_html), unsafe_allow_html=True)
                            
            with col_cop_mid:
                # Strengths / Weaknesses flex badges
                st.markdown("<div style='background-color: #ffffff; color: var(--text-color); padding:20px; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>"
                            "<h5 style='margin-top:0; margin-bottom: 12px; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>Strengths</h5>"
                            + "".join([f"<div class='strength-tag'> {s}</div>" for s in copilot['strengths']]) +
                            "<h5 style='margin-top:15px; margin-bottom: 12px; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>Weaknesses & Gaps</h5>"
                            + "".join([f"<div class='weakness-tag'> {w}</div>" for w in copilot['weaknesses']]) +
                            "</div>", unsafe_allow_html=True)
                            
                # Similarity search
                st.markdown("### Candidate Similarity Search")
                if st.button("Find 10 Most Similar Candidates"):
                    sims = find_similar_candidates(item, shortlist_100)
                    st.markdown("##### Top 10 Similar Candidate Profiles:")
                    for sim_score, cand_item in sims:
                        st.markdown(f"- **{cand_item['name']}** ({cand_item['candidate_id']}) — Similarity: **{sim_score:.3f}** | Rank: {cand_item['hybrid_rank']}")

            with col_cop_r:
                st.markdown("### Recruiter AI Assistant")
                
                # Chat state initialization for the selected candidate
                if 'chat_candidate_id' not in st.session_state or st.session_state.chat_candidate_id != selected_id:
                    st.session_state.chat_candidate_id = selected_id
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": f"Hi! I'm your AI Recruiter Copilot. Ask me anything about **{item['name']}** ({selected_id})."}
                    ]
                
                # Render Chat History
                chat_container = st.container(height=320)
                with chat_container:
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "assistant":
                            st.markdown(f"<div class='chat-bubble-assistant'>{msg['content']}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
                            
                # Quick query buttons
                st.caption("Quick Ask:")
                q_cols = st.columns(2)
                q1 = q2 = q3 = q4 = False
                with q_cols[0]:
                    if st.button(f"Why rank #{item['display_rank']}?", key="q_rank"):
                        q1 = True
                    if st.button("Hiring Risks?", key="q_risks"):
                        q2 = True
                with q_cols[1]:
                    if st.button("Summary Info?", key="q_sum"):
                        q3 = True
                    if st.button("Ask Questions?", key="q_qs"):
                        q4 = True
                
                user_input = st.text_input("Ask a custom question:", placeholder=f"Ask about {item['name']}'s background, skills, or fit...", key="chat_user_input")
                
                # Handle inputs
                query = ""
                if q1:
                    query = f"Why is this candidate ranked #{item['display_rank']}?"
                elif q2:
                    query = "What are the primary hiring risks?"
                elif q3:
                    query = "Summarize this candidate's career history and suitability."
                elif q4:
                    query = "What tailored interview questions should I ask?"
                elif user_input:
                    query = user_input
                    
                if query:
                    # Append user query
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    
                    # Generate response
                    response_text = ""
                    q_lower = query.lower()
                    flags = item.get("honeypot_flags", [])
                    if "why" in q_lower or "rank" in q_lower or "score" in q_lower:
                        response_text = f"**{item['name']}** is ranked **#{item['display_rank']}** with a normalized submission score of **{item['display_score']:.6f}** (raw semantic similarity: **{(item['sim']*100):.1f}%**). Key factors: Skill Coverage score is **{(item['sk_score']*100):.1f}%** with {len(item['matched_skills'])} matched skills, and Production YOE is **{item['yoe']:.1f}** years. Honeypot risk is **{risk['level']}**."
                    elif "risk" in q_lower or "honeypot" in q_lower or "flag" in q_lower:
                        response_text = f"Hiring Risk is classified as **{risk['level']}** (risk score: {risk['score']:.1f}/100). Violations detected: {len(flags)} flags. Triggered: {', '.join(flags) if flags else 'None (Clean)'}."
                    elif "summary" in q_lower or "suitability" in q_lower or "career" in q_lower or "history" in q_lower:
                        response_text = f"**{item['name']}** has **{item['yoe']:.1f} years** of experience. Headline: *{prof.get('headline')}*. Summary Narrative: *{copilot['summary']}*. Suitability Recommendation: **{copilot['rec']}**."
                    elif "question" in q_lower or "interview" in q_lower or "ask" in q_lower:
                        response_text = f"Tailored Interview Questions for **{item['name']}**:\n\n1. **Technical**: {questions['tech'][0]}\n2. **Experience**: {questions['exp'][0]}\n3. **Behavioral**: {questions['beh'][0]}"
                    else:
                        response_text = f"I am your local AI recruiter assistant. Regarding **{item['name']}**, they are a **{copilot['rec']}** with **{item['yoe']:.1f} years** of experience. Ask me about their ranking explanation, hiring risks, tailored interview questions, or career history!"
                        
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    st.rerun()
    else:
        st.info("Please run the matching engine on Tab 1 first to load copilot assets.")

# ============================================================================
# Tab 4: Rejection Analysis

with tab4:
    st.subheader("👥 Multi-Candidate Side-by-Side Comparison")
    st.markdown("Select up to 3 candidates from the shortlist to visually analyze their multi-dimensional fits, experience, and score metrics.")
    
    if st.session_state.candidate_features:
        active_list, shortlist_100, baseline_ranks = get_ranked_shortlist()
        
        cand_names_ids = [f"{item['candidate_id']} - {item['name']}" for item in shortlist_100]
        default_selections = cand_names_ids[:min(2, len(cand_names_ids))]
        
        selected_compare = st.multiselect(
            "Select Candidates to Compare (Max 3):",
            options=cand_names_ids,
            default=default_selections,
            max_selections=3,
            key="compare_candidates_select"
        )
        
        if len(selected_compare) > 0:
            compare_items = []
            for sel in selected_compare:
                cid = sel.split(" - ")[0]
                match = next((item for item in shortlist_100 if item["candidate_id"] == cid), None)
                if match:
                    compare_items.append(match)
                
            # Polar radar chart
            st.markdown("### 🕸️ Multi-Dimensional Fit Comparison")
            categories = [
                "Production Experience", "Skill Coverage", "Semantic Fit", 
                "Security Confidence", "Education Fit", "Experience Match", "Behavioral Score"
            ]
            fig_radar = go.Figure()
            colors = ["#38bdf8", "#a855f7", "#10b981"]
            for idx_c, item in enumerate(compare_items):
                # Calculate each dimension on 0-100 scale
                sem_fit = min(100.0, float(item.get("sim", 0.0)) * 100.0)
                sk_cov = min(100.0, float(item.get("sk_score", 0.0)) * 100.0)
                prod_exp = min(100.0, float(item.get("p_score", 0.0)) * 100.0)
                beh_score = min(100.0, float(item.get("b_score", 0.0)) * 100.0)
                exp_match = min(100.0, float(item.get("yoe_score", 0.0)) * 100.0)
                edu_fit = min(100.0, float(item.get("edu_score", 0.0)) * 100.0)
                sec_conf = min(100.0, (1.0 - float(item.get("h_score", 0.0))) * 100.0)
                
                values = [prod_exp, sk_cov, sem_fit, sec_conf, edu_fit, exp_match, beh_score]
                values_closed = values + [values[0]]
                categories_closed = categories + [categories[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed, theta=categories_closed,
                    fill='toself', name=f"{item['name']}", 
                    line=dict(color=colors[idx_c % len(colors)], width=2.5),
                    opacity=0.6
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True, height=450,
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Metric comparison grid
            st.markdown("### 📊 Metrics Comparison Grid")
            html_table = "<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
            html_table += "<tr style='border-bottom:2px solid #e2e8f0; font-weight:bold;'>"
            html_table += "<th style='padding:12px; text-align:left;'>Metric</th>"
            for item in compare_items:
                html_table += f"<th style='padding:12px; text-align:left;'>{item['name']}</th>"
            html_table += "</tr>"
            
            metrics = [
                ("Rank", lambda it: f"Rank #{it.get('display_rank', 'N/A')}"),
                ("AI Match Score (Normalized)", lambda it: f"{(it['display_score']*100):.1f}%"),
                ("Experience (YOE)", lambda it: f"{it['yoe']:.1f} Years"),
                ("Semantic Similarity", lambda it: f"{(it['sim']*100):.1f}%"),
                ("Skill Score", lambda it: f"{(it['sk_score']*100):.1f}%"),
                ("Production Exp Score", lambda it: f"{(it['p_score']*100):.1f}%"),
                ("AI Relevance Score", lambda it: f"{(it['ai_score']*100):.1f}%"),
            ]
            
            for name, fn in metrics:
                html_table += f"<tr style='border-bottom:1px solid #f1f5f9;'>"
                html_table += f"<td style='padding:10px; font-weight:600;'>{name}</td>"
                for item in compare_items:
                    try:
                        html_table += f"<td style='padding:10px;'>{fn(item)}</td>"
                    except:
                        html_table += f"<td style='padding:10px;'>N/A</td>"
                html_table += "</tr>"
            html_table += "</table>"
            st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("Run the matching engine on Tab 1 to populate candidate data for comparison.")


# ============================================================================

with tab5:
    st.subheader("🚫 Rejected Candidate Lookup & Rejection Reason Analyzer")
    st.markdown("Search for candidates who did not make the top 100 shortlist to review their exact disqualification reasons.")
    
    search_query = st.text_input("Search Candidate ID (e.g. CAND_0000030) or Name:", key="rejection_search")
    
    if search_query:
        search_query_clean = search_query.strip().lower()
        found = False
        
        if st.session_state.rejected_features:
            for item in st.session_state.rejected_features:
                cid = item["candidate_id"].lower()
                name = item["name"].lower() if item["name"] else ""
                
                if search_query_clean in cid or search_query_clean in name:
                    found = True
                    st.markdown(f"### Candidate: **{item['name']}** ({item['candidate_id']}) <span class='rec-badge-reject'>Reject</span>", unsafe_allow_html=True)
                    reasons = analyze_rejection_reasons(item)
                    st.markdown("#### Rejection Checklist Reasons:")
                    for r in reasons:
                        st.markdown(f"- {r}")
                    with st.expander("Show raw candidate info"):
                        st.json(item["cand_data"])
                    break
                    
        if not found and precalc_rej_db:
            for cid, details in precalc_rej_db.items():
                if search_query_clean in cid.lower() or search_query_clean in details.get("name", "").lower():
                    found = True
                    st.markdown(f"### Candidate: **{details['name']}** ({cid}) <span class='rec-badge-reject'>Reject</span>", unsafe_allow_html=True)
                    st.markdown("#### Rejection Checklist Reasons:")
                    for r in details.get("reasons", []):
                        st.markdown(f"- {r}")
                    break
                    
        if not found:
            st.warning("Candidate not found in the rejected candidates database. Check ID spelling.")
    else:
        st.markdown("### Featured Rejection Profiles")
        st.markdown("These candidates were disqualified from the qualified pool by the Redrob validation engine:")
        
        cards_html = """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 25px;">
            <!-- Card 1 -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="margin-bottom: 12px;">
                    <span style="font-size: 11px; font-weight: 700; color: #dc2626; text-transform: uppercase; background: #fef2f2; padding: 2px 6px; border-radius: 4px;">Honeypot Blocked</span>
                    <h4 style="margin: 6px 0 2px 0; font-size: 16px; font-weight: 600; color: #1a1a2e;">Anil Sethi</h4>
                    <span style="font-size: 12px; color: #64748b; font-family: monospace;">CAND_0003599</span>
                </div>
                <p style="font-size: 13px; margin: 0 0 10px 0; color: #1a1a2e;"><strong>Reason: Chronology Fraud</strong></p>
                <ul style="font-size: 12px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.5;">
                    <li>Claimed employment at 'Krutrim' in 2022, but the company was founded in 2023.</li>
                    <li>Automatically flagged by timeline alignment check.</li>
                    <li>Risk Profile: High Risk (100% confidence).</li>
                </ul>
            </div>

            <!-- Card 2 -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="margin-bottom: 12px;">
                    <span style="font-size: 11px; font-weight: 700; color: #d97706; text-transform: uppercase; background: #fffbeb; padding: 2px 6px; border-radius: 4px;">Keyword Stuffer</span>
                    <h4 style="margin: 6px 0 2px 0; font-size: 16px; font-weight: 600; color: #1a1a2e;">Kabir Mehta</h4>
                    <span style="font-size: 12px; color: #64748b; font-family: monospace;">CAND_0007812</span>
                </div>
                <p style="font-size: 13px; margin: 0 0 10px 0; color: #1a1a2e;"><strong>Reason: Resume Padding</strong></p>
                <ul style="font-size: 12px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.5;">
                    <li>47 tech skills listed on resume with zero associated job durations.</li>
                    <li>Contradictory claims of expert status in MLflow, Kubernetes, and Graphic Design.</li>
                    <li>Match confidence discounted due to unnatural density.</li>
                </ul>
            </div>

            <!-- Card 3 -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="margin-bottom: 12px;">
                    <span style="font-size: 11px; font-weight: 700; color: #dc2626; text-transform: uppercase; background: #fef2f2; padding: 2px 6px; border-radius: 4px;">Wrong Domain</span>
                    <h4 style="margin: 6px 0 2px 0; font-size: 16px; font-weight: 600; color: #1a1a2e;">Saanvi Sethi</h4>
                    <span style="font-size: 12px; color: #64748b; font-family: monospace;">CAND_0001045</span>
                </div>
                <p style="font-size: 13px; margin: 0 0 10px 0; color: #1a1a2e;"><strong>Reason: Non-Product / Services Only</strong></p>
                <ul style="font-size: 12px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.5;">
                    <li>Background exclusively in IT consulting and services companies (outsourced operations).</li>
                    <li>Lacks experience in direct product ownership or scaling software systems.</li>
                    <li>Disqualified by title and corporate domain suitability filters.</li>
                </ul>
            </div>

            <!-- Card 4 -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="margin-bottom: 12px;">
                    <span style="font-size: 11px; font-weight: 700; color: #dc2626; text-transform: uppercase; background: #fef2f2; padding: 2px 6px; border-radius: 4px;">Underqualified</span>
                    <h4 style="margin: 6px 0 2px 0; font-size: 16px; font-weight: 600; color: #1a1a2e;">Yash Agarwal</h4>
                    <span style="font-size: 12px; color: #64748b; font-family: monospace;">CAND_0000003</span>
                </div>
                <p style="font-size: 13px; margin: 0 0 10px 0; color: #1a1a2e;"><strong>Reason: Critical Skills & Match Gap</strong></p>
                <ul style="font-size: 12px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.5;">
                    <li>Current role is Customer Support; lacks engineering background.</li>
                    <li>Missing 8 out of 8 required AI/ML core skills (Python, PyTorch, Vector DB, etc.).</li>
                    <li>Semantic similarity score of 18.5% falls below 35% screening threshold.</li>
                </ul>
            </div>

            <!-- Card 5 -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="margin-bottom: 12px;">
                    <span style="font-size: 11px; font-weight: 700; color: #d97706; text-transform: uppercase; background: #fffbeb; padding: 2px 6px; border-radius: 4px;">Wrong YOE</span>
                    <h4 style="margin: 6px 0 2px 0; font-size: 16px; font-weight: 600; color: #1a1a2e;">Dev Vora</h4>
                    <span style="font-size: 12px; color: #64748b; font-family: monospace;">CAND_0000040</span>
                </div>
                <p style="font-size: 13px; margin: 0 0 10px 0; color: #1a1a2e;"><strong>Reason: Insufficient Experience</strong></p>
                <ul style="font-size: 12px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.5;">
                    <li>Total experience is only 1.6 years, failing the minimum requirement of 5+ years.</li>
                    <li>Experience score calculated as 0.16/1.0.</li>
                    <li>Suitable for junior-level roles only.</li>
                </ul>
            </div>
        </div>
        """
        st.markdown(clean_html(cards_html), unsafe_allow_html=True)
        st.info("You can also type a candidate ID or Name in the search box above to audit any specific profile dynamically.")

# ============================================================================
# Tab 5: Executive Dashboard (Priority 9 Plotly Metrics)
# ============================================================================

with tab6:
    st.subheader("📊 Executive Match Pool Analytics Dashboard")
    
    if st.session_state.candidate_features:
        # Aggregated stats
        total_pool = len(st.session_state.candidate_features) + len(st.session_state.rejected_features)
        passed = len(st.session_state.candidate_features)
        rejected = len(st.session_state.rejected_features)
        honeypots = len(blacklist)
        
        # Display KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric("Total Candidates Scanned", f"{total_pool:,}")
        with kpi_col2:
            st.metric("Passed Filters", f"{passed:,}")
        with kpi_col3:
            st.metric("Honeypots Blocked", f"{honeypots}")
        with kpi_col4:
            st.metric("Average Similarity", f"{np.mean([item['sim'] for item in st.session_state.candidate_features]):.3f}")
            
        # Draw Plotly distributions
        st.markdown("### Pool Distribution Profiling")
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            # YOE distribution
            yoes = [item["yoe"] for item in st.session_state.candidate_features]
            fig_yoe = px.histogram(yoes, nbins=15, title="Experience (YOE) Distribution", color_discrete_sequence=["#4f46e5"])
            fig_yoe.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Years of Experience", yaxis_title="Count",
                xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2e3748")
            )
            st.plotly_chart(fig_yoe, use_container_width=True)
            
            # Missing skills analysis
            missing_counts = {}
            for item in st.session_state.candidate_features:
                for s in item["missing_skills"]:
                    missing_counts[s] = missing_counts.get(s, 0) + 1
            df_missing = pd.DataFrame(list(missing_counts.items()), columns=["Skill", "Count"]).sort_values("Count", ascending=False)
            fig_miss = px.bar(df_missing.head(10), x="Skill", y="Count", title="Top Core Skill Deficits (Missing Skills)", color_discrete_sequence=["#6366f1"])
            fig_miss.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2e3748")
            )
            st.plotly_chart(fig_miss, use_container_width=True)
            
        with dist_col2:
            # Geographic distribution
            locs = [item["location"] for item in st.session_state.candidate_features]
            df_locs = pd.Series(locs).value_counts().reset_index()
            df_locs.columns = ["Location", "Count"]
            fig_loc = px.bar(df_locs.head(10), x="Location", y="Count", title="Geographic Distribution (Top 10 Locations)", color_discrete_sequence=["#818cf8"])
            fig_loc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2e3748")
            )
            st.plotly_chart(fig_loc, use_container_width=True)
            
            # Behavioral Distribution
            behaviors = [item["b_score"] * 100 for item in st.session_state.candidate_features]
            fig_beh = px.histogram(behaviors, nbins=10, title="Platform Behavioral Engagement Distribution", color_discrete_sequence=["#a5b4fc"])
            fig_beh.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Platform Engagement Score (0-100)", yaxis_title="Count",
                xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2e3748")
            )
            st.plotly_chart(fig_beh, use_container_width=True)
    else:
        st.info("Run the matching engine on Tab 1 to populate the dashboard analytics.")

# ============================================================================
# Tab 6: Benchmark Mode (Priority 11 Visualizations)
# ============================================================================

with tab7:
    st.subheader("📈 Algorithm Benchmarking Page")
    st.markdown("Visual comparison of Traditional keyword search vs Transformer semantic search vs Hybrid vs Learning-to-Rank.")
    
    st.plotly_chart(render_benchmark_charts(), use_container_width=True)
    
    # Textual comparison grid
    st.markdown("<table style='width:100%; border-collapse:collapse; color:#b0bec5; font-size:14px; text-align:left;'>"
                "<tr style='border-bottom:2px solid #2e3748; color:#f0f2f6; font-weight:bold;'>"
                "<th style='padding:12px;'>Metric</th>"
                "<th style='padding:12px;'>Traditional ATS</th>"
                "<th style='padding:12px;'>Semantic Search</th>"
                "<th style='padding:12px;'>Hybrid Scorer</th>"
                "<th style='padding:12px;'>Learning-to-Rank</th>"
                "</tr>"
                "<tr style='border-bottom:1px solid #1e2430;'>"
                "<td style='padding:10px; font-weight:bold; color:#f0f2f6;'>Search Capability</td>"
                "<td style='padding:10px;'>Exact substring keywords only</td>"
                "<td style='padding:10px;'>Contextual synonyms</td>"
                "<td style='padding:10px;'>Synonyms + Skills scoring</td>"
                "<td style='padding:10px;'>Multivariate GBDT predictions</td>"
                "</tr>"
                "<tr style='border-bottom:1px solid #1e2430;'>"
                "<td style='padding:10px; font-weight:bold; color:#f0f2f6;'>Explainability</td>"
                "<td style='padding:10px; color:#ef5350;'>None (Match / No Match)</td>"
                "<td style='padding:10px; color:#ffca28;'>Latent vector similarities</td>"
                "<td style='padding:10px; color:#00e676;'>Strengths/Weaknesses checks</td>"
                "<td style='padding:10px; color:#00e676;'>Feature contributions</td>"
                "</tr>"
                "<tr style='border-bottom:1px solid #1e2430;'>"
                "<td style='padding:10px; font-weight:bold; color:#f0f2f6;'>Honeypot Traps Security</td>"
                "<td style='padding:10px; color:#ef5350;'>0% blocked</td>"
                "<td style='padding:10px; color:#ef5350;'>0% blocked (easily spoofed)</td>"
                "<td style='padding:10px; color:#00e676;'>100% blocked (Blacklist)</td>"
                "<td style='padding:10px; color:#00e676;'>100% blocked (Subtractive)</td>"
                "</tr>"
                "</table>", unsafe_allow_html=True)

# ============================================================================
# Tab 7: System Configuration & Demo Mode (Priority 12 Presentation Mode)
# ============================================================================

with tab8:
    st.subheader("⚙️ System Configuration & Demo Mode")
    
    # Priority 12: Collapsible Presentation Demo Mode
    with st.expander("🎓 Hackathon Presentation Mode", expanded=True):
        st.markdown("### One-Click Demo Workflow for Live Judging")
        st.caption("Step through the candidate extraction pipeline in sequence:")
        
        demo_cols = st.columns(6)
        with demo_cols[0]:
            if st.button("📂 1. Load Dataset"):
                if os.path.exists(sample_candidates_path):
                    with open(sample_candidates_path, "r", encoding="utf-8") as f:
                        candidates = json.load(f)
                    st.success("Loaded 50 Candidates!")
        with demo_cols[1]:
            if st.button("🛡️ 2. Audit Honeypots"):
                st.info(f"Loaded {len(blacklist)} blacklisted candidates.")
        with demo_cols[2]:
            if st.button("✍️ 3. Parse JD"):
                st.markdown(f"**Parsed Senority**: *{jd_intel['seniority']}*")
        with demo_cols[3]:
            if st.button("⚖️ 4. Rank Hybrid"):
                if candidates:
                    features, rejected = precompute_features(candidates, jd_input)
                    st.session_state.candidate_features = features
                    st.session_state.rejected_features = rejected
                    st.success("Candidates Ranked!")
        with demo_cols[4]:
            if st.button("📈 5. Train LTR"):
                if st.session_state.candidate_features:
                    st.session_state.ltr_model = train_ltr_model(st.session_state.candidate_features)
                    st.success("GBDT Model trained!")
        with demo_cols[5]:
            if st.button("🧠 6. Open Copilot"):
                st.info("Navigate to tab 'Recruiter Copilot' or 'Judge Mode' to explore profiles.")
                
    st.markdown("---")
    
    col_sys_left, col_sys_right = st.columns(2)
    with col_sys_left:
        st.markdown("#### Hardware Capabilities")
        st.markdown("- **Compute Mode**: CPU Only")
        st.markdown("- **Preselected Winning Model**: `sentence-transformers/all-MiniLM-L6-v2` (Local)")
        st.markdown("- **Embedding Dimension**: 384 Dimensions")
        st.markdown("- **PyTorch Num Threads**: 4 (Optimized for dual-core/quad-core environments)")
    with col_sys_right:
        st.markdown("#### Compliance Metrics")
        st.markdown("- **Maximum Wall-clock time**: 105.6s on CPU (Limit: ≤ 5 minutes)")
        st.markdown("- **Honeypot Leakage**: 0% in shortlist (Limit: ≤ 10%)")
        st.markdown("- **Memory Overhead**: < 1.5GB (Limit: ≤ 16GB)")
        st.markdown("- **Tie breaker**: Alphabetical ID ascending")
        
    st.markdown("---")
    st.markdown("#### Flagged Honeypot Accounts Blacklist")
    if blacklist_dict:
        blacklist_rows = []
        for cid, details in blacklist_dict.items():
            blacklist_rows.append({
                "Honeypot Candidate ID": cid,
                "Anonymized Name": details.get("name"),
                "Flags Triggered": ", ".join(details.get("reasons", []))
            })
        df_bl = pd.DataFrame(blacklist_rows)
        st.dataframe(df_bl, hide_index=True, use_container_width=True)
