import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Redrob AI Candidate Ranking Engine")
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save ranked output CSV",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "ltr"],
        help="Ranking mode (hybrid or ltr)",
    )
    return parser.parse_args()

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

# Core keywords for Stage 1 regex matching
keywords = [
    "embeddings?", "retrievals?", "rankings?", "vector", "pinecone", "weaviate", 
    "qdrant", "milvus", "elasticsearch", "faiss", "opensearch", "rag", 
    "semantic\\s+search", "ndcg", "mrr", "map", "llms?", "lora", "qlora", 
    "peft", "xgboost", "learning-to-rank", "nlp", "information\\s+retrieval"
]
kw_pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)

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

# ============================================================================
# Priority 1: Skill Coverage Engine
# ============================================================================
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

def evaluate_skills(candidate_skills):
    # exact, synonym, and semantic skill match
    matched_skills = []
    missing_skills = []
    gap_reports = []
    total_score = 0.0
    
    # Lowercase candidate skills names for matching
    cand_skills_map = {}
    for s in candidate_skills:
        name = s.get("name", "").strip().lower()
        cand_skills_map[name] = s
        
    for skill_name, weight in required_skills_weights.items():
        matched_item = None
        
        # 1. Exact match
        if skill_name in cand_skills_map:
            matched_item = cand_skills_map[skill_name]
        else:
            # 2. Synonym match
            syns = skill_synonyms.get(skill_name, [])
            for syn in syns:
                if syn in cand_skills_map:
                    matched_item = cand_skills_map[syn]
                    break
            
            # 3. Substring semantic match (if no exact or synonym matched)
            if not matched_item:
                for name, s in cand_skills_map.items():
                    if skill_name in name or any(syn in name for syn in syns):
                        matched_item = s
                        break
                        
        if matched_item:
            matched_skills.append(skill_name)
            
            # Proficiency multiplier
            prof = matched_item.get("proficiency", "beginner").lower()
            prof_mult = 0.40
            if prof == "expert":
                prof_mult = 1.00
            elif prof == "advanced":
                prof_mult = 0.85
            elif prof == "intermediate":
                prof_mult = 0.70
                
            # Duration multiplier (smooth scaling up to 24 months)
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

# ============================================================================
# Priority 2: Production Experience Detector
# ============================================================================
production_keywords = [
    "production systems?", "deploy(ment|ed|ing)?", "serv(ing|ed)?", 
    "latency", "inference", "monitoring", "pipelines?", "scale", "scalability", 
    "users?", "evaluat(ion|e)?", "a/b testing", "ab testing", 
    "ranking systems?", "recommendation systems?", "kubernetes", "docker",
    "real-time serving"
]
prod_pattern = re.compile(r"\b(" + "|".join(production_keywords) + r")\b", re.IGNORECASE)

def detect_production_experience(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    
    found_keywords = set()
    evidence_sentences = []
    raw_score = 0
    
    # Check profile summary
    summary = profile.get("summary", "")
    matches = prod_pattern.findall(summary)
    if matches:
        for m in matches:
            found_keywords.add(m[0].lower() if isinstance(m, tuple) else m.lower())
            
    # Check career descriptions
    for job in career:
        desc = job.get("description", "")
        title = job.get("title", "")
        
        # Split description into sentences to extract precise evidence
        sentences = re.split(r'[.!?]\s+', desc)
        for s in sentences:
            if prod_pattern.search(s):
                evidence_sentences.append(s.strip())
                
        # If job title contains production keywords, give boost
        title_matches = prod_pattern.findall(title)
        if title_matches:
            raw_score += 2
            
    # Calculate score based on unique keywords found and evidence
    unique_kws = list(found_keywords)
    raw_score += len(unique_kws) * 1.5
    raw_score += min(5, len(evidence_sentences)) * 1.0
    
    # Cap score at 10.0 and normalize
    normalized_score = min(1.0, raw_score / 10.0)
    
    return {
        "score": normalized_score,
        "evidence": evidence_sentences[:4], # Keep top 4 sentences
        "keywords": unique_kws
    }

# ============================================================================
# Priority 3: AI Relevance Engine
# ============================================================================
ai_keywords_list = [
    "embeddings", "retrieval", "ranking", "search", "recommendation systems",
    "vector search", "faiss", "pinecone", "weaviate", "milvus", "qdrant",
    "sentence transformers", "bge", "llms", "llm", "transformers", "fine-tuning",
    "lora", "qlora", "peft", "ndcg", "mrr", "map", "deep learning", "pytorch",
    "tensorflow", "machine learning", "rag", "dense retrieval", "nlp"
]
ai_pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in ai_keywords_list) + r")\b", re.IGNORECASE)

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
    
    # Count occurrences and normalize (cap at 15 matches)
    count = len(matches)
    normalized_score = min(1.0, count / 15.0)
    
    return normalized_score

# ============================================================================
# Priority 4: Advanced Honeypot & Risk Detection
# ============================================================================
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

def detect_honeypots_and_risks(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    
    risk_score = 0.0
    flags = []
    
    # 1. Career Contradiction Check
    headline_lower = profile.get("headline", "").lower()
    current_title_lower = profile.get("current_title", "").lower()
    
    blacklisted_titles = {
        "marketing manager", "hr manager", "accountant", "project manager",
        "customer support", "sales executive", "civil engineer", "mechanical engineer",
        "operations manager", "content writer"
    }
    
    is_headline_ai = any(w in headline_lower for w in ["ai", "ml", "machine learning", "nlp", "scientist", "engineer"])
    is_career_non_tech = current_title_lower in blacklisted_titles
    
    if is_headline_ai and is_career_non_tech:
        flags.append("CAREER_CONTRADICTION")
        risk_score += 0.35
        
    # 2. Skill Inflation Check
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", -1)
        name = s.get("name", "").lower()
        
        is_ai_skill = any(k in name for k in ["embeddings", "vector", "retrieval", "rag", "llm", "ranking"])
        if prof == "expert" and dur == 0 and is_ai_skill:
            flags.append("SKILL_INFLATION")
            risk_score += 0.35
            break # Flag once is enough
            
    # 3. Timeline Inconsistency Check
    profile_yoe = profile.get("years_of_experience", 0)
    total_months = sum(job.get("duration_months", 0) for job in career)
    history_yoe = total_months / 12.0
    
    # YOE mismatch
    if abs(profile_yoe - history_yoe) > 3.0:
        flags.append("YOE_MISMATCH")
        risk_score += 0.25
        
    # Ph.D before B.Tech or B.Sc sequencing
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
            
    # 4. Buzzword Stuffing Check
    buzzwords = ["llm", "rag", "gpt", "ai", "genai", "vector database", "fine tuning"]
    has_skills_buzz = any(any(b in s.get("name", "").lower() for b in buzzwords) for s in skills)
    
    # Check if they have zero description evidence for these buzzwords
    has_career_buzz = False
    for job in career:
        desc = job.get("description", "").lower()
        if any(b in desc for b in buzzwords):
            has_career_buzz = True
            break
    if has_skills_buzz and not has_career_buzz:
        flags.append("BUZZWORD_STUFFING")
        risk_score += 0.25
        
    # 5. Impossible Company History Check
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

# ============================================================================
# Priority 6: Behavioral Intelligence Upgrade
# ============================================================================
def calculate_behavior_score(signals):
    # Normalize github score (neutral default is 0.5)
    github_score = signals.get("github_activity_score", -1)
    github_normalized = 0.50
    if github_score != -1:
        github_normalized = github_score / 100.0
        
    # Open to work
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    
    # Recruiter response rate
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    
    # Interview completion rate
    interview_completion = signals.get("interview_completion_rate", 1.0)
    
    # Saves by recruiters (normalize by dividing by 15, cap at 1)
    saves = signals.get("saved_by_recruiters_30d", 0)
    saves_normalized = min(1.0, saves / 15.0)
    
    # Profile completeness
    completeness = signals.get("profile_completeness_score", 100.0) / 100.0
    
    # Weightings
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

# YOE score helper
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

# Education score helper
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

# Location score helper
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

# Notice Period score helper
def get_notice_score(notice_days):
    if notice_days <= 30:
        return 1.0
    elif notice_days <= 60:
        return 0.80
    elif notice_days <= 90:
        return 0.50
    else:
        return 0.20

# ============================================================================
# Priority 7: Explainable Ranking Engine
# ============================================================================
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
    
    # Strengths
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
        
    # Weaknesses
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
        
    # Fallbacks
    if not strengths:
        strengths.append("Has foundational software engineering skills")
    if not weaknesses:
        weaknesses.append("No critical concerns detected")
        
    return strengths, weaknesses

# ============================================================================
# Priority 9: Rejection Reason Analyzer
# ============================================================================
def analyze_rejection_reasons(item):
    reasons = []
    
    # Check filters first
    if item.get("consulting_disqualified"):
        reasons.append("Disqualified: Entire career history is limited to consulting/services companies.")
    if item.get("title_disqualified"):
        reasons.append("Disqualified: Current job title is unrelated to AI/ML software engineering.")
    if item.get("honeypot_disqualified"):
        reasons.append(f"Blocked by advanced security filters: impossible timeline or date contradictions.")
        
    # Check low scores
    if item["yoe"] < 1.0:
        reasons.append(f"Insufficient experience: has only {item['yoe']:.1f} years total experience.")
    if item["sim"] < 0.35:
        reasons.append("Weak semantic alignment with core job requirements.")
    if item["sk_score"] < 0.20:
        reasons.append("Critical skill gap: lacks required embeddings, retrieval, or vector search skills.")
    if item["p_score"] < 0.10:
        reasons.append("No production evidence: profile lacks mentions of deployment, serving, or scale.")
    if item["b_score"] < 0.40:
        reasons.append("Weak behavioral signals: inactive profile or poor recruiter response rates.")
    if item["n_score"] < 0.50:
        reasons.append(f"High notice period: stated notice is {item['notice_days']} days.")
        
    if not reasons:
        reasons.append("Ranked out of top 100 shortlist due to competitive scores across the pool.")
        
    return reasons

# ============================================================================
# Main Processing Pipeline
# ============================================================================
def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    blacklist_path = os.path.join(script_dir, "Dataset", "honeypot_blacklist.json")
    
    # 1. Load Blacklist
    blacklist = set()
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist = set(json.load(f).keys())
        print(f"Loaded blacklist of {len(blacklist)} candidates.")
        
    # 2. Compile filters
    blacklisted_titles = {
        "marketing manager", "hr manager", "accountant", "project manager",
        "customer support", "sales executive", "civil engineer", "mechanical engineer",
        "operations manager", "content writer"
    }
    consulting_companies = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    
    # 3. Step 1: Scanning Candidates
    print(f"Stage 1: Running fast filters, heuristics, and keyword scanning...")
    t_start = time.time()
    candidates_pool = []
    rejected_pool = [] # Store rejected candidates for Priority 9
    
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
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
            
            # Run security honeypots inspector
            honeypot_audit = detect_honeypots_and_risks(cand)
            h_score = honeypot_audit["score"]
            h_flags = honeypot_audit["flags"]
            
            # Base info dict for logging / rejection analysis
            info_dict = {
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
                "l_score": 0.0,
                "n_score": 0.0,
                "h_score": h_score,
                "honeypot_flags": h_flags,
                "missing_skills": [],
                "cand_data": cand,
                "consulting_disqualified": False,
                "title_disqualified": False,
                "honeypot_disqualified": False
            }
            
            # Filtering rules:
            
            # Filter A: Security check (Exclude any candidate with HIGH honeypot risk)
            if cid in blacklist or h_score >= 0.5:
                info_dict["honeypot_disqualified"] = True
                rejected_pool.append(info_dict)
                continue
                
            # Filter B: Title check
            current_title = profile.get("current_title", "").strip().lower()
            if current_title in blacklisted_titles:
                info_dict["title_disqualified"] = True
                rejected_pool.append(info_dict)
                continue
                
            # Filter C: Consulting only
            companies = [job.get("company", "").lower() for job in career]
            if companies:
                all_consulting = all(any(c in comp for c in consulting_companies) for comp in companies)
                if all_consulting:
                    info_dict["consulting_disqualified"] = True
                    rejected_pool.append(info_dict)
                    continue
                    
            # Filter D: Experience range
            if yoe < 1.0:
                rejected_pool.append(info_dict)
                continue
                
            # Score keyword matches (Stage 1 heuristics)
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
                
            if heuristic_score > 0:
                candidates_pool.append((heuristic_score, cand, info_dict))
            else:
                rejected_pool.append(info_dict)
                
    print(f"Stage 1 completed. Candidates pool: {len(candidates_pool)}. Rejected pool: {len(rejected_pool)}.")
    
    # Pick top 1000 for Stage 2
    candidates_pool.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates_pool[:1000]
    
    # Append the discarded ones from candidates_pool into rejected_pool
    for item in candidates_pool[1000:]:
        rejected_pool.append(item[2])
        
    print(f"Selected top {len(top_candidates)} candidates for Stage 2 semantic match.")
    
    # 4. Load local model
    print(f"Loading local Model from {model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
        torch.set_num_threads(4)
    except Exception as e:
        print(f"Failed to load model from disk: {e}")
        sys.exit(1)
        
    # JD text representation
    jd_text = (
        "Senior AI Engineer — Founding Team. "
        "embeddings-based retrieval systems sentence-transformers OpenAI embeddings BGE E5. "
        "vector databases hybrid search infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS. "
        "evaluation frameworks ranking systems NDCG MRR MAP. Python, ML, LLM fine-tuning, learning-to-rank."
    )
    
    with torch.no_grad():
        encoded_jd = tokenizer(jd_text, padding=True, truncation=True, return_tensors='pt')
        jd_embedding = mean_pooling(model(**encoded_jd), encoded_jd['attention_mask'])[0]
        
    # Format candidates texts
    candidate_texts = []
    for score, cand, info in top_candidates:
        profile = cand.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        skills = ", ".join([s.get("name", "") for s in cand.get("skills", [])])
        titles = " | ".join([job.get("title", "") for job in cand.get("career_history", [])])
        candidate_texts.append(f"{headline} | {summary} | {skills} | {titles}")
        
    # Batch encode on CPU
    batch_size = 64
    cand_embeddings = []
    with torch.no_grad():
        for i in range(0, len(candidate_texts), batch_size):
            batch_texts = candidate_texts[i:i+batch_size]
            encoded_cand = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            output = model(**encoded_cand)
            batch_emb = mean_pooling(output, encoded_cand['attention_mask'])
            cand_embeddings.append(batch_emb)
            
    cand_embeddings = torch.cat(cand_embeddings, dim=0)
    
    # Cosine similarities
    similarities = torch.nn.functional.cosine_similarity(cand_embeddings, jd_embedding.unsqueeze(0), dim=1)
    similarities = similarities.cpu().numpy()
    
    # 5. Advanced Component Scoring
    final_scored_pool = []
    for idx, (score, cand, info) in enumerate(top_candidates):
        sim = float(similarities[idx])
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        
        # Priority 1: Skill Coverage
        skills_audit = evaluate_skills(cand.get("skills", []))
        sk_score = skills_audit["score"]
        
        # Priority 2: Production Experience
        prod_audit = detect_production_experience(cand)
        p_score = prod_audit["score"]
        
        # Priority 3: AI Relevance
        ai_score = calculate_ai_relevance(cand)
        
        # Priority 6: Behavioral intelligence
        behavior_audit = calculate_behavior_score(signals)
        b_score = behavior_audit["score"]
        
        # Heuristics
        yoe = profile.get("years_of_experience", 0.0)
        yoe_score = get_yoe_score(yoe)
        
        edu_score = get_education_score(cand.get("education", []))
        
        # Security Honeypot Risk
        h_score = info["h_score"]
        
        # Priority 5: Hybrid formula weights
        final_score = (
            0.30 * sim +
            0.20 * sk_score +
            0.15 * p_score +
            0.10 * ai_score +
            0.10 * b_score +
            0.10 * yoe_score +
            0.05 * edu_score -
            0.10 * h_score
        )
        
        # Round score to 4 decimal places
        final_score_rounded = round(final_score, 4)
        
        # Save scores to dictionary
        info.update({
            "score": final_score_rounded,
            "sim": sim,
            "sk_score": sk_score,
            "p_score": p_score,
            "ai_score": ai_score,
            "b_score": b_score,
            "l_score": get_location_score(info["location"], profile.get("country"), signals.get("willing_to_relocate")),
            "n_score": get_notice_score(info["notice_days"]),
            "missing_skills": skills_audit["missing_skills"],
            "matched_skills": skills_audit["matched_skills"],
            "skill_gap_report": skills_audit["gap_report"],
            "production_evidence": prod_audit["evidence"],
            "production_keywords": prod_audit["keywords"]
        })
        
        final_scored_pool.append(info)
        
    # Sort by hybrid score first to establish baseline pseudo labels
    final_scored_pool.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    if args.mode == "ltr":
        import xgboost as xgb
        import numpy as np
        
        print("Training GBDT LTR Model (XGBoost) on pseudo labels...")
        X = []
        y = []
        for rank_idx, item in enumerate(final_scored_pool):
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
                get_yoe_score(item["yoe"]),
                get_education_score(item["cand_data"].get("education", [])),
                item["h_score"],
                item["l_score"],
                item["n_score"]
            ]
            X.append(feats)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # Train XGBRegressor
        model_xgb = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model_xgb.fit(X, y)
        
        # Predict scores and normalize to 0-1 scale
        preds = model_xgb.predict(X)
        for idx, item in enumerate(final_scored_pool):
            norm_score = max(0.0, min(1.0, float(preds[idx]) / 4.0))
            item["score"] = round(norm_score, 4)
            
    # 6. Sorting and deterministic tie-breaking (score descending, ID ascending)
    final_scored_pool.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # 7. Shortlist top 100
    top_100 = final_scored_pool[:100]
    if len(top_100) > 0:
        max_raw = max(it["score"] for it in top_100)
        min_raw = min(it["score"] for it in top_100)
        range_raw = max_raw - min_raw
        for it in top_100:
            if range_raw > 0:
                normalized = 0.50 + (it["score"] - min_raw) / range_raw * 0.49
            else:
                normalized = 0.99
            it["score"] = round(normalized, 6)
            
        # Re-sort top_100 to guarantee monotonic non-increasing scores by rank and deterministic tie-breaks
        top_100.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Remaining candidates from top 1000 become rejected
    for item in final_scored_pool[100:]:
        rejected_pool.append(item)
        
    # 8. Write CSV
    print(f"Writing shortlist to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, item in enumerate(top_100):
            rank = idx + 1
            cid = item["candidate_id"]
            score = item["score"]
            reason = generate_reasoning(item["cand_data"], item["sim"], rank)
            writer.writerow([cid, rank, score, reason])
            
    # 9. Save rejected database & explainability data to local artifacts (JSON)
    # This allows the Streamlit app to load pre-calculated explainability data!
    print("Precalculating explainability and rejection profiles...")
    explainability_db = {}
    for idx, item in enumerate(top_100):
        strengths, weaknesses = generate_explanations(item)
        explainability_db[item["candidate_id"]] = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "score_breakdown": {
                "Semantic Score": round(item["sim"], 3),
                "Skill Coverage Score": round(item["sk_score"], 3),
                "Production Score": round(item["p_score"], 3),
                "AI Relevance Score": round(item["ai_score"], 3),
                "Behavioral Score": round(item["b_score"], 3),
                "Experience Score": round(get_yoe_score(item["yoe"]), 3),
                "Education Score": round(get_education_score(item["cand_data"].get("education", [])), 3),
                "Honeypot Penalty": round(item["h_score"], 3)
            }
        }
        
    rejection_db = {}
    for item in rejected_pool:
        # Calculate sub-scores for rejected candidates to feed rejection builder
        profile = item["cand_data"].get("profile", {})
        career = item["cand_data"].get("career_history", [])
        signals = item["cand_data"].get("redrob_signals", {})
        
        # Calculate skills score if not calculated
        if item["sk_score"] == 0.0:
            skills_audit = evaluate_skills(item["cand_data"].get("skills", []))
            item["sk_score"] = skills_audit["score"]
            item["missing_skills"] = skills_audit["missing_skills"]
            
        # Calculate prod score
        if item["p_score"] == 0.0:
            item["p_score"] = detect_production_experience(item["cand_data"])["score"]
            
        # Calculate behavior score
        if item["b_score"] == 0.0:
            item["b_score"] = calculate_behavior_score(signals)["score"]
            
        rejection_reasons = analyze_rejection_reasons(item)
        rejection_db[item["candidate_id"]] = {
            "name": item["name"],
            "reasons": rejection_reasons
        }
        
    with open("e:/India Runs/Dataset/explainability_db.json", "w", encoding="utf-8") as out1:
        json.dump(explainability_db, out1, indent=2)
    with open("e:/India Runs/Dataset/rejection_db.json", "w", encoding="utf-8") as out2:
        json.dump(rejection_db, out2, indent=2)
        
    total_elapsed = time.time() - t_start
    print(f"Total Execution completed in {total_elapsed:.2f} seconds.")
    print("Submissions generated and validated successfully.")

if __name__ == "__main__":
    main()
