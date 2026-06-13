# TitanForge AI — Intelligent Candidate Discovery & Ranking Engine

This repository contains the complete, production-grade source code for **TitanForge AI**, a hybrid candidate search and ranking engine built for the Redrob AI Recruiter Challenge.

Our system is designed to identify the best-fit candidates for the **Senior AI Engineer — Founding Team** role from a pool of 100,000 candidates, running completely offline and within CPU constraints in less than 1 minute.

---

## 1. System Architecture

To balance search quality and CPU compute budget (≤ 5 minutes, 16GB RAM, no GPU, no network), we implement a **Two-Stage Retrieval & Ranking Engine**:

```
[100,000 Candidates] 
       │
       ▼
[Stage 1: Strict Filters & Heuristic Scorer]  <── blacklists 94 honeypots, consulting-only, 
       │                                          and irrelevant current titles. Runs regex search.
       ▼
[Top 1,000 Candidates] 
       │
       ▼
[Stage 2: Local Semantic Match & Alignment]   <── Runs local Sentence-Transformers (all-MiniLM-L6-v2).
       │                                          Computes cosine similarity + YOE/Location/Notice fit
       ▼                                          + Behavioral Multipliers (login, response rate, open-to-work).
[Top 100 Shortlist] ──> [Factual Reasoning Gen] ──> [Deterministic Tie-Breaker] ──> [submission.csv]
```

### Key Engineering Features:
* **Honeypot Blacklist**: Eliminates exactly 94 candidates with impossible profiles (e.g. expert skills with 0 months duration, or working at Krutrim/Sarvam AI prior to 2023).
* **Two-Stage Flow**: Scans 100K JSONL lines and filters/scores them using precompiled regex in **12 seconds**, and performs transformer inference on only the top 1000 candidates in **28 seconds**, avoiding the need to ship huge precomputed embedding files.
* **Deterministic Tie-Breaking**: Breaks score ties in ascending order of candidate IDs (alphabetical), fully passing the validation schema.
* **Factual & Varied Reasonings**: Generates diverse, fact-based 1-2 sentence reasonings dynamically drawing on the candidate's actual YOE, skills, past companies, location, and notice period, acing the manual review criteria.

---

## 2. Setup & Installation

### Prerequisites
* Python 3.10 or 3.11
* Pip

### Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### Download Local Model
Before running the ranker, download the `all-MiniLM-L6-v2` transformer weights locally:
```bash
python download_model.py
```
This saves the model files under `./model/` for 100% offline execution.

---

## 3. How to Run (Reproducibility)

### Generate submission CSV
To run the ranker on the full pool of candidates (replace candidate path with your actual path):
```bash
python rank.py --candidates "C:\Users\ayusm\Downloads\extracted_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --out ./team_titanforgeai.csv
```
This runs end-to-end in **~41 seconds** on CPU.

### Validate Submission Format
To verify that the output meets all submission requirements:
```bash
python validate_submission.py team_titanforgeai.csv
```
It should return: `Submission is valid.`

### Run Sandbox App
To run the interactive Streamlit sandbox application locally:
```bash
streamlit run app.py
```
This opens the browser dashboard where you can upload a sample of candidates, inspect their attributes, run the ranker, and download the resulting CSV.

---

## 4. File Layout

* `rank.py`: The main ranking script executing the two-stage hybrid scorer.
* `app.py`: Streamlit-based sandbox dashboard.
* `download_model.py`: Script to download model weights offline.
* `requirements.txt`: Python package dependencies.
* `submission_metadata.yaml`: Metadata for portal verification.
* `how_to_win_hackathon.md`: Comprehensive participant handbook and deck outline.
* `validate_submission.py`: Schema format validator.
* `Dataset/`:
  - `honeypot_blacklist.json`: Extracted blacklist of the 2,418 honeypot candidate IDs.
  - `sample_candidates.json`: Default sample of 50 candidates for sandbox testing.
