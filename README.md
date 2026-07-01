# 🏆 Redrob Intelligent Candidate Ranker

An optimized, rule-based candidate ranking system developed for the **Redrob Hackathon - Intelligent Candidate Discovery & Ranking Challenge**.

This system processes a pool of **100,000 candidates** and generates a curated shortlist of the **top 100 candidates** optimized for a **Senior AI Engineer** role, complying with all resource, runtime (≤ 5 minutes), and format validation constraints.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Hard Filtering** | Removes candidates with <2 years experience, no technical AI/ML signals, or clearly unrelated professions |
| **Tiered Skill Matching** | Evaluates candidates against JD-critical skills (Vector DB, Retrieval, Ranking, LLMs) across three priority tiers |
| **Career Scoring** | Assesses title relevance, company quality, experience, and seniority |
| **Behavioral Scoring** | Evaluates availability, responsiveness, activity, and platform verification |
| **Reasoning Generation** | Provides factual, 1-2 sentence explanations for each candidate |
| **Streamlit Dashboard** | Interactive UI for visualizing rankings, filtering, and inspecting profiles |

---

## 🎯 Scoring Components

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Skill Score** | 55% | Tier A (Vector DB, Retrieval, Ranking) + Tier B (Embeddings, LoRA, PEFT) + Tier C (ML, DL, NLP) |
| **Career Score** | 30% | Title relevance, company quality, experience, seniority |
| **Behavior Score** | 15% | Response rate, notice period, open to work, activity |

```
Final Score = (Skill × 0.55) + (Career × 0.30) + (Behavior × 0.15)
```

---

## 📁 Project Structure

```
ai_candidate_ranker/
│
├── rank.py                          # Main CLI execution entry point
├── validate_submission.py           # CSV format validator
├── README.md                        # Project documentation
├── requirements.txt                 # dependencies
├── submission_metadata.yaml         # Portal metadata for Stage 3 validation
├── final_candidates.csv             # Generated final top-100 ranked output CSV
│
├── data/                            # Challenge resources and datasets
│   ├── candidates.jsonl             # 100K candidate profiles
│   ├── candidate_schema.json        # JSON schema
│   ├── sample_candidates.json       # Sample 50 profiles
│   └── sample_submission.csv        # Sample submission format
│
├── src/                             # Pipeline source code modules
│   ├── __init__.py                  # Python package initialization
│   ├── loader.py                    # Loads, flattens, and processes candidates
│   ├── hard_filter.py               # Eliminates ineligible candidates quickly
│   ├── skill_matcher.py             # Computes tiered skill relevance metrics
│   ├── career_matcher.py            # Scores job title, progression, and experience
│   ├── behavioral_scorer.py         # Calculates behavioral platform signals
│   ├── ranking_engine.py            # Merges sub-scores into weighted composite
│   ├── reasoning_generator.py       # Formulates factual, rank-consistent justifications
│   ├── llm_reranker.py              # Deterministic reranking adjustments
│   └── submission_builder.py        # Builds final submission CSV
│
├── outputs/                         # Generated output directory
│   └── final_candidates.csv
│
└── sandbox/                         # Interactive demo space
    └── app.py                       # Streamlit web application

```

---

## 📋 Step-by-Step Run Guide

### Step 1: Environment Setup

Ensure you are using **Python 3.11+**. The core ranker uses Python's standard library with minimal external dependencies.

```bash
# Clone the repository
git clone https://github.com/Yash-Raj-96/redrob_hackathon_26.git
cd redrob_hackathon_26

# Install dependencies
pip install -r requirements.txt

```

### Step 2: Running the Ranker on the Full Dataset

Run the ranking pipeline on the full **100,000-candidate** dataset:

```bash
python rank.py --candidates data/candidates.jsonl --out outputs/final_candidates.csv
```

**Expected Output:**
```
Loading candidates...
Loaded 100,000 candidates
13,780 candidates passed hard filtering
Submission written to output/final_candidates.csv
Top candidate: CAND_0018499 (score=0.979600)
```

### Step 3: Validating the Output

Run the validator script on the generated CSV file to confirm formatting and scoring validity:

```bash
python validate_submission.py final_candidates.csv        
```

If successful, the console will print:

```
Submission is valid.
```

### Step 4: Check the Output

```bash
# View first 5 lines
head -n 5 outputs/final_candidates.csv

# Or on Windows PowerShell
Get-Content outputs/final_candidates.csv -TotalCount 5
```

---

## 🖥️ Launching the Streamlit Sandbox

For testing smaller candidate samples interactively, launch the local Streamlit dashboard:

```bash
# Install sandbox dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run sandbox/app.py
```

Open the local link shown in your terminal (typically `http://localhost:8501`).

**Features:**
- Upload `sample_candidates.json` from the challenge bundle
- View ranking distribution and component scores
- Apply filters by title, experience, location, and score
- Inspect individual candidate profiles with detailed reasoning

---

## 📊 Output Format

The system generates a CSV file with the following columns:

```csv
candidate_id,rank,score,reasoning
CAND_0018499,1,0.979600,"Senior Machine Learning Engineer with 7.2 yrs experience at Zomato; relevant skills: Pinecone, Weaviate, Milvus; Noida, Uttar Pradesh. Platform signals are strong: immediate availability; GitHub score 95."
```

| Column | Description |
|--------|-------------|
| `candidate_id` | Unique candidate identifier |
| `rank` | Rank position (1-100) |
| `score` | Composite score (0-1) |
| `reasoning` | Factual 1-2 sentence explanation |

---

## ✅ Sample Output Preview

```
Top 10 Candidates:
  1. CAND_0018499 | Senior Machine Learning Engineer | 0.9900
  2. CAND_0071974 | Senior AI Engineer               | 0.9880
  3. CAND_0077337 | Staff Machine Learning Engineer  | 0.9870
  4. CAND_0002025 | Senior AI Engineer               | 0.9850
  5. CAND_0046525 | Senior Machine Learning Engineer | 0.9810
  6. CAND_0086022 | Senior Applied Scientist         | 0.9680
  7. CAND_0006557 | NLP Engineer                     | 0.9650
  8. CAND_0070398 | Machine Learning Engineer        | 0.9650
  9. CAND_0009691 | Applied ML Engineer              | 0.9600
 10. CAND_0008425 | Senior NLP Engineer              | 0.9590
```

---

## 🐳 Docker Deployment (Hugging Face Spaces)

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "sandbox/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
```

---

## 🚀 Quick Run Commands

| Command | Description |
|---------|-------------|
| `python rank.py --candidates data/candidates.jsonl --out outputs/final_candidates.csv` | Run ranking pipeline |
| `python validate_submission.py final_candidates.csv` | Validate output |
| `streamlit run sandbox/app.py` | Launch dashboard |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core development language |
| **Pandas** | Data processing and CSV handling |
| **NumPy** | Numerical operations |
| **Streamlit** | Interactive sandbox dashboard |
| **Plotly** | Visualizations in dashboard |
| **Docker** | Reproducible deployment |
| **Hugging Face Spaces** | Sandbox hosting |

---

## 📝 Submission Assets

| Asset | Location |
|-------|----------|
| **GitHub Repository** | [https://github.com/Yash-Raj-96/redrob_hackathon_26](https://github.com/Yash-Raj-96/redrob_hackathon_26) |
| **Hugging Face Space** | [https://huggingface.co/spaces/HiddenBeauty/redrob_hackathon](https://huggingface.co/spaces/HiddenBeauty/redrob_hackathon) |
| **Submission CSV** | [`final_candidates.csv`](https://github.com/Yash-Raj-96/redrob_hackathon_26/blob/main/output/final_candidates.csv
)|
| **Metadata** | [`submission_metadata.yaml`](https://github.com/Yash-Raj-96/redrob_hackathon_26/blob/main/submission_metadata.yaml) |

---
