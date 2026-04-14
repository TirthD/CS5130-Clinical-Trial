# 🔬 Clinical Trial Finder

An AI-powered assistant that helps people find relevant clinical trials on [ClinicalTrials.gov](https://clinicaltrials.gov) and explains eligibility criteria in plain English.

## The Problem

There are 460,000+ clinical trials on ClinicalTrials.gov, but only ~5% of eligible patients ever enroll. The barriers: medical jargon, complex eligibility criteria, and no easy way to search by location.

## What It Does

Ask a question in plain English → get matching trials with eligibility explained simply.

```
You:   "I'm 45 with Type 2 diabetes in Boston. What trials am I eligible for?"

Agent: Found 3 recruiting trials near Boston:
       1. RESET System Pivotal Trial (NCT04101669) — Brigham and Women's Hospital
          Ages 22-65, requires HbA1c ≥7.5%, BMI ≥30...
       2. ...
```

## How It Works

A user query flows through five specialized tools, orchestrated by an LLM agent:

| Tool | What It Does |
|------|-------------|
| Medical Term Mapper | "heart attack" → "myocardial infarction" |
| Trial Searcher | Queries ClinicalTrials.gov API (live data) |
| Geo Matcher | Filters and ranks trials by distance from user |
| Eligibility Parser | Checks if user meets age/gender/condition criteria |
| Plain Language Translator | Rewrites medical jargon into simple English |

The agent (Llama 3.3 on Groq) decides which tools to call and in what order, then assembles a clear response. Safety guardrails check every response for medical advice, hallucinated trial IDs, and missing disclaimers.

## Setup

### Prerequisites
- Python 3.12+
- [Groq API key](https://console.groq.com) (free tier)

### Install

```bash
git clone https://github.com/your-repo/CS5130-Clinical-Trial.git
cd CS5130-Clinical-Trial
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

### Run

**Web app (Streamlit):**
```bash
streamlit run app.py
```

**Terminal (single query):**
```bash
python3 scripts/run_agent.py --query "Type 2 diabetes trials in Boston"
```

**Terminal (interactive):**
```bash
python3 scripts/run_agent.py
```

## Evaluation

### GeoMatcher Evaluation
```bash
python3 -m tests.evaluation.eval_runner --verbose
```
Results: F1 1.00 | Ranking Accuracy 1.00 | Distance MAE 1.4 mi

### Agent End-to-End Evaluation
```bash
python3 tests/evaluation/agent_eval.py --verbose
```
Results: 100% relevance | 100% NCT citations | 0% medical advice

## Project Structure

```
├── app.py                    # Streamlit web app
├── config.py                 # API keys and model config
├── scripts/
│   └── run_agent.py          # Terminal entry point
├── src/
│   ├── agent/
│   │   ├── agent.py          # LLM orchestrator (Groq)
│   │   ├── prompts.py        # System prompt and templates
│   │   ├── safety.py         # Safety guardrails
│   │   └── tool_registry.py  # Tool declarations and dispatcher
│   ├── api/
│   │   ├── client.py         # ClinicalTrials.gov HTTP client
│   │   ├── endpoints.py      # URL and parameter building
│   │   ├── exceptions.py     # Custom error types
│   │   └── models.py         # Trial, Location, Eligibility models
│   └── tools/
│       ├── trial_searcher.py
│       ├── medical_term_mapper.py
│       ├── eligibility_parser.py
│       ├── geo_matcher.py
│       └── plain_language.py
├── tests/
│   └── evaluation/
│       ├── agent_eval.py     # End-to-end agent tests
│       ├── eval_runner.py    # GeoMatcher evaluation
│       ├── metrics.py        # Precision, recall, F1, MAE
│       └── test_cases.json   # GeoMatcher test fixtures
└── data/
    ├── medical_synonyms.json
    └── condition_codes.json
```

## Team

| Member | Contribution |
|--------|-------------|
| Member 1 | Agent orchestrator, trial searcher, safety guardrails, Streamlit app, evaluation suite |
| Member 2 | Medical term mapper, eligibility parser, plain language translator |
| Member 3 | Geo matcher, testing infrastructure, GeoMatcher evaluation |

## Disclaimer

This tool is for informational purposes only and is NOT medical advice. Always consult your healthcare provider before making decisions about clinical trial participation.