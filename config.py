import os
from dotenv import load_dotenv

load_dotenv()

# ---- ClinicalTrials.gov API ----
CT_API_BASE_URL = "https://clinicaltrials.gov/api/v2"
CT_API_STUDIES_ENDPOINT = f"{CT_API_BASE_URL}/studies"
CT_API_RATE_LIMIT = 50          # requests per minute
CT_API_PAGE_SIZE = 10           # default results per page
CT_API_MAX_PAGE_SIZE = 100

VALID_STATUSES = [
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "COMPLETED",
    "ENROLLING_BY_INVITATION",
    "SUSPENDED",
    "TERMINATED",
    "WITHDRAWN",
    "AVAILABLE",
]

# ---- Google Gemini Configuration ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))

# ---- Geographic Defaults ----
DEFAULT_SEARCH_RADIUS_MILES = 50
DEFAULT_COUNTRY = "United States"

# ---- Safety ----
MEDICAL_DISCLAIMER = (
    "This tool provides clinical trial information only. "
    "It does NOT provide medical advice, diagnosis, or treatment recommendations. "
    "Always consult with a qualified healthcare provider before making any medical decisions "
    "or enrolling in a clinical trial."
)

# ---- Caching ----
CACHE_TTL_SECONDS = 3600
CACHE_MAX_SIZE = 500

# ---- Logging ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")