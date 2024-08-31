import os

# Directories
REPORTS_FOLDER = "ev_reports/"
STORAGE_FOLDER = "./storage/"

# Ensure directories exist
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# LLM settings
LLM_MODEL = "llama3.1"
LLM_TEMPERATURE = 0.5
LLM_TIMEOUT = 1800000  # 30 minutes

# Embedding model settings
EMBED_MODEL = "nomic-embed-text"
EMBED_TIMEOUT = 1800000  # 30 minutes

# Concurrency settings
MAX_CONCURRENCY = 1

# PDF settings
PDF_FONT = "Arial"
PDF_FONT_SIZE = 12
PDF_TITLE_ALIGN = 'C'
PDF_CONTENT_MARGIN = 10

# Workflow settings
WORKFLOW_TIMEOUT = 1800000  # 30 minutes