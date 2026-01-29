"""Core Constants"""

IMAGE_ANALYSIS_FAILURE_MARKER = "Unable to analyze"

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

INITIAL_RECALL_K = 25
FINAL_PRECISION_K = 5

# Minimum relevance score threshold for context to be considered relevant
# FlashRank reranker scores typically range from 0 to 1
# Set very low to allow model to decide relevance based on content
MIN_RELEVANCE_SCORE = 0.001

RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5
