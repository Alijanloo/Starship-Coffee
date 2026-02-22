"""Application-wide paths, constants, and injection-detection patterns."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

for _proxy_key in (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
):
    os.environ.pop(_proxy_key, None)

DOCS_DIR = Path(__file__).parent / "docs"
SECRETS_DIR = Path(__file__).parent / "secrets"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "starship_docs"

EMBED_MODEL: str = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "starship-coffee",
    "X-Title": "Starship Coffee RAG",
}

INJECTION_PATTERNS: list[str] = [
    r"secret",
    r"api[_\s]?key",
    r"password",
    r"reveal",
    r"ignore (previous|above|prior)",
    r"forget (previous|above|prior)",
    r"you are now",
    r"act as",
    r"show (me )?(the )?(content|file|full|raw)",
    r"print (the )?(content|file|raw)",
    r"read (the )?(file|content)",
    r"admin",
]
