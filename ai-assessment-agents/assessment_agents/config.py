import os

from dotenv import load_dotenv


load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma2:latest"
)

PASS_THRESHOLD = float(
    os.getenv(
        "PASS_THRESHOLD",
        "80"
    )
)

SEND_EMAIL = (
    os.getenv(
        "SEND_EMAIL",
        "false"
    ).lower() == "true"
)

SMTP_HOST = os.getenv(
    "SMTP_HOST",
    ""
)

SMTP_PORT = int(
    os.getenv(
        "SMTP_PORT",
        "587"
    )
)

SMTP_USER = os.getenv(
    "SMTP_USER",
    ""
)

SMTP_PASSWORD = os.getenv(
    "SMTP_PASSWORD",
    ""
)

SMTP_FROM = os.getenv(
    "SMTP_FROM",
    SMTP_USER
)