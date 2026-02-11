import os

import requests
from dotenv import load_dotenv

load_dotenv()


def get_openrouter_base_url():
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def get_openrouter_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    return api_key


def get_openrouter_headers():
    headers = {
        "Authorization": f"Bearer {get_openrouter_api_key()}",
        "Content-Type": "application/json",
    }

    referer = os.getenv("OPENROUTER_SITE_URL")
    app_title = os.getenv("OPENROUTER_APP_NAME", "Patent Research Assistant")
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title
    return headers


def list_openrouter_models(timeout=10):
    url = f"{get_openrouter_base_url()}/models"
    response = requests.get(url, headers=get_openrouter_headers(), timeout=timeout)
    response.raise_for_status()
    data = response.json().get("data", [])
    return [item.get("id") for item in data if item.get("id")]
