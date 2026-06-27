import os
import re
import time
import base64
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Use the Windows certificate store when the local Python CA bundle cannot
# validate HTTPS traffic (common with managed antivirus/proxy certificates).
try:
    import truststore
except ImportError:
    try:
        import pip._vendor.truststore as truststore
    except ImportError:
        truststore = None

if truststore is not None:
    truststore.inject_into_ssl()

import requests

load_dotenv()

app = FastAPI(title="Nutri Ninja API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OFF_BASE = "https://world.openfoodfacts.org"
product_cache: dict[str, dict[str, Any]] = {}
user_profiles: dict[str, dict[str, Any]] = {}
scan_history: list[dict[str, Any]] = []


class ProductPayload(BaseModel):
    product: dict[str, Any]
    profile: dict[str, Any] | None = None


class RecommendationPayload(BaseModel):
    product: dict[str, Any]
    candidates: list[dict[str, Any]] = []


class LabelPayload(BaseModel):
    text: str
    allergies: str = ""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatPayload(BaseModel):
    messages: list[ChatMessage]
    profile: dict[str, Any] | None = None


class VoiceChatPayload(ChatPayload):
    audio_base64: str
    mime_type: str = "audio/webm"


SUGAR_TERMS = ["sugar", "glucose", "fructose", "dextrose", "maltose", "sucrose", "jaggery", "corn syrup"]
SALT_TERMS = ["salt", "sodium", "baking soda", "sodium chloride"]
ADDITIVE_TERMS = [
    "high fructose corn syrup",
    "hydrogenated",
    "partially hydrogenated",
    "palm oil",
    "msg",
    "monosodium glutamate",
    "sodium benzoate",
    "tartrazine",
    "aspartame",
]
GOOD_TERMS = ["whole wheat", "oats", "millet", "ragi", "jowar", "bran", "fiber", "fibre", "protein", "lentil"]
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def health_score(product: dict[str, Any]) -> int:
    nutriments = product.get("nutriments") or {}
    score = 100.0

    score -= min(float(nutriments.get("sugars_100g") or 0) * 1.8, 25)
    score -= min(float(nutriments.get("saturated-fat_100g") or 0) * 2.2, 25)
    score -= min(float(nutriments.get("salt_100g") or 0) * 1.5, 15)
    score += min(float(nutriments.get("fiber_100g") or 0) * 1.5, 15)
    score += min(float(nutriments.get("proteins_100g") or 0) * 1.2, 15)

    additives = product.get("additives_n") or len(product.get("additives_tags") or [])
    score -= min(float(additives or 0) * 2, 12)

    energy = float(nutriments.get("energy_100g") or 0)
    if energy > 300:
        score -= min((energy - 300) / 50, 10)

    return round(max(1, min(100, score)))


def nutri_score(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "E"


def warnings(product: dict[str, Any]) -> list[dict[str, str]]:
    nutriments = product.get("nutriments") or {}
    result: list[dict[str, str]] = []

    sugar = float(nutriments.get("sugars_100g") or 0)
    salt = float(nutriments.get("salt_100g") or 0)
    saturated_fat = float(nutriments.get("saturated-fat_100g") or 0)

    if sugar > 25:
        result.append({"severity": "high", "title": "High sugar", "message": f"{sugar:g}g sugar per 100g"})
    elif sugar > 12.5:
        result.append({"severity": "medium", "title": "Moderate sugar", "message": f"{sugar:g}g sugar per 100g"})

    if salt > 1.5:
        result.append({"severity": "high", "title": "High salt", "message": f"{salt:g}g salt per 100g"})
    elif salt > 0.5:
        result.append({"severity": "medium", "title": "Moderate salt", "message": f"{salt:g}g salt per 100g"})

    if saturated_fat > 5:
        result.append({"severity": "high", "title": "High saturated fat", "message": f"{saturated_fat:g}g saturated fat per 100g"})
    elif saturated_fat > 3:
        result.append({"severity": "medium", "title": "Moderate saturated fat", "message": f"{saturated_fat:g}g saturated fat per 100g"})

    return result or [{"severity": "low", "title": "Balanced profile", "message": "No major label warning detected"}]


def product_summary(product: dict[str, Any]) -> dict[str, Any]:
    image_url = (
        product.get("image_front_url")
        or product.get("image_url")
        or product.get("image_front_small_url")
        or product.get("image_small_url")
        or product.get("image_front_thumb_url")
        or product.get("image_thumb_url")
        or ""
    )
    if isinstance(image_url, str) and image_url.startswith("http://"):
        image_url = f"https://{image_url[7:]}"

    return {
        "code": product.get("code") or product.get("_id"),
        "product_name": product.get("product_name") or "Unknown product",
        "brands": product.get("brands") or "",
        "image_front_url": image_url,
        "image_url": image_url,
        "image_front_small_url": product.get("image_front_small_url") or "",
        "image_small_url": product.get("image_small_url") or "",
        "image_front_thumb_url": product.get("image_front_thumb_url") or "",
        "image_thumb_url": product.get("image_thumb_url") or "",
        "categories": product.get("categories") or product.get("categories_en") or "",
        "nutriments": product.get("nutriments") or {},
        "health_score": health_score(product),
    }


def analyze(product: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    score = health_score(product)
    personalized: list[str] = []
    profile = profile or {}
    goal = profile.get("goal", "general")
    nutriments = product.get("nutriments") or {}

    if goal == "diabetes" and float(nutriments.get("sugars_100g") or 0) > 10:
        personalized.append("Sugar is high for diabetes mode.")
    if goal == "muscle_gain" and float(nutriments.get("proteins_100g") or 0) >= 8:
        personalized.append("Useful protein contribution for muscle gain.")
    if goal == "heart_health" and float(nutriments.get("salt_100g") or 0) > 0.75:
        personalized.append("Salt is high for heart health mode.")

    return {
        "product": product_summary(product),
        "health_score": score,
        "nutri_score": nutri_score(score),
        "warnings": warnings(product),
        "personalized_insights": personalized,
    }


def label_analysis(text: str, allergies: str = "") -> dict[str, Any]:
    normalized = text.lower()
    findings: list[dict[str, str]] = []

    def matched(terms: list[str]) -> list[str]:
        return [term for term in terms if term in normalized]

    allergy_hits = [
        item.strip().lower()
        for item in allergies.split(",")
        if item.strip() and item.strip().lower() in normalized
    ]
    if allergy_hits:
        findings.append({
            "title": "Allergy alert",
            "detail": f"Matches allergy list: {', '.join(allergy_hits)}",
            "severity": "danger",
        })

    for title, terms, severity in [
        ("Additive caution", ADDITIVE_TERMS, "danger"),
        ("Sugar sources found", SUGAR_TERMS, "warning"),
        ("Salt or sodium found", SALT_TERMS, "warning"),
        ("Positive ingredients", GOOD_TERMS, "good"),
    ]:
        hits = matched(terms)
        if hits:
            findings.append({
                "title": title,
                "detail": f"Detected {', '.join(hits[:4])}",
                "severity": severity,
            })

    if not findings:
        findings.append({
            "title": "No major flags",
            "detail": "No common sugar, sodium, additive, or allergy terms were detected.",
            "severity": "info",
        })

    score = 75
    for finding in findings:
        if finding["severity"] == "danger":
            score -= 22
        elif finding["severity"] == "warning":
            score -= 12
        elif finding["severity"] == "good":
            score += 8

    return {"score": max(1, min(100, score)), "findings": findings}


def build_system_prompt(profile: dict[str, Any]) -> str:
    return (
        "You are Nutri Ninja, a concise and friendly nutrition assistant. "
        "Give practical food, ingredient, grocery, and label guidance. "
        "Respect the user's allergies and dietary goals. Do not diagnose or replace a clinician. "
        "For urgent health symptoms, advise professional care. "
        "Keep answers under 150 words unless detail is requested. "
        f"User profile — Name: {profile.get('name', 'User')}; "
        f"goal: {profile.get('goal', 'general')}; "
        f"dietary preferences: {', '.join(profile.get('restrictions') or []) or 'none'}; "
        f"allergies: {profile.get('allergies', 'none') or 'none'}; "
        f"conditions: {profile.get('conditions', 'none') or 'none'}; "
        f"ingredients to avoid: {profile.get('dislikedIngredients', 'none') or 'none'}."
    )


def call_openai_chat(messages: list[dict[str, Any]], profile: dict[str, Any] | None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI chat is not configured. Add OPENAI_API_KEY to Render environment variables.",
        )

    profile = profile or {}
    openai_messages = [
        {"role": "system", "content": build_system_prompt(profile)},
        *[{"role": m["role"], "content": m["content"]} for m in messages if m.get("content", "").strip()],
    ]

    try:
        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": openai_messages, "max_tokens": 400},
            timeout=60,
        )

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"].strip()
            if not answer:
                raise HTTPException(status_code=502, detail="OpenAI returned an empty response.")
            return answer

        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="AI is busy right now. Please try again in a moment.")

        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=503,
                detail="OpenAI API key is invalid. Update OPENAI_API_KEY in Render environment variables.",
            )

        try:
            err = response.json().get("error", {}).get("message", "")
        except Exception:
            err = ""
        raise HTTPException(status_code=502, detail=f"OpenAI error {response.status_code}: {err or 'Unknown error'}")

    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Could not reach OpenAI. Check your internet connection.") from exc


def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    import io
    api_key = os.getenv("OPENAI_API_KEY", "")
    ext = "webm" if "webm" in mime_type else ("mp3" if "mp3" in mime_type else "wav")
    response = requests.post(
        f"{OPENAI_API_BASE}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (f"audio.{ext}", io.BytesIO(audio_bytes), mime_type)},
        data={"model": "whisper-1"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("text", "").strip()


@app.get("/")
def home():
    return {
        "message": "Nutri Ninja API running",
        "features": ["product lookup", "search", "analysis", "recommendations", "profile", "history"],
    }


@app.get("/product/{barcode}")
def get_product(barcode: str):
    if barcode in product_cache:
        return product_cache[barcode]

    response = requests.get(f"{OFF_BASE}/api/v0/product/{barcode}.json", timeout=15)
    data = response.json()
    product_cache[barcode] = data

    if data.get("product"):
        scan_history.insert(0, product_summary(data["product"]))
        del scan_history[50:]

    return data


@app.get("/search")
def search_products(query: str = Query(..., min_length=1), page_size: int = 20):
    response = requests.get(
        f"{OFF_BASE}/cgi/search.pl",
        params={"search_terms": query, "search_simple": 1, "json": 1, "page_size": page_size},
        timeout=15,
    )
    data = response.json()
    return {"products": [product_summary(item) for item in data.get("products", []) if item.get("product_name")]}


@app.post("/analyze")
def analyze_product(payload: ProductPayload):
    return analyze(payload.product, payload.profile)


@app.post("/recommendations")
def recommendations(payload: RecommendationPayload):
    current_score = health_score(payload.product)
    ranked = [
        product_summary(item)
        for item in payload.candidates
        if item.get("product_name") and item.get("nutriments")
    ]
    return {
        "better": sorted([item for item in ranked if item["health_score"] > current_score], key=lambda item: item["health_score"], reverse=True)[:10],
        "worse": sorted([item for item in ranked if item["health_score"] < current_score], key=lambda item: item["health_score"])[:10],
    }


@app.post("/label/analyze")
def analyze_label(payload: LabelPayload):
    return label_analysis(payload.text, payload.allergies)


@app.post("/chat")
def chat(payload: ChatPayload):
    messages = payload.messages[-12:]
    if not messages or not messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="A message is required.")

    msgs = [{"role": m.role, "content": m.content.strip()} for m in messages if m.content.strip()]
    answer = call_openai_chat(msgs, payload.profile)
    return {"message": answer, "model": OPENAI_MODEL}


@app.post("/chat/voice")
def voice_chat(payload: VoiceChatPayload):
    try:
        audio_bytes = base64.b64decode(payload.audio_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="The recorded audio is invalid.") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio was recorded.")
    if len(audio_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Voice recording is too long. Keep it under one minute.")

    # Transcribe audio with Whisper, then answer with GPT
    transcript = transcribe_audio(audio_bytes, payload.mime_type)
    if not transcript:
        raise HTTPException(status_code=400, detail="Could not understand the audio. Please try speaking clearly.")

    history = [{"role": m.role, "content": m.content.strip()} for m in payload.messages[-8:] if m.content.strip()]
    history.append({"role": "user", "content": transcript})
    answer = call_openai_chat(history, payload.profile)
    return {"message": answer, "model": OPENAI_MODEL, "transcript": transcript}


@app.post("/user/profile/{user_id}")
def save_profile(user_id: str, profile: dict[str, Any]):
    user_profiles[user_id] = profile
    return {"saved": True, "profile": profile}


@app.get("/user/profile/{user_id}")
def get_profile(user_id: str):
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    return user_profiles[user_id]


@app.get("/history")
def history():
    return {"items": scan_history}


@app.get("/test-ai")
def test_ai():
    """Open in browser to verify OpenAI API key is working."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"status": "error", "reason": "OPENAI_API_KEY not set in Render environment variables."}
    try:
        r = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 50},
            timeout=30,
        )
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            return {"status": "ok", "model": OPENAI_MODEL, "reply": reply}
        try:
            err = r.json().get("error", {})
        except Exception:
            err = {}
        return {
            "status": "error",
            "http_status": r.status_code,
            "message": err.get("message", r.text[:300]),
            "fix": "Update OPENAI_API_KEY in Render environment variables.",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
