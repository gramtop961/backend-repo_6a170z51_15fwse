import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI(title="GenAI Media Service", description="Generate images and videos from text prompts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text description for the image")
    width: int = Field(768, ge=256, le=1536)
    height: int = Field(768, ge=256, le=1536)
    model: Optional[str] = Field("replicate/flux-schnell", description="Model identifier")


class ImageResponse(BaseModel):
    prompt: str
    urls: List[str]
    provider: str
    note: Optional[str] = None


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text description for the video")
    seconds: int = Field(4, ge=2, le=12)
    model: Optional[str] = Field("replicate/text-to-video", description="Model identifier")


class VideoResponse(BaseModel):
    prompt: str
    url: str
    provider: str
    note: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "GenAI Media Backend running", "image": "/api/generate/image", "video": "/api/generate/video"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
        "database_url": None,
        "database_name": None,
        "connection_status": "N/A",
        "collections": []
    }
    # We don't require DB for this app by default
    return response


@app.get("/api/models")
def list_models():
    return {
        "image": {
            "replicate/flux-schnell": bool(REPLICATE_API_TOKEN),
            "stability/sd": bool(STABILITY_API_KEY),
            "fallback/placeholder": True,
        },
        "video": {
            "replicate/text-to-video": bool(REPLICATE_API_TOKEN),
            "fallback/sample": True,
        }
    }


def replicate_create_prediction(model: str, input_payload: dict) -> Optional[List[str]]:
    """Create a Replicate prediction and return output URLs if successful."""
    if not REPLICATE_API_TOKEN:
        return None
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "version": model,
        "input": input_payload,
    }
    r = requests.post("https://api.replicate.com/v1/predictions", json=payload, headers=headers, timeout=30)
    if r.status_code not in (200, 201):
        return None
    pred = r.json()
    poll_url = pred.get("urls", {}).get("get")
    # Poll until completed or failed
    for _ in range(60):
        pr = requests.get(poll_url, headers=headers, timeout=30)
        pj = pr.json()
        status = pj.get("status")
        if status in ("succeeded", "failed", "canceled"):
            if status == "succeeded":
                out = pj.get("output")
                if isinstance(out, list):
                    return out
                elif isinstance(out, str):
                    return [out]
            return None
        time.sleep(2)
    return None


def stability_generate_image(prompt: str, width: int, height: int) -> Optional[List[str]]:
    # Placeholder: prefer Replicate integration for simplicity; Stability REST varies per plan
    return None


def placeholder_image_urls(prompt: str, width: int, height: int, n: int = 2) -> List[str]:
    import urllib.parse as up
    seed = up.quote_plus(prompt.strip())[:60] or "seed"
    return [f"https://picsum.photos/seed/{seed}-{i}/{width}/{height}" for i in range(n)]


@app.post("/api/generate/image", response_model=ImageResponse)
def generate_image(req: ImageRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Try Replicate FLUX (schnell)
    if REPLICATE_API_TOKEN and (req.model.startswith("replicate") or req.model == "auto"):
        # Replicate expects model version hash; we'll use a stable alias via the collections API
        # For reliability, call the model name endpoint with 'black-forest-labs/flux-schnell'
        model_name = "black-forest-labs/flux-schnell"
        input_payload = {"prompt": prompt, "width": req.width, "height": req.height, "go_fast": True}
        urls = replicate_create_prediction(model=model_name, input_payload=input_payload)
        if urls:
            return ImageResponse(prompt=prompt, urls=urls, provider="replicate")

    # Try Stability if configured (not implemented here)
    if STABILITY_API_KEY and req.model.startswith("stability"):
        urls = stability_generate_image(prompt, req.width, req.height)
        if urls:
            return ImageResponse(prompt=prompt, urls=urls, provider="stability")

    # Fallback placeholder images
    urls = placeholder_image_urls(prompt, req.width, req.height, n=3)
    return ImageResponse(prompt=prompt, urls=urls, provider="placeholder", note="Using placeholder images. Add REPLICATE_API_TOKEN to enable real generation.")


@app.post("/api/generate/video", response_model=VideoResponse)
def generate_video(req: VideoRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # If Replicate token exists, try a lightweight text-to-video
    if REPLICATE_API_TOKEN and (req.model.startswith("replicate") or req.model == "auto"):
        # Use a known community model; versions change often, so we rely on name endpoint
        model_name = "fofr/kolors-video"  # fallback name; may vary
        input_payload = {"prompt": prompt, "num_frames": req.seconds * 6}
        urls = replicate_create_prediction(model=model_name, input_payload=input_payload)
        if urls and len(urls) > 0:
            return VideoResponse(prompt=prompt, url=urls[-1], provider="replicate")

    # Fallback: sample royalty-free video
    sample_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
    return VideoResponse(prompt=prompt, url=sample_url, provider="sample", note="Using sample video. Add REPLICATE_API_TOKEN to enable real generation.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
