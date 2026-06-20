import os, requests, base64, re
from typing import List, Dict, Any, Optional, Iterable

BASE = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
TOKEN = os.getenv("OLLAMA_BEARER")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:11b")

def _headers():
    h = {"Accept":"application/json","Content-Type":"application/json"}
    if TOKEN: h["Authorization"] = f"Bearer {TOKEN}"
    return h

def list_models() -> Dict[str,Any]:
    r = requests.get(f"{BASE}/api/tags", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def _to_ollama_messages(openai_messages: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Convert OpenAI-style messages to Ollama chat format.
    Supports text and image_url parts on user messages by converting to 'images':[b64,...].
    """
    out = []
    for m in openai_messages:
        role = m.get("role","user")
        content = m.get("content")
        if isinstance(content, list):
            # assemble text + images
            text_chunks = []
            images_b64 = []
            for part in content:
                if part.get("type") == "text":
                    text_chunks.append(part.get("text",""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url","")
                    # expect data:image/jpeg;base64,<b64>
                    if url.startswith("data:"):
                        b64 = url.split("base64,",1)[-1]
                        images_b64.append(b64)
            out.append({"role": role, "content": "\n".join(text_chunks).strip(), **({"images":images_b64} if images_b64 else {})})
        else:
            out.append({"role": role, "content": str(content or "")})
    return out

def chat(messages: List[Dict[str,Any]], model: Optional[str]=None, stream: bool=False) -> Dict[str,Any] | Iterable[str]:
    payload = {"model": model or CHAT_MODEL, "messages": _to_ollama_messages(messages), "stream": stream}
    r = requests.post(f"{BASE}/api/chat", headers=_headers(), json=payload, timeout=None)
    r.raise_for_status()
    if stream:
        return (line.decode("utf-8") for line in r.iter_lines() if line)
    return r.json()

def chat_text(messages: List[Dict[str,Any]], stream: bool=False) -> Dict[str,Any] | Iterable[str]:
    return chat(messages, model=CHAT_MODEL, stream=stream)

def chat_vision(messages: List[Dict[str,Any]], stream: bool=False) -> Dict[str,Any] | Iterable[str]:
    return chat(messages, model=VISION_MODEL, stream=stream)

def extract_text(resp):
    msg = (resp or {}).get("message", {})
    text = (msg.get("content") or "").strip()
    # strip reasoning dumps
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    # strip code fences if any
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    return text