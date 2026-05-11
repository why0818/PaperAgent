from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .models import SearchResult
from .paths import DATA_DIR


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_CHAT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_ANALYZE_MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus"
CONFIG_PATH = DATA_DIR / "config.local.json"
LAST_ERROR = ""

# Available models for selection
SUPPORTED_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    "moonshotai/Kimi-K2-Instruct-0905",
]


def local_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_llm_models(chat_model: str, analyze_model: str):
    cfg = local_config()
    if chat_model not in SUPPORTED_MODELS:
        chat_model = DEFAULT_CHAT_MODEL
    if analyze_model not in SUPPORTED_MODELS:
        analyze_model = DEFAULT_ANALYZE_MODEL
    cfg["chat_model"] = chat_model
    cfg["analyze_model"] = analyze_model
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def llm_config() -> dict:
    cfg = local_config()
    api_key = os.getenv("PAPERAGENT_LLM_API_KEY", "").strip() or str(cfg.get("api_key", "")).strip()
    
    # Dual model configuration
    chat_model = os.getenv("PAPERAGENT_CHAT_MODEL", "").strip() or str(cfg.get("chat_model", DEFAULT_CHAT_MODEL)).strip()
    analyze_model = os.getenv("PAPERAGENT_ANALYZE_MODEL", "").strip() or str(cfg.get("analyze_model", DEFAULT_ANALYZE_MODEL)).strip()
    if chat_model not in SUPPORTED_MODELS:
        chat_model = DEFAULT_CHAT_MODEL
    if analyze_model not in SUPPORTED_MODELS:
        analyze_model = DEFAULT_ANALYZE_MODEL
    
    # Legacy fallback
    legacy_model = os.getenv("PAPERAGENT_LLM_MODEL", "").strip() or str(cfg.get("model", "")).strip()
    if legacy_model and not cfg.get("chat_model"):
        chat_model = legacy_model

    base_url = os.getenv("PAPERAGENT_LLM_BASE_URL", "").strip() or str(cfg.get("base_url", DEFAULT_BASE_URL)).strip()
    enabled = bool(api_key)
    
    allow_external_paper_content = bool(cfg.get("allow_external_paper_content")) or os.getenv(
        "PAPERAGENT_ALLOW_EXTERNAL_LLM", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    
    return {
        "enabled": enabled,
        "paper_content_allowed": allow_external_paper_content,
        "chat_model": chat_model,
        "analyze_model": analyze_model,
        "supported_models": SUPPORTED_MODELS,
        "base_url": base_url,
        "provider": "SiliconFlow/OpenAI-compatible",
        "missing": [
            name
            for name, value in {
                "PAPERAGENT_LLM_API_KEY": api_key,
            }.items()
            if not value
        ],
    }


def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    response_format: dict | None = None,
    timeout: int = 180,
    role: str = "chat", # "chat" or "analyze"
) -> str | None:
    global LAST_ERROR
    LAST_ERROR = ""
    config = llm_config()
    if not config["enabled"]:
        LAST_ERROR = "LLM not enabled"
        return None

    model = config["analyze_model"] if role == "analyze" else config["chat_model"]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    cfg = local_config()
    api_key = os.getenv("PAPERAGENT_LLM_API_KEY", "").strip() or str(cfg.get("api_key", "")).strip()
    request = urllib.request.Request(
        config["base_url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        LAST_ERROR = f"HTTP {exc.code}: {body[:1200]}"
        return None
    except urllib.error.URLError as exc:
        LAST_ERROR = f"URL error: {exc}"
        return None
    except TimeoutError:
        LAST_ERROR = "Timeout"
        return None
    except json.JSONDecodeError as exc:
        LAST_ERROR = f"Invalid response JSON: {exc}"
        return None
    except KeyError as exc:
        LAST_ERROR = f"Missing response key: {exc}"
        return None

    try:
        content = data["choices"][0]["message"]["content"]
        reasoning = data["choices"][0]["message"].get("reasoning_content")
    except (KeyError, IndexError, TypeError) as exc:
        LAST_ERROR = f"Unexpected response shape: {exc}; preview={str(data)[:1000]}"
        return None
    
    # For analyze role, we don't stream, but we might want the thought later
    if reasoning and role == "chat":
         return f"<thought>\n{reasoning}\n</thought>\n\n{content}"
    return str(content or "").strip() or None


def stream_chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    role: str = "chat",
    timeout: int = 120,
):
    """Generator for streaming chat completion chunks."""
    config = llm_config()
    if not config["enabled"]:
        yield {"error": "LLM not enabled"}
        return

    model = config["analyze_model"] if role == "analyze" else config["chat_model"]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    cfg = local_config()
    api_key = os.getenv("PAPERAGENT_LLM_API_KEY", "").strip() or str(cfg.get("api_key", "")).strip()
    
    try:
        request = urllib.request.Request(
            config["base_url"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                    if content or reasoning:
                        yield {"content": content, "reasoning": reasoning}
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        yield {"error": str(e)}


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def answer_with_llm(question: str, results: list[SearchResult]) -> str | None:
    if not llm_config().get("paper_content_allowed"):
        return None
    messages = build_messages(question, results)
    return chat_completion(messages, temperature=0.2, max_tokens=4096, timeout=180)


def build_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    sources = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        sources.append(
            f"证据片段[{index}] {chunk.paper_title} | {chunk.section or 'Unknown section'} | "
            f"p.{chunk.page_start}-{chunk.page_end}\n{chunk.text[:2600]}"
        )
    source_text = "\n\n".join(sources)
    system = (
        "你是 PaperAgent，一个严谨的论文阅读 Agent。你会先综合证据，再回答用户。"
        "只能基于给定证据回答；如果证据不足，要明确说不足。"
        "回答要结构化、具体、中文为主。引用时必须把编号理解为证据片段编号，而不是文档编号；"
        "关键结论后标注证据片段编号，例如“[1]”。不要写“文档[1]”。"
        "不要照搬长段原文，要提炼 Motivation、Method、Result、Limitation 等模块化信息。"
    )
    user = f"问题：{question}\n\n证据：\n{source_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def llm_last_error() -> str:
    global LAST_ERROR
    return LAST_ERROR
