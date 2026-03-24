import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CryptoSignal AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "https://poc.moph.go.th/vllm/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "sk-my-secret-key")
VLLM_MODEL = os.getenv("VLLM_MODEL", "mastermind-coder")


class AnalysisRequest(BaseModel):
    coin: str
    price: float
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    ema12: float | None = None
    ema26: float | None = None
    volume_change: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    fib_levels: dict | None = None
    news_summary: str | None = None
    market_rank_data: dict | None = None


class ChatRequest(BaseModel):
    message: str
    context: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_signal(req: AnalysisRequest):
    indicators = []
    if req.rsi is not None:
        status = "Oversold" if req.rsi < 30 else "Overbought" if req.rsi > 70 else "Neutral"
        indicators.append(f"RSI(14): {req.rsi:.1f} ({status})")
    if req.macd is not None and req.macd_signal is not None:
        cross = "Bullish" if req.macd > req.macd_signal else "Bearish"
        indicators.append(f"MACD: {req.macd:.4f}, Signal: {req.macd_signal:.4f} ({cross} Crossover)")
    if req.ema12 is not None and req.ema26 is not None:
        cross = "Golden Cross" if req.ema12 > req.ema26 else "Death Cross"
        indicators.append(f"EMA12: {req.ema12:.2f}, EMA26: {req.ema26:.2f} ({cross})")
    if req.volume_change is not None:
        indicators.append(f"Volume Change: {req.volume_change:+.1f}%")
    if req.bb_upper is not None and req.bb_lower is not None:
        pos = "Near Upper Band" if req.price > (req.bb_upper + req.bb_lower) / 2 else "Near Lower Band"
        indicators.append(f"Bollinger Bands: Upper {req.bb_upper:.2f}, Lower {req.bb_lower:.2f} ({pos})")

    indicator_text = "\n".join(f"- {ind}" for ind in indicators) if indicators else "- No indicator data available"

    news_text = f"\nNews: {req.news_summary}" if req.news_summary else ""
    rank_text = ""
    if req.market_rank_data:
        rank_text = f"\nMarket Rank Data: Social Score={req.market_rank_data.get('socialScore', 'N/A')}, Smart Money={req.market_rank_data.get('smartMoney', 'N/A')}"

    prompt = f"""You are a Senior Crypto Technical Analyst. Analyze the following data for {req.coin} and provide a trading signal.

Current Price: ${req.price:,.2f}

Technical Indicators:
{indicator_text}
{news_text}
{rank_text}

Provide your analysis in this exact JSON format:
{{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100,
  "risk": "LOW" or "MEDIUM" or "HIGH",
  "target_price": number,
  "stop_loss": number,
  "analysis": "Thai language analysis summary (2-3 sentences)",
  "reasons": ["reason1", "reason2", "reason3"]
}}

Respond ONLY with the JSON, no other text."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VLLM_API_KEY}",
                },
                json={
                    "model": VLLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a Senior Crypto Technical Analyst. Always respond in valid JSON format."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            import json
            # Try to parse JSON from the response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1).strip())
                else:
                    result = {
                        "signal": "HOLD",
                        "confidence": 50,
                        "risk": "MEDIUM",
                        "target_price": req.price * 1.05,
                        "stop_loss": req.price * 0.95,
                        "analysis": content[:200],
                        "reasons": ["AI response parsing error"],
                    }

            return {"status": "ok", "data": result}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    system_prompt = "You are a Senior Crypto Analyst AI assistant named CryptoSignal AI. Answer in Thai. Be concise and helpful."
    if req.context:
        system_prompt += f"\n\nContext:\n{req.context}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {VLLM_API_KEY}",
                },
                json={
                    "model": VLLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": req.message},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {"status": "ok", "reply": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899)
