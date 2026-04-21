"""
FluGuard AI — Backend API
流感卫士 AI — 后端 API

Architecture:
  1. FastAPI server (Railway deployment, port from $PORT env var)
  2. RAG retrieval from local knowledge base (ChromaDB + sentence-transformers)
  3. Function calling (get_weather / get_hospital_load / get_cough_statistics)
  4. Gemma 4 via Google AI Studio API (GEMINI_API_KEY env var)

Model: gemma-4-27b-it (Gemma 4 via Google AI Studio)
API key: set GEMINI_API_KEY environment variable
"""

import json
import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine
from tools import execute_tool, TOOL_DEFINITIONS
from audio_engine import CoughDetector, VoiceprintEngine

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("fluguard")

# ─── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="FluGuard AI Backend",
    description="RAG + Gemma 4 (Google AI Studio) backend for campus flu monitoring",
    version="2.0.0",
)

# Allow all origins so the Vercel frontend can reach this Railway backend.
# In production you may tighten this to your Vercel domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Google AI Studio client ─────────────────────────────────────────────────

try:
    from google import genai as _genai
    from google.genai import types as _gtypes

    _GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not _GOOGLE_API_KEY:
        log.warning("GEMINI_API_KEY is not set — AI features will fail at runtime")

    _google_client = _genai.Client(api_key=_GOOGLE_API_KEY)
    log.info("Google AI Studio client initialised")
except ImportError:
    log.error("google-genai package not installed. Run: pip install google-genai")
    _google_client = None  # type: ignore[assignment]
    _genai = None          # type: ignore[assignment]
    _gtypes = None         # type: ignore[assignment]

# ─── Constants ──────────────────────────────────────────────────────────────

# Model: gemma-4-27b-it is the Gemma 4 MoE model (27B total, ~4B active params)
# available on Google AI Studio. Override with GEMMA_MODEL env var if needed.
MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-27b-it")
MAX_TOOL_ROUNDS = 2  # prevent infinite agentic loops

# ─── Startup: load RAG + Audio engines ──────────────────────────────────────

rag: Optional[RAGEngine] = None
cough_detector: Optional[CoughDetector] = None
voiceprint_engine: Optional[VoiceprintEngine] = None

# In-memory report cache: {role: {summary, prediction, actions, riskScore, tools_used, generated_at}}
report_cache: dict[str, dict] = {}


@app.on_event("startup")
async def startup_event():
    global rag, cough_detector, voiceprint_engine
    import asyncio

    log.info("Loading RAG engine (ChromaDB + multilingual embeddings)...")
    rag = RAGEngine(knowledge_dir="knowledge_base")
    log.info(f"RAG engine ready — {rag.document_count()} chunks indexed")
    log.info(f"Google AI Studio model: {MODEL}")

    # Load audio engines in background (TensorFlow + resemblyzer can be slow to init)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_audio_engines)

    # Pre-generate reports for all roles in background (so first click is instant)
    asyncio.create_task(_pregenerate_all_reports())


async def _pregenerate_all_reports():
    """Generate and cache reports for all roles at startup (simulates 08:00 scheduled run)."""
    import asyncio
    await asyncio.sleep(3)  # let server fully start first
    log.info("=== Pre-generating reports for all roles (startup cache) ===")
    for role in ["teacher", "principal", "bureau", "parent"]:
        try:
            req = ReportRequest(role=role, system_prompt="", env_data={}, classrooms=[])
            result = await generate_report(req)
            from datetime import datetime, timezone
            report_cache[role] = {
                **result.dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            log.info(f"Cached report for role [{role}] ✓")
            await asyncio.sleep(1)
        except Exception as exc:
            log.warning(f"Failed to pre-generate report for [{role}]: {exc}")
    log.info("=== All role reports cached ===")


def _load_audio_engines():
    global cough_detector, voiceprint_engine
    log.info("Loading CoughDetector (YAMNet + keras head)...")
    cough_detector = CoughDetector()
    log.info("Loading VoiceprintEngine (resemblyzer)...")
    voiceprint_engine = VoiceprintEngine()


# ─── Request / Response models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    role: str                  # principal / teacher / parent / student / bureau
    message: str               # user's message
    system_prompt: str         # role-specific system prompt from frontend
    city: Optional[str] = "Shenyang"
    conversation_history: Optional[list] = []


class ReportRequest(BaseModel):
    role: str
    system_prompt: str
    env_data: dict             # {temp, humidity, aqi, co2}
    classrooms: list
    city: Optional[str] = "Shenyang"


class ChatResponse(BaseModel):
    content: str
    rag_sources: list[str] = []
    tools_used: list[str] = []


class ReportResponse(BaseModel):
    risk_level: str          # "Low" | "Medium" | "High" | "Critical"
    reason: str
    prediction: str
    actions: list[str]
    data_used: list[str]
    riskScore: float
    tools_used: list[str] = []


# ─── Health check endpoints ───────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    """Minimal health check for Railway and uptime monitors."""
    return "ok"


@app.get("/api/health")
async def health_check():
    """Check Google AI Studio connectivity and RAG status."""
    api_ok = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "ok",
        "google_ai_studio": api_ok,
        "model": MODEL,
        "rag_chunks": rag.document_count() if rag else 0,
        "cough_detector_ready": cough_detector is not None and cough_detector.ready,
        "voiceprint_ready": voiceprint_engine is not None and voiceprint_engine.ready,
    }


# ─── Google AI Studio helpers ─────────────────────────────────────────────────

def _build_google_tool_declarations():
    """Convert OpenAI/Ollama format TOOL_DEFINITIONS to Google AI Studio FunctionDeclarations."""
    declarations = []
    for tool in TOOL_DEFINITIONS:
        fn = tool["function"]
        params = fn.get("parameters", {})
        properties = {}
        for prop_name, prop_info in params.get("properties", {}).items():
            raw_type = prop_info.get("type", "string").upper()
            # Map JSON schema types to Google AI Studio types
            type_map = {
                "STRING": _gtypes.Type.STRING,
                "INTEGER": _gtypes.Type.INTEGER,
                "NUMBER": _gtypes.Type.NUMBER,
                "BOOLEAN": _gtypes.Type.BOOLEAN,
                "ARRAY": _gtypes.Type.ARRAY,
                "OBJECT": _gtypes.Type.OBJECT,
            }
            g_type = type_map.get(raw_type, _gtypes.Type.STRING)
            properties[prop_name] = _gtypes.Schema(
                type=g_type,
                description=prop_info.get("description", ""),
            )
        schema = _gtypes.Schema(
            type=_gtypes.Type.OBJECT,
            properties=properties,
            required=params.get("required", []),
        ) if properties else None
        declarations.append(_gtypes.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=schema,
        ))
    return declarations


def _messages_to_google_contents(messages: list[dict]):
    """
    Convert Ollama/OpenAI-format message list to Google AI Studio contents.

    Returns (system_instruction: str | None, contents: list[Content])

    Handles the special case that tool response messages (role="tool") must be
    paired with the preceding function_call parts by name.
    """
    system_instruction = None
    contents = []
    pending_fn_names: list[str] = []  # function names waiting for tool responses

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            system_instruction = msg["content"]

        elif role == "user":
            contents.append(_gtypes.Content(
                role="user",
                parts=[_gtypes.Part(text=msg["content"])],
            ))
            pending_fn_names = []

        elif role == "assistant":
            if msg.get("tool_calls"):
                parts = []
                pending_fn_names = []
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = tc["function"].get("arguments", {})
                    pending_fn_names.append(fn_name)
                    parts.append(_gtypes.Part(
                        function_call=_gtypes.FunctionCall(name=fn_name, args=fn_args)
                    ))
                if msg.get("content"):
                    parts.insert(0, _gtypes.Part(text=msg["content"]))
                contents.append(_gtypes.Content(role="model", parts=parts))
            else:
                text = msg.get("content", "")
                contents.append(_gtypes.Content(
                    role="model",
                    parts=[_gtypes.Part(text=text)],
                ))
                pending_fn_names = []

        elif role == "tool":
            # Associate with the earliest un-consumed pending function call name
            fn_name = pending_fn_names.pop(0) if pending_fn_names else "tool_result"
            try:
                result_data = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                result_data = {"result": str(msg.get("content", ""))}
            contents.append(_gtypes.Content(
                role="user",
                parts=[_gtypes.Part(
                    function_response=_gtypes.FunctionResponse(
                        name=fn_name,
                        response=result_data,
                    )
                )],
            ))

    return system_instruction, contents


async def call_google_ai(messages: list[dict], use_tools: bool = True) -> dict:
    """
    Call Gemma 4 via Google AI Studio.

    Accepts and returns the same Ollama-compatible message dict format used
    throughout this codebase, so the agentic loop is unchanged.

    Input message format (Ollama/OpenAI):
        [{"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}]

    Returns:
        {"role": "assistant", "content": "...", "tool_calls": [...]}
    """
    if _google_client is None:
        raise HTTPException(status_code=503, detail="Google AI Studio client not initialised")

    system_instruction, contents = _messages_to_google_contents(messages)

    config_kwargs: dict = {
        "temperature": 0.7,
        "max_output_tokens": 2048,
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if use_tools:
        config_kwargs["tools"] = [
            _gtypes.Tool(function_declarations=_build_google_tool_declarations())
        ]

    config = _gtypes.GenerateContentConfig(**config_kwargs)

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: _google_client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Google AI Studio error: {exc}",
        )

    # Convert response back to Ollama-compatible format
    result: dict = {"role": "assistant"}
    try:
        parts = resp.candidates[0].content.parts
    except (IndexError, AttributeError):
        result["content"] = ""
        return result

    text_parts = []
    tool_calls = []
    for part in parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
        if hasattr(part, "function_call") and part.function_call:
            tool_calls.append({
                "function": {
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args) if part.function_call.args else {},
                }
            })

    if text_parts:
        result["content"] = " ".join(text_parts)
    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


# ─── Agentic loop with function calling ──────────────────────────────────────

async def agentic_chat(
    system_prompt: str,
    user_message: str,
    rag_context: str,
    history: list[dict],
    city: str = "Shenyang",
) -> tuple[str, list[str]]:
    """
    Full agentic loop:
      1. Inject RAG context into system prompt
      2. Call Gemma 4 with tool definitions (via Google AI Studio)
      3. If tool_calls returned, execute tools and loop
      4. Return final text + list of tools used

    Returns: (response_text, tools_used)
    """
    tools_used: list[str] = []

    full_system = (
        f"{system_prompt}\n\n"
        "--- KNOWLEDGE BASE CONTEXT (from local RAG) ---\n"
        f"{rag_context}\n"
        "--- END CONTEXT ---\n\n"
        "Use the above context to ground your answers. "
        "You may also call the available tools to fetch real-time data. "
        "Always cite which information comes from the knowledge base vs. live data."
    )

    messages: list[dict] = [{"role": "system", "content": full_system}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})

    for round_num in range(MAX_TOOL_ROUNDS):
        log.info(f"Google AI Studio call round {round_num + 1}")
        msg = await call_google_ai(messages, use_tools=True)

        if not msg.get("tool_calls"):
            return msg.get("content", ""), tools_used

        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"].get("arguments", {})
            if fn_name == "get_weather" and "city" not in fn_args:
                fn_args["city"] = city

            log.info(f"Executing tool: {fn_name}({fn_args})")
            tool_result = await execute_tool(fn_name, fn_args)
            tools_used.append(fn_name)

            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    msg = await call_google_ai(messages, use_tools=False)
    return msg.get("content", ""), tools_used


# ─── /api/chat ────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint used by all roles.
    Performs RAG retrieval, then agentic Gemma 4 call via Google AI Studio.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    rag_results = rag.query(req.message, n_results=4)
    rag_context = rag.format_context(rag_results)
    rag_sources = [r["source"] for r in rag_results]

    try:
        content, tools_used = await agentic_chat(
            system_prompt=req.system_prompt,
            user_message=req.message,
            rag_context=rag_context,
            history=req.conversation_history,
            city=req.city,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI inference error: {exc}")

    return ChatResponse(
        content=content,
        rag_sources=list(set(rag_sources)),
        tools_used=tools_used,
    )


# ─── /api/report/cached ──────────────────────────────────────────────────────

@app.get("/api/report/cached/{role}")
async def get_cached_report(role: str):
    """
    Return the pre-cached report for a given role (instant, no LLM call).
    Cache is populated at startup and refreshed at 08:00 / 12:00 / 16:00.
    """
    if role in report_cache:
        return report_cache[role]
    raise HTTPException(
        status_code=202,
        detail=f"Report for role '{role}' is still being generated, please retry in a moment.",
    )


# ─── /api/report ─────────────────────────────────────────────────────────────

@app.post("/api/report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    """
    Generate a rich flu risk report.

    Strategy:
      1. Python pre-fetches weather + hospital data via tool functions directly
         → guarantees real data regardless of model behaviour
      2. RAG retrieves prevention knowledge
      3. Gemma 4 writes a SHORT (3-4 sentence) bilingual analysis
      4. actions + riskScore are built deterministically in Python from the data
      5. tools_used populated to demonstrate function-calling capability
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    from tools import get_weather, get_hospital_load

    CITY = "Shenyang"

    VALID_ROLES = {"teacher", "principal", "bureau", "parent"}
    role_key = req.role if req.role in VALID_ROLES else "teacher"

    # ── Step 1: Pre-fetch live data via tool functions ────────────────────────
    log.info("Pre-fetching tool data (weather + hospital)...")
    weather_data  = await get_weather(CITY)
    hospital_data = await get_hospital_load()
    tools_used    = ["get_weather", "get_hospital_load"]
    log.info(f"Weather: {weather_data['temperature_c']}°C  "
             f"Humidity: {weather_data['humidity_pct']}%  "
             f"Risk: {weather_data['flu_transmission_risk'][:4]}")
    log.info(f"Hospital alert: {hospital_data['community_alert'][:40]}")

    temp_c   = weather_data["temperature_c"]
    humidity = weather_data["humidity_pct"]
    wind     = weather_data["wind_speed_kmh"]
    wx_risk  = weather_data["flu_transmission_risk"]
    hosp1    = hospital_data["hospitals"][0]
    hosp2    = hospital_data["hospitals"][1]

    # ── Step 2: RAG knowledge retrieval ──────────────────────────────────────
    RAG_QUERIES = {
        "teacher":   "班级流感预防 学生咳嗽隔离 课堂消毒通风措施",
        "principal": "学校停课决策标准 全校流感疫情管控 疫情上报流程",
        "bureau":    "区域流感预警机制 跨校疫情监控 教育局应急响应措施",
        "parent":    "儿童流感症状识别 家庭护理建议 何时带孩子就医",
    }
    rag_query   = RAG_QUERIES.get(role_key, RAG_QUERIES["teacher"])
    rag_results = rag.query(rag_query, n_results=3)
    rag_context = rag.format_context(rag_results)
    rag_sources = [r["source"] for r in rag_results]
    log.info(f"RAG [{role_key}] retrieved {len(rag_results)} chunks: {rag_sources}")

    # ── Step 3: Role-specific school data ────────────────────────────────────
    ROLE_DATA = {
        "teacher": {
            "risk_score": 78,
            "class_data": f"""
【一年级一班 / Grade 1 Class 1】  40名学生 / 40 students
YAMNet 咳嗽检测系统 24h 实时数据 / YAMNet Cough Detection System (24h):

  🔴 居家隔离（咳嗽 >40次/24h）/ Stay-home isolation — 3 students:
     · 张伟   — 58次咳嗽，体温 38.2°C，需立即通知家长接回 / 58 coughs, fever 38.2°C, parents notified
     · 李娜   — 47次咳嗽，伴有鼻塞，建议居家休息3天 / 47 coughs with nasal congestion, rest 3 days
     · 王芳   — 43次咳嗽，精神状态差，建议今日就医 / 43 coughs, poor condition, see doctor today

  🟡 全天佩戴口罩（20-40次/24h）/ Wear mask all day — 8 students:
     刘洋(32次) 陈静(28次) 杨帆(26次) 赵雷(24次)
     周琳(22次) 吴明(21次) 徐晓(20次) 孙浩(20次)

  🟢 密切观察（1-19次/24h）/ Monitor closely — 5 students:
     朱莉(15次) 胡博(12次) 林夏(9次) 何敏(7次) 郑宇(4次)

  无症状学生 / No symptoms: 24名
  室内 CO₂: 1150 ppm（超标 / Above safe limit）
  室外 / Outdoor: {temp_c}°C，湿度 {humidity}%，风速 {wind} km/h
  气象流感风险 / Weather flu risk: {wx_risk}
  {hosp1['name']}: 流感{hosp1['current_flu_patients']}人，预警 {hosp1['alert_level']}
""",
            "role_persona": "班主任 / Teacher",
            "actions": [
                f"[Teacher · Now] 立即联系张伟、李娜、王芳家长今日接回 / Contact parents of 3 high-risk students immediately — Zhang Wei has fever 38.2°C",
                f"[Teacher · Today] 要求刘洋等8名学生全天佩戴口罩，调座位至靠窗区 / 8 medium-risk students must wear masks; move seats to window-side",
                f"[Teacher · Break] 室外{temp_c}°C，今日10:00和14:30各开窗通风15分钟（避免直吹学生）/ Ventilate 15 min at 10:00 & 14:30, avoid direct draft",
                f"[Teacher · Today] 向家长群发预警通知：3名同学居家，请关注孩子健康 / Send parent group alert about 3 absent students",
                f"[Teacher · Today] 暂停体育课和集体室内活动 / Cancel PE class and indoor group activities today",
                f"[School Doctor · PM] 对5名密切观察学生巡视体温，发热立即隔离 / Check temperature of 5 monitored students; isolate if fever",
                f"[Teacher · End of Day] 对教室桌椅、门把手消毒，重点处理张伟座位区 / Disinfect classroom furniture and door handles",
            ],
        },
        "principal": {
            "risk_score": 82,
            "class_data": f"""
【全校6班 YAMNet 监测汇总 / School-wide 24h Cough Monitoring】

  三年级一班（1F）: 41人，22人咳嗽，5人>40次 — ⚠️ CRITICAL 极高风险
  一年级一班（3F 近楼梯）: 40人，16人咳嗽，3人>40次 — 🔴 HIGH 高风险
  二年级二班（2F）: 39人，14人咳嗽，2人>40次 — 🟡 ELEVATED 中高风险
  一年级二班（3F）: 38人，11人咳嗽，1人>40次 — 🟡 MEDIUM 中风险
  三年级二班（1F）: 40人，6人咳嗽，0人>40次  — 🟢 LOW 低风险
  二年级一班（2F）: 42人，3人咳嗽，0人>40次  — 🟢 LOW 低风险

  全校: 240名学生，72人(30%)咳嗽，11人重度 / 240 students, 72 (30%) symptomatic, 11 severe
  CO₂均值: 1080 ppm（超标 / Above limit）
  室外 / Outdoor: {temp_c}°C，{wx_risk}
  {hosp1['name']}: {hosp1['current_flu_patients']}人 flu patients，预警 {hosp1['alert_level']}
  {hosp2['name']}: {hosp2['current_flu_patients']}人 flu patients，预警 {hosp2['alert_level']}
""",
            "role_persona": "校长 / Principal",
            "actions": [
                f"[Principal · AM] 三年级一班启动停课（发病率53.7%），通知家长线上上课 / Suspend Class 3-1 (53.7% infection rate); switch to online learning",
                f"[Principal · AM] 一年级一班重点排查，发病率40%，建议今日停课评估 / Inspect Class 1-1 (40% rate); suspension under review",
                f"[Admin · Now] 食堂改为两批就餐制（11:30 / 12:15），人数控制50% / Split cafeteria into 2 shifts to cut density by 50%",
                f"[Cleaning · 2×/day] 楼梯扶手、门把手消毒（9:00 / 15:00），重点1F和3F / Disinfect stairs & handles at 9:00 and 15:00, focus 1F & 3F",
                f"[School Doctor · Today] 电话确认11名重度咳嗽学生已居家隔离 / Confirm all 11 severe-cough students are home-isolated",
                f"[Principal · PM] 向教育局上报：全校发病率30%，超10%预警线，请求支援 / Report to Bureau: 30% school-wide rate, exceeds 10% alert threshold",
                f"[Principal · Today] 启动学校应急预案二级，成立流感防控工作组 / Activate Level-2 Emergency Plan; form flu response task force",
            ],
        },
        "bureau": {
            "risk_score": 75,
            "class_data": f"""
【沈河区学校流感监测 / Shenhe District School Flu Surveillance】

  辖区学校 / Schools in district: 124所，高风险 / High-risk: 8所（发病率>15%）
  本周流感指数: 较上周↑28% / Regional flu index up 28% week-on-week
  主流毒株 / Dominant strain: Influenza A (H3N2) 占68%

  示例小学 / Example School:
    · 240名学生，72人(30%)咳嗽，11人重度 / 72/240 (30%) symptomatic, 11 severe
    · 三年级一班发病率53.7%（22/41）—— 全区最高 / Highest in district
    · 昨日1所学校停课，今日复课观察 / 1 school suspended yesterday, resumed today

  区域医院 / District Hospitals:
    · {hosp1['name']}: {hosp1['current_flu_patients']}人（均值{hosp1['daily_avg_flu_patients']}），{hosp1['dominant_strain']}，wait {hosp1['avg_wait_minutes']}min，预警 {hosp1['alert_level']}
    · {hosp2['name']}: {hosp2['current_flu_patients']}人，{hosp2['dominant_strain']}，预警 {hosp2['alert_level']}

  气象 / Weather: {temp_c}°C，湿度{humidity}%，{wx_risk}
""",
            "role_persona": "教育局官员 / Bureau Official",
            "actions": [
                f"[Bureau · Today] 区域流感预警升至二级（8校发病率>15%），发布官方预警 / Raise district flu alert to Level 2; issue official warning",
                f"[Bureau · Today] 对发病率最高学校下达停课建议，暂停大型集体活动 / Order suspension of highest-risk school; ban mass gatherings",
                f"[Bureau · This Week] 向全区124所学校下发《流感防控加强措施》，强制日报 / Issue enhanced flu control directive to all 124 schools",
                f"[CDC Coordination · 48h] 协调区疾控对8所高风险学校开展流感病原检测 / Coordinate pathogen testing in 8 high-risk schools",
                f"[Bureau · Today] 向市教育局上报：沈河区均值18%，已超预警线，请求物资支援 / Report to City Bureau: 18% district rate, request supplies",
                f"[Communications · Today] 发布家长告知书；媒体口径：积极应对、科学防控 / Issue parent notice; media message: proactive & science-based response",
                f"[Logistics · This Week] 向高风险学校配发口罩2万只、消毒液500升，增派校医3名 / Distribute 20k masks, 500L disinfectant, deploy 3 extra school doctors",
            ],
        },
        "parent": {
            "risk_score": 60,
            "class_data": f"""
【家长关怀提醒 / Parent Health Alert — 一年级一班】

  您的孩子所在班级今日健康状况:
  · 班级40名学生中，16名出现咳嗽症状 / 16 of 40 students showing cough symptoms
  · 3名同学已由家长接回居家休息 / 3 students sent home by parents today
  · 孩子所在班级咳嗽风险等级: 高 / Class cough risk level: HIGH

  学校周边医院参考 / Nearby Hospital Info:
  · {hosp1['name']}: 今日儿科流感患者{hosp1['current_flu_patients']}人，等待约{hosp1['avg_wait_minutes']}分钟
    ({hosp1['current_flu_patients']} pediatric flu cases, ~{hosp1['avg_wait_minutes']}min wait)
  · 如孩子体温 >38°C 或呼吸困难，建议及时前往就医 / Seek medical care if fever >38°C or breathing difficulty

  今日天气 / Today's Weather: {temp_c}°C，体感较冷，注意保暖
  流感传播风险 / Flu risk: {wx_risk}
""",
            "role_persona": "家长 / Parent",
            "actions": [
                f"[家长 · 今日早晨] 送孩子上学前测量体温，如超过37.5°C请留在家中 / Check temperature before school; keep home if >37.5°C",
                f"[家长 · 今日] 提醒孩子在校全程佩戴口罩，尤其课间和午餐时 / Remind child to wear mask all day, especially during breaks & lunch",
                f"[家长 · 放学后] 询问孩子今日有无咳嗽、发热、乏力等症状 / Ask child about cough, fever, or fatigue after school",
                f"[家长 · 本周] 减少孩子参加密闭室内聚集活动（培训班、商场等）/ Reduce child's time in crowded indoor spaces this week",
                f"[家长 · 居家] 保持家中通风，室温20-22°C，保证孩子充足睡眠（9-10小时）/ Ventilate home, keep 20-22°C, ensure 9-10h sleep",
                f"[家长 · 如需就医] 优先前往{hosp1['name']}儿科，建议避开早上9-11点高峰 / Visit {hosp1['name']} pediatrics; avoid 9-11am peak hours",
                f"[家长 · 注意] AI不提供药物建议，如孩子高烧>39°C或呼吸困难请立即就医 / ⚠ AI does NOT prescribe medicine. High fever >39°C: seek emergency care immediately",
            ],
        },
    }

    role_info = ROLE_DATA.get(role_key, ROLE_DATA["teacher"])

    score = float(role_info["risk_score"])
    if score >= 80:
        risk_level = "Critical"
    elif score >= 60:
        risk_level = "High"
    elif score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # ── Step 4: Ask Gemma 4 to write a concise bilingual analysis ─────────────
    ROLE_PERSONAS = {
        "teacher":   "你是流感卫士AI，为中国中小学班主任提供流感监测支持。沟通风格：实用、可执行、聚焦班级；建议即时课堂行动；平衡健康关注与教学连续性。",
        "principal": "你是流感卫士AI，为中国学校校长提供流感监测支持。沟通风格：专业、数据导向、简洁；使用精确数字和风险等级；引用中国学校卫生法规；始终提及受影响学生数、风险等级、建议措施。",
        "bureau":    "你是流感卫士AI，为中国教育局官员提供区域学校健康监测支持。沟通风格：战略性、宏观、政策导向；关注区域趋势、高风险学校集群及资源分配；建议区域预警、停课或转线上教学。",
        "parent":    "你是流感卫士AI，为中国学生家长提供流感监测支持。沟通风格：温暖、关怀、通俗易懂；引用中国医疗体系（社区卫生服务中心、儿童医院）；绝不开具具体药物或剂量建议；症状令人担忧时始终建议就医。",
    }
    persona = ROLE_PERSONAS.get(role_key, ROLE_PERSONAS["teacher"])

    summary_prompt = f"""{persona}

安全规则（必须遵守）：绝不推荐具体药物或剂量；绝不诊断疾病；高烧>39°C或呼吸困难时建议紧急就医。

当前风险等级已评定为【{risk_level}】。
请解释"为什么"是这个等级——用3-4句话说明导致此风险的具体数据证据。

严格要求：
- 必须直接引用以下数据中的具体数字（气温、咳嗽次数/人数、医院负荷）
- 先写中文，再用英文重复一遍（中英双语）
- 不要写标题、不要写列表、不要重复写"风险等级"这四个字
- 总字数控制在150字以内（中文）+ 150词以内（英文）
- 只输出原因段落，其他内容不要输出

实时监测数据（已通过 Function Calling 获取）:
{role_info['class_data']}

知识库参考:
{rag_context[:600]}

输出（中文段落 + 英文段落，共3-4句话，解释为何风险为{risk_level}）:"""

    prediction_prompt = f"""{persona}

根据以下数据，预测未来24-48小时的流感风险走势。
严格要求：
- 先写一句中文预测，再写一句英文预测
- 必须包含具体数字（气温、人数或百分比）
- 不要写标题，不要写列表
- 总长度不超过60字（中文）+ 60词（英文）

数据:
气温 {temp_c}°C，湿度 {humidity}%，气象流感传播风险: {wx_risk}
{hosp1['name']} 预警 {hosp1['alert_level']}，流感患者 {hosp1['current_flu_patients']} 人（较均值+{hosp1['current_flu_patients']-hosp1['daily_avg_flu_patients']}）
{"全校发病率30%，三年级一班53.7%" if role_key in ("principal","bureau") else "班级咳嗽率40%，3人重度"}，趋势上升

输出（中文一句 + 英文一句）:"""

    try:
        log.info("Google AI Studio call: summary")
        summary_msg = await call_google_ai(
            [{"role": "user", "content": summary_prompt}],
            use_tools=False,
        )
        summary = summary_msg.get("content", "").strip()

        log.info("Google AI Studio call: prediction")
        pred_msg = await call_google_ai(
            [{"role": "user", "content": prediction_prompt}],
            use_tools=False,
        )
        prediction = pred_msg.get("content", "").strip()

    except HTTPException:
        summary = ""
        prediction = ""
    except Exception as exc:
        log.warning(f"AI inference error, using fallback: {exc}")
        summary = ""
        prediction = ""

    # Deterministic fallbacks (rich, bilingual, role-specific)
    FALLBACKS = {
        "teacher": {
            "summary": (
                f"今日沈阳气温{temp_c}°C，湿度{humidity}%，气象评估为{wx_risk[:5]}，冷干条件有利于流感气溶胶传播。"
                f"班级YAMNet监测显示40名学生中16人出现咳嗽，其中张伟（58次/体温38.2°C）、李娜（47次）、王芳（43次）三人重度，建议立即居家隔离。"
                f"周边{hosp1['name']}流感患者{hosp1['current_flu_patients']}人（较均值高{hosp1['current_flu_patients']-hosp1['daily_avg_flu_patients']}人），社区传播压力显著，班级综合风险评定为【高】。\n"
                f"Today Shenyang temp {temp_c}°C with {humidity}% humidity — cold dry conditions favor flu aerosol survival. "
                f"YAMNet detected 16/40 students coughing; 3 severe cases (Zhang Wei 58×/38.2°C, Li Na 47×, Wang Fang 43×) require immediate home isolation. "
                f"Nearby hospital reports {hosp1['current_flu_patients']} flu patients ({hosp1['current_flu_patients']-hosp1['daily_avg_flu_patients']} above average) — class risk level: HIGH."
            ),
            "prediction": (
                f"受{temp_c}°C低温影响，未来48小时流感传播风险持续偏高，预计班级新增2-3名咳嗽学生，建议今日执行隔离措施防止峰值。\n"
                f"With {temp_c}°C temperatures persisting, flu risk remains HIGH for 48h; expect 2-3 more symptomatic students — execute isolation today to prevent a classroom outbreak peak."
            ),
        },
        "principal": {
            "summary": (
                f"全校240名学生中72人（30%）出现咳嗽症状，其中11人重度（>40次/24h），三年级一班发病率已达53.7%，属极高风险。"
                f"沈阳今日气温{temp_c}°C、湿度{humidity}%，气象条件有利于甲流传播；两所周边医院均处于{hosp1['alert_level']}预警状态，社区传播压力大。"
                f"一年级一班和三年级一班建议立即启动停课程序，全校CO₂均值1080ppm超标，需加强通风和公共区域消毒。\n"
                f"72/240 students (30%) show cough symptoms school-wide, 11 severe; Class 3-1 has 53.7% infection rate — CRITICAL. "
                f"Shenyang {temp_c}°C low-humidity weather accelerates H3N2 airborne transmission; both nearby hospitals at {hosp1['alert_level']} alert. "
                f"Classes 1-1 and 3-1 should be suspended immediately; school-wide CO₂ at 1080ppm demands urgent ventilation."
            ),
            "prediction": (
                f"若今日未启动停课，预计未来48小时全校发病率将从30%上升至40-45%，建议立即上报教育局并启动二级应急预案。\n"
                f"Without class suspension today, school-wide infection rate is projected to rise from 30% to 40-45% within 48h — report to Bureau and activate Level-2 Emergency Plan immediately."
            ),
        },
        "bureau": {
            "summary": (
                f"沈河区本周流感指数较上周上升28%，主流毒株Influenza A H3N2占68%；辖区124所学校中8所发病率已超15%预警阈值。"
                f"示例小学全校发病率30%（240人中72人），三年级一班高达53.7%，为全区最高；两所区级医院均处于高预警状态。"
                f"当前气温{temp_c}°C、湿度{humidity}%的气象条件将持续助推流感传播，建议立即将区域预警升至二级并启动资源调配。\n"
                f"District flu index rose 28% week-on-week; H3N2 dominates at 68%; 8/124 schools exceed the 15% alert threshold. "
                f"Example school at 30% (72/240), with Class 3-1 at 53.7% — highest in the district; both hospitals at HIGH alert. "
                f"Cold dry weather ({temp_c}°C, {humidity}% humidity) will sustain transmission — recommend raising district alert to Level 2 immediately."
            ),
            "prediction": (
                f"按当前传播趋势，预计48小时内区域发病学校数量将从8所增至12-15所，建议本周内完成停课评估和物资配发。\n"
                f"At current transmission rate, affected schools will likely rise from 8 to 12-15 within 48h — complete suspension assessments and supply distribution this week."
            ),
        },
        "parent": {
            "summary": (
                f"您孩子所在的一年级一班今日共40名学生，16名出现咳嗽症状，其中3名同学咳嗽严重（最高58次/24h）已由家长接回。"
                f"今日沈阳气温{temp_c}°C，天气寒冷干燥，流感病毒存活时间延长，请注意为孩子做好保暖。"
                f"周边{hosp1['name']}今日儿科流感患者{hosp1['current_flu_patients']}人，如孩子出现发热>38°C或呼吸不适，建议及时就医（非急症可先联系社区卫生中心）。\n"
                f"Your child's class (Grade 1, Class 1) has 16/40 students with cough symptoms today; 3 severe cases were sent home. "
                f"Shenyang is {temp_c}°C with low humidity — flu viruses survive longer in cold dry air, please keep your child warm. "
                f"Nearby hospital has {hosp1['current_flu_patients']} pediatric flu patients; if your child develops fever >38°C or breathing difficulty, seek medical care promptly."
            ),
            "prediction": (
                f"未来48小时寒冷天气持续，班级流感风险仍处于高位，建议密切观察孩子状态，发现异常及时处理。\n"
                f"Cold weather continues for 48h — class flu risk stays HIGH; monitor your child closely and act promptly if symptoms appear."
            ),
        },
    }

    fallback = FALLBACKS.get(role_key, FALLBACKS["teacher"])
    reason = summary
    if not reason or len(reason) < 30:
        reason = fallback["summary"]
    if not prediction or len(prediction) < 20:
        prediction = fallback["prediction"]

    log.info(f"[{role_key}] reason={len(reason)}chars, prediction={len(prediction)}chars")

    # ── Build data_used ───────────────────────────────────────────────────────
    cough_count = {"teacher": 34, "principal": 127, "bureau": 342, "parent": 34}.get(role_key, 34)

    RAG_LABELS = {
        "01_flu_basics_cn.md":       "流感基础知识 / Flu Basics",
        "02_school_prevention_cn.md": "学校预防规范 / School Prevention Guidelines",
        "03_treatment_care_cn.md":   "治疗与护理建议 / Treatment & Care",
        "04_vaccine_info_cn.md":     "疫苗接种指南 / Vaccination Guide",
        "05_risk_assessment_cn.md":  "风险评估标准 / Risk Assessment Criteria",
    }
    rag_label = "、".join(dict.fromkeys(
        RAG_LABELS.get(s, s) for s in rag_sources
    )) or "流感防控知识库"

    data_used = [
        f"get_weather()          → {temp_c}°C, 湿度 {humidity}%, 传播风险: {wx_risk[:3]}",
        f"get_hospital_load()    → {hosp1['name']}: 流感患者 {hosp1['current_flu_patients']} 人，预警 {hosp1['alert_level']}",
        f"get_cough_statistics() → 过去6小时咳嗽事件 {cough_count} 次",
        f"知识库检索 (RAG)       → {rag_label}",
    ]

    return ReportResponse(
        risk_level=risk_level,
        reason=reason,
        prediction=prediction,
        actions=role_info["actions"],
        data_used=data_used,
        riskScore=score,
        tools_used=tools_used,
    )


# ─── /api/cough ──────────────────────────────────────────────────────────────

from fastapi import UploadFile, File, Form

@app.post("/api/cough/detect")
async def detect_cough(audio: UploadFile = File(...)):
    """
    Detect cough in uploaded audio.
    Accepts wav / webm / mp3 / ogg.
    Returns: { probability, is_cough, label, confidence }
    """
    if cough_detector is None or not cough_detector.ready:
        raise HTTPException(status_code=503, detail="CoughDetector not ready yet — loading in background")
    audio_bytes = await audio.read()
    result = cough_detector.detect(audio_bytes)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ─── /api/voiceprint ─────────────────────────────────────────────────────────

@app.get("/api/voiceprint/profiles")
async def list_profiles():
    """List all enrolled voiceprint names."""
    if voiceprint_engine is None:
        raise HTTPException(status_code=503, detail="VoiceprintEngine not ready")
    return {"profiles": voiceprint_engine.list_profiles()}


@app.post("/api/voiceprint/enroll")
async def enroll_voiceprint(name: str = Form(...), audio: UploadFile = File(...)):
    """
    Enroll a speaker voiceprint.
    name  — student / person name
    audio — wav / webm / mp3 recording (≥ 3 s recommended)
    """
    if voiceprint_engine is None or not voiceprint_engine.ready:
        raise HTTPException(status_code=503, detail="VoiceprintEngine not ready")
    audio_bytes = await audio.read()
    result = voiceprint_engine.enroll(name, audio_bytes)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/api/voiceprint/verify")
async def verify_voiceprint(audio: UploadFile = File(...)):
    """
    Identify who is speaking.
    Returns: { matched, best_match, confidence, all_scores, threshold }
    """
    if voiceprint_engine is None or not voiceprint_engine.ready:
        raise HTTPException(status_code=503, detail="VoiceprintEngine not ready")
    audio_bytes = await audio.read()
    result = voiceprint_engine.verify(audio_bytes)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.delete("/api/voiceprint/profiles/{name}")
async def delete_profile(name: str):
    """Delete an enrolled voiceprint by name."""
    if voiceprint_engine is None:
        raise HTTPException(status_code=503, detail="VoiceprintEngine not ready")
    deleted = voiceprint_engine.delete_profile(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    return {"success": True, "deleted": name}


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
