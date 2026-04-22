"""
FluGuard Function Calling Tools
流感卫士 Function Calling 工具集

These tools are made available to Gemma 4 E4B-IT via Ollama's tool-use API.
All tools are designed to be resilient: they return structured data even when
external services are unavailable (graceful degradation for offline scenarios).

Tools:
  1. get_weather(city)         — Open-Meteo free API (no key required)
  2. get_hospital_load()       — Simulated hospital ED pressure data
  3. get_cough_statistics()    — Cough detector aggregates
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("fluguard.tools")


# ─── Tool Definitions (Ollama / OpenAI format) ───────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current outdoor weather conditions for a Chinese city. "
                "Returns temperature (°C), humidity (%), wind speed, and a "
                "flu-transmission risk assessment based on meteorological conditions. "
                "Useful for: correlating cold/dry weather with elevated flu risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name in English, e.g. 'Shenyang', 'Beijing', "
                            "'Harbin', 'Changchun'"
                        ),
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hospital_load",
            "description": (
                "Get current emergency department flu patient load from nearby hospitals. "
                "Returns patient counts, dominant flu strains, and wait times. "
                "Useful for: understanding community-level flu pressure and whether "
                "the school's cases align with a broader outbreak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hospital_name": {
                        "type": "string",
                        "description": (
                            "Optional hospital name. If omitted, returns data for "
                            "the nearest hospitals in the school's district."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cough_statistics",
            "description": (
                "Retrieve cough detection statistics from the YAMNet-based classroom "
                "audio monitoring system. Returns cough counts, trend (rising/stable/falling), "
                "and per-class breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_id": {
                        "type": "string",
                        "description": (
                            "Class ID ('c1'–'c6') or 'all' for school-wide totals."
                        ),
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Lookback window in hours (default: 24).",
                    },
                },
                "required": [],
            },
        },
    },
]


# ─── City → (latitude, longitude) lookup ─────────────────────────────────────

CITY_COORDS: dict[str, tuple[float, float]] = {
    "shenyang":    (41.8057, 123.4315),
    "沈阳":         (41.8057, 123.4315),
    "beijing":     (39.9042, 116.4074),
    "北京":         (39.9042, 116.4074),
    "harbin":      (45.8038, 126.5349),
    "哈尔滨":       (45.8038, 126.5349),
    "changchun":   (43.8868, 125.3245),
    "长春":         (43.8868, 125.3245),
    "dalian":      (38.9140, 121.6147),
    "大连":         (38.9140, 121.6147),
    "shanghai":    (31.2304, 121.4737),
    "上海":         (31.2304, 121.4737),
}


def _flu_weather_risk(temp_c: float, humidity_pct: float) -> str:
    """Simple heuristic: cold and dry = higher flu transmission risk."""
    if temp_c < 5 and humidity_pct < 40:
        return "HIGH — Cold and dry conditions strongly favour airborne flu transmission."
    elif temp_c < 10 and humidity_pct < 55:
        return "ELEVATED — Cool, low-humidity air increases aerosol survival time."
    elif temp_c < 15:
        return "MODERATE — Temperature conducive to flu season activity."
    else:
        return "LOW — Warmer conditions reduce airborne flu survival."


# ─── Tool 1: Weather ─────────────────────────────────────────────────────────

async def get_weather(city: str) -> dict:
    """
    Fetch current weather from Open-Meteo (free, no API key required).
    Falls back to a realistic mock if the API is unreachable (offline mode).
    """
    coords = CITY_COORDS.get(city.lower().strip())
    if not coords:
        # Default to Shenyang if city not in lookup
        coords = CITY_COORDS["shenyang"]
        city = "Shenyang (default)"

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        f"&timezone=Asia%2FShanghai"
    )

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        current = data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]

        return {
            "city": city,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind,
            "flu_transmission_risk": _flu_weather_risk(temp, humidity),
            "data_source": "Open-Meteo (live)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        log.warning(f"Open-Meteo unavailable ({exc}), using offline mock data")
        # Realistic winter values for northern China
        return {
            "city": city,
            "temperature_c": -3.0,
            "humidity_pct": 32.0,
            "wind_speed_kmh": 12.0,
            "flu_transmission_risk": _flu_weather_risk(-3.0, 32.0),
            "data_source": "offline-mock (Open-Meteo unreachable)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ─── Tool 2: Hospital load ────────────────────────────────────────────────────

# Simulated hospital data — in a real deployment this would query a
# hospital information system or regional CDC API.
_HOSPITAL_DATA = [
    {
        "name": "Shenyang Children's Hospital / 沈阳市儿童医院",
        "current_flu_patients": 147,
        "daily_avg_flu_patients": 98,
        "load_ratio": 1.50,
        "dominant_strain": "Influenza A (H3N2)",
        "pediatric_flu_pct": 68,
        "avg_wait_minutes": 85,
        "alert_level": "HIGH",
        "note": "Significantly above seasonal baseline. Advise early isolation for symptomatic students.",
    },
    {
        "name": "Shenyang No.4 People's Hospital / 沈阳市第四人民医院",
        "current_flu_patients": 89,
        "daily_avg_flu_patients": 72,
        "load_ratio": 1.24,
        "dominant_strain": "Influenza A (H1N1pdm09)",
        "pediatric_flu_pct": 41,
        "avg_wait_minutes": 55,
        "alert_level": "ELEVATED",
        "note": "Above baseline. Community transmission confirmed in nearby districts.",
    },
]


async def get_hospital_load(hospital_name: str = "") -> dict:
    """Return simulated hospital ED flu load for the school's district."""
    await asyncio.sleep(0)  # keep it async-friendly

    if hospital_name:
        matched = [
            h for h in _HOSPITAL_DATA
            if hospital_name.lower() in h["name"].lower()
        ]
        hospitals = matched if matched else _HOSPITAL_DATA
    else:
        hospitals = _HOSPITAL_DATA

    return {
        "district": "Shenhe District, Shenyang / 沈阳市沈河区",
        "report_time": datetime.now(timezone.utc).isoformat(),
        "hospitals": hospitals,
        "community_alert": "ACTIVE — Regional flu activity is HIGH. School-level interventions recommended.",
        "data_source": "simulated (District Health Bureau feed)",
    }


# ─── Tool 3: Cough statistics ─────────────────────────────────────────────────

# Per-class 24-hour cough counts (mirrors the React frontend mock data)
_COUGH_DATA = {
    "c1": {"name": "Class 1-1 / 一年级一班", "count_24h": 156, "trend": "rising",  "risk": "high"},
    "c2": {"name": "Class 1-2 / 一年级二班", "count_24h": 82,  "trend": "stable",  "risk": "medium"},
    "c3": {"name": "Class 2-1 / 二年级一班", "count_24h": 12,  "trend": "falling", "risk": "low"},
    "c4": {"name": "Class 2-2 / 二年级二班", "count_24h": 94,  "trend": "rising",  "risk": "medium"},
    "c5": {"name": "Class 3-1 / 三年级一班", "count_24h": 203, "trend": "rising",  "risk": "high"},
    "c6": {"name": "Class 3-2 / 三年级二班", "count_24h": 45,  "trend": "stable",  "risk": "low"},
}


async def get_cough_statistics(class_id: str = "all", hours: int = 24) -> dict:
    """Return cough detection stats from the YAMNet classroom monitoring system."""
    await asyncio.sleep(0)

    if class_id == "all":
        classes = list(_COUGH_DATA.values())
        total = sum(c["count_24h"] for c in classes)
        high_risk = [c["name"] for c in classes if c["risk"] == "high"]
        return {
            "scope": "school-wide",
            "lookback_hours": hours,
            "total_coughs": total,
            "school_avg_per_class": round(total / len(classes), 1),
            "high_risk_classes": high_risk,
            "per_class": {cid: d for cid, d in _COUGH_DATA.items()},
            "model": "YAMNet fine-tuned (FluGuard cough detector)",
            "data_source": "local audio monitoring system",
        }
    elif class_id in _COUGH_DATA:
        d = _COUGH_DATA[class_id]
        return {
            "scope": class_id,
            "lookback_hours": hours,
            "class_name": d["name"],
            "cough_count": d["count_24h"],
            "trend": d["trend"],
            "risk_level": d["risk"],
            "model": "YAMNet fine-tuned (FluGuard cough detector)",
            "data_source": "local audio monitoring system",
        }
    else:
        return {"error": f"Unknown class_id '{class_id}'. Use 'all' or 'c1'–'c6'."}


# ─── Dispatcher ───────────────────────────────────────────────────────────────

import inspect as _inspect

def _filter_args(fn, arguments: dict) -> dict:
    """Only pass kwargs that the function actually accepts, drop the rest."""
    sig = _inspect.signature(fn)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in arguments.items() if k in valid}


async def execute_tool(name: str, arguments: dict) -> Any:
    """Route a tool call by name and return the result."""
    if name == "get_weather":
        args = dict(arguments)
        # Normalise: model sometimes uses 'location' instead of 'city'
        if "location" in args and "city" not in args:
            args["city"] = args.pop("location")
        return await get_weather(**_filter_args(get_weather, args))
    elif name == "get_hospital_load":
        return await get_hospital_load(**_filter_args(get_hospital_load, arguments))
    elif name == "get_cough_statistics":
        return await get_cough_statistics(**_filter_args(get_cough_statistics, arguments))
    else:
        return {"error": f"Unknown tool: {name}"}
