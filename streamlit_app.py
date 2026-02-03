# file: streamlit_app.py
"""
Travel Helper: Packing + Health + Places + Transport + Food
Now with optional real-world POI enrichment via:
- OpenTripMap Places API (geoname, radius, xid details)  https://api.opentripmap.com/0.1/en/places/...  (see example)  :contentReference[oaicite:4]{index=4}
- Google Places API (New) Nearby Search + Text Search via POST + X-Goog-FieldMask headers  :contentReference[oaicite:5]{index=5}

Run:
  pip install streamlit requests
  streamlit run streamlit_app.py

Medical note:
- This tool provides planning prompts, not medical advice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

import requests
import streamlit as st

TripStyle = Literal["Budget", "Mid-range", "Luxury"]
LuggageType = Literal["Backpack", "Carry-on only", "Checked bag"]
WeatherFeel = Literal["Cold", "Mild", "Hot"]
Accommodation = Literal["Hotel", "Hostel", "Airbnb/Apartment", "Resort", "Visiting friends/family", "Other"]

Activity = Literal[
    "City exploring",
    "Business",
    "Beach",
    "Hiking",
    "Ski/Snow",
    "Nightlife",
    "Museums/Art",
    "Food tour",
    "Theme parks",
    "Road trip",
    "Camping",
    "Water sports",
]

Provider = Literal["Offline (no API)", "OpenTripMap", "Google Places (New)"]


@dataclass(frozen=True)
class TravelInput:
    departure: str
    destination: str
    start_date: date
    end_date: date
    travelers: int
    trip_style: TripStyle
    accommodation: Accommodation
    luggage: LuggageType
    weather: WeatherFeel
    rain_likelihood: int  # 0-100
    activities: Tuple[Activity, ...]
    dietary_notes: str
    mobility_notes: str
    health_notes: str
    budget_notes: str


@dataclass
class ChecklistItem:
    item: str
    why: str
    tags: Tuple[str, ...] = ()


@dataclass
class PlaceSuggestion:
    name: str
    address: str = ""
    url: str = ""
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    category: str = ""


@dataclass
class GeneratedPlan:
    packing: Dict[str, List[ChecklistItem]] = field(default_factory=dict)
    health: Dict[str, List[ChecklistItem]] = field(default_factory=dict)
    places: Dict[str, List[str]] = field(default_factory=dict)
    transport: Dict[str, List[str]] = field(default_factory=dict)
    food: Dict[str, List[str]] = field(default_factory=dict)
    enriched: Dict[str, List[PlaceSuggestion]] = field(default_factory=dict)
    reminders: List[str] = field(default_factory=list)


# ----------------------------
# Small internal datasets
# ----------------------------
RIDE_HAILING_BY_REGION: Dict[str, List[str]] = {
    "uk_ie": ["Uber (varies by city)", "Bolt (some cities)", "Free Now (some cities)", "Local licensed minicabs"],
    "eu": ["Bolt (many cities)", "Uber (many cities)", "Free Now (some cities)", "Licensed taxi ranks"],
    "us_canada": ["Uber", "Lyft", "Airport shuttles", "Licensed taxis"],
    "latam": ["Uber (some cities)", "DiDi (some cities)", "Cabify (some cities)", "Use official taxi apps where available"],
    "mena": ["Careem (some cities)", "Uber (some cities)", "Official airport taxis", "Hotel-arranged transfers"],
    "south_asia": ["Uber (some cities)", "Ola (some cities)", "Official prepaid taxi counters (airports)", "Hotel transfers"],
    "se_asia": ["Grab (many countries)", "Gojek (some countries)", "Official airport taxis", "Metered taxis where common"],
    "east_asia": ["Official taxi services", "Public transport apps", "Some cities: Uber (limited)", "Hotel-arranged cars"],
    "oceania": ["Uber", "Local taxi companies", "Airport shuttles", "Public transit cards/passes"],
    "unknown": ["Official airport taxi", "Hotel-arranged transfer", "Licensed taxi ranks", "Reputable local ride-hailing app"],
}

FOOD_STARTERS_BY_COUNTRY: Dict[str, List[str]] = {
    "japan": ["Ramen", "Sushi", "Okonomiyaki", "Tempura", "Kaiseki (if splurging)"],
    "italy": ["Regional pasta specialty", "Pizza (local style)", "Gelato", "Aperitivo snacks", "Espresso + pastry"],
    "mexico": ["Tacos (regional)", "Mole (where common)", "Pozole", "Tamales", "Street elote/esquites"],
    "thailand": ["Pad kra pao", "Som tam", "Tom yum", "Khao soi (north)", "Mango sticky rice"],
    "france": ["Boulangerie bread/pastries", "Cheese plate", "Regional stew/specialty", "Crêpes (if common)", "Local wine (optional)"],
    "india": ["Regional thali", "Chaat", "Dosa (south)", "Biryani (where famous)", "Masala chai"],
    "spain": ["Tapas crawl", "Paella (where typical)", "Jamón", "Tortilla española", "Churros con chocolate"],
    "vietnam": ["Phở", "Bánh mì", "Bún chả", "Gỏi cuốn (fresh rolls)", "Cà phê sữa đá"],
    "greece": ["Souvlaki/gyros", "Greek salad", "Seafood (coast/islands)", "Moussaka", "Baklava"],
}


# ----------------------------
# Utility
# ----------------------------
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def trip_length_days(start: date, end: date) -> int:
    delta = (end - start).days
    return max(1, delta + 1)


def uniq_items(items: List[ChecklistItem]) -> List[ChecklistItem]:
    seen: Set[str] = set()
    out: List[ChecklistItem] = []
    for it in items:
        key = normalize_text(it.item)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def infer_region(destination: str) -> str:
    d = normalize_text(destination)
    if any(k in d for k in ["uk", "united kingdom", "england", "scotland", "wales", "northern ireland", "ireland", "dublin", "london"]):
        return "uk_ie"
    if any(k in d for k in ["france", "germany", "italy", "spain", "portugal", "netherlands", "belgium", "austria", "switzerland", "sweden", "norway", "denmark", "finland", "poland", "czech", "hungary", "greece", "croatia", "romania"]):
        return "eu"
    if any(k in d for k in ["united states", "usa", "new york", "los angeles", "san francisco", "canada", "toronto", "vancouver", "montreal"]):
        return "us_canada"
    if any(k in d for k in ["mexico", "brazil", "argentina", "chile", "colombia", "peru", "costa rica"]):
        return "latam"
    if any(k in d for k in ["uae", "dubai", "abu dhabi", "qatar", "doha", "saudi", "riyadh", "jeddah", "egypt", "cairo", "morocco", "marrakesh"]):
        return "mena"
    if any(k in d for k in ["india", "delhi", "mumbai", "bangalore", "pakistan", "lahore", "karachi", "bangladesh", "dhaka", "nepal", "kathmandu", "sri lanka", "colombo"]):
        return "south_asia"
    if any(k in d for k in ["thailand", "bangkok", "vietnam", "hanoi", "ho chi minh", "philippines", "manila", "indonesia", "jakarta", "bali", "malaysia", "kuala lumpur", "singapore"]):
        return "se_asia"
    if any(k in d for k in ["japan", "tokyo", "osaka", "kyoto", "china", "beijing", "shanghai", "hong kong", "taiwan", "taipei", "korea", "seoul"]):
        return "east_asia"
    if any(k in d for k in ["australia", "sydney", "melbourne", "new zealand", "auckland", "wellington"]):
        return "oceania"
    return "unknown"


def extract_country_key(destination: str) -> Optional[str]:
    d = normalize_text(destination)
    for country in FOOD_STARTERS_BY_COUNTRY.keys():
        if country in d:
            return country
    return None


def _safe_get_text_field(v) -> str:
    # Google Places sometimes returns displayName as {"text": "..."}.
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if isinstance(v.get("text"), str):
            return v["text"]
    return ""


# ----------------------------
# Core generators (offline baseline)
# ----------------------------
def base_packing() -> Dict[str, List[ChecklistItem]]:
    return {
        "Documents & money": [
            ChecklistItem("Passport/ID", "Core ID for travel, hotels, and emergencies", ("docs",)),
            ChecklistItem("Travel insurance details", "Helps with medical issues, delays, and lost items", ("docs", "health")),
            ChecklistItem("Payment cards + some cash", "Backup when terminals fail or tips are cash-based", ("money",)),
            ChecklistItem("Copies of key docs (digital + paper)", "Recovery if originals are lost", ("docs",)),
        ],
        "Tech": [
            ChecklistItem("Phone + charger", "Navigation, tickets, communication", ("tech",)),
            ChecklistItem("Power adapter (if needed)", "Sockets differ by country/region", ("tech",)),
            ChecklistItem("Power bank", "Long days out; helps with maps and photos", ("tech",)),
            ChecklistItem("Headphones", "Flights, commutes, calls", ("tech",)),
        ],
        "Toiletries": [
            ChecklistItem("Toothbrush/toothpaste", "Basics", ("toiletries",)),
            ChecklistItem("Deodorant", "Basics", ("toiletries",)),
            ChecklistItem("Sunscreen", "Sun exposure even in cities", ("toiletries", "health")),
            ChecklistItem("Hand sanitizer", "Useful in transit", ("toiletries", "health")),
        ],
        "Clothing (base)": [
            ChecklistItem("Underwear/socks", "Comfort and hygiene", ("clothes",)),
            ChecklistItem("Everyday outfit(s)", "Mix-and-match layers", ("clothes",)),
            ChecklistItem("Sleepwear", "Comfort", ("clothes",)),
        ],
        "Safety & misc": [
            ChecklistItem("Reusable water bottle", "Hydration + savings", ("misc", "health")),
            ChecklistItem("Small day bag", "Day trips, museums, markets", ("misc",)),
            ChecklistItem("Small lock (optional)", "Hostels/shared storage", ("safety",)),
        ],
    }


def weather_module(weather: WeatherFeel, rain: int) -> Dict[str, List[ChecklistItem]]:
    out: Dict[str, List[ChecklistItem]] = {"Weather add-ons": []}
    if weather == "Cold":
        out["Weather add-ons"] += [
            ChecklistItem("Warm jacket", "Core warmth layer", ("weather", "cold")),
            ChecklistItem("Thermal base layer", "Warmth without bulk", ("weather", "cold")),
            ChecklistItem("Gloves + beanie", "Extremities lose heat fast", ("weather", "cold")),
        ]
    elif weather == "Hot":
        out["Weather add-ons"] += [
            ChecklistItem("Breathable tops", "Heat comfort", ("weather", "hot")),
            ChecklistItem("Hat/cap", "Sun protection", ("weather", "hot")),
            ChecklistItem("Lightweight sandals (optional)", "Heat-friendly footwear", ("weather", "hot")),
        ]
    else:
        out["Weather add-ons"] += [
            ChecklistItem("Light jacket", "Evenings can be cooler", ("weather", "mild")),
            ChecklistItem("Layering top", "Flexible comfort", ("weather", "mild")),
        ]

    if rain >= 60:
        out["Weather add-ons"] += [
            ChecklistItem("Compact umbrella", "Quick rain coverage", ("weather", "rain")),
            ChecklistItem("Light rain jacket", "Hands-free rain protection", ("weather", "rain")),
            ChecklistItem("Water-resistant shoes (optional)", "Avoid soaked feet on long days", ("weather", "rain")),
        ]
    return out


def activity_modules(activities: Tuple[Activity, ...], trip_days: int, trip_style: TripStyle) -> Dict[str, List[ChecklistItem]]:
    out: Dict[str, List[ChecklistItem]] = {"Activity add-ons": []}

    if "Business" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("Business outfit", "Meetings/dinners", ("activity", "business")),
            ChecklistItem("Portable steamer (optional)", "Keep clothes crisp if you care", ("activity", "business")),
        ]
    if "Hiking" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("Comfortable walking/hiking shoes", "Injury prevention + comfort", ("activity", "hiking")),
            ChecklistItem("Lightweight rain/wind layer", "Weather changes fast outdoors", ("activity", "hiking")),
            ChecklistItem("Blister care (plasters/moleskin)", "Stops small pain becoming a problem", ("activity", "hiking", "health")),
        ]
    if "Beach" in activities or "Water sports" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("Swimwear", "Beach/pool", ("activity", "beach")),
            ChecklistItem("Quick-dry towel (optional)", "Convenient on day trips", ("activity", "beach")),
            ChecklistItem("Waterproof phone pouch (optional)", "Protects phone near water", ("activity", "beach", "tech")),
        ]
    if "Ski/Snow" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("Ski socks", "Warmth + fit", ("activity", "snow")),
            ChecklistItem("Neck gaiter/buff", "Wind protection", ("activity", "snow")),
            ChecklistItem("Goggles (if not renting)", "Eye protection in snow glare", ("activity", "snow")),
        ]
    if "Nightlife" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("One nicer outfit", "Dress codes vary", ("activity", "nightlife")),
            ChecklistItem("Small crossbody/secure wallet", "Crowded areas", ("activity", "nightlife", "safety")),
        ]
    if "Road trip" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("Phone mount (optional)", "Safer navigation", ("activity", "roadtrip")),
            ChecklistItem("Offline maps downloaded", "Coverage gaps happen", ("activity", "roadtrip", "tech")),
        ]
    if "Camping" in activities:
        out["Activity add-ons"] += [
            ChecklistItem("Headlamp", "Hands-free light", ("activity", "camping")),
            ChecklistItem("Light first-aid kit", "Remote areas", ("activity", "camping", "health")),
        ]

    if trip_days >= 7 and trip_style != "Luxury":
        out["Activity add-ons"].append(ChecklistItem("Laundry kit (small detergent sheets)", "Light packing for longer trips", ("misc",)))

    return out


def health_checklist(region: str) -> Dict[str, List[ChecklistItem]]:
    common = [
        ChecklistItem("Check official travel health advice for your destination", "Guidance changes; use official sources", ("health",)),
        ChecklistItem("Confirm routine vaccines are up to date", "Baseline protection", ("health",)),
        ChecklistItem("Carry personal meds in original packaging", "Helps at borders and in emergencies", ("health",)),
        ChecklistItem("Consider a basic first-aid kit", "Blisters, minor cuts, headaches", ("health",)),
        ChecklistItem("Verify if proof of vaccination is required for entry/transit", "Some routes have requirements", ("health", "docs")),
    ]
    region_prompts: Dict[str, List[ChecklistItem]] = {
        "se_asia": [
            ChecklistItem("Ask a clinician about mosquito-borne illness prevention", "Repellent + behavior planning", ("health",)),
            ChecklistItem("Food/water hygiene plan", "Reduce stomach issues", ("health",)),
        ],
        "south_asia": [
            ChecklistItem("Ask a clinician about stomach illness prevention", "Hygiene and contingency meds", ("health",)),
            ChecklistItem("Heat and hydration strategy", "High temps can be risky", ("health",)),
        ],
        "latam": [
            ChecklistItem("Ask a clinician about mosquito-borne illness prevention", "Repellent + clothing", ("health",)),
            ChecklistItem("Altitude planning (if relevant)", "Some areas require acclimatization", ("health",)),
        ],
        "mena": [ChecklistItem("Heat and sun plan", "Hydration + shade + sunscreen", ("health",))],
        "unknown": [ChecklistItem("If unsure, consult a travel clinic 4–8 weeks before travel", "Some vaccines need time/boosters", ("health",))],
    }
    return {
        "Health & vaccines (checklist)": common,
        "Destination prompts (verify with clinician)": region_prompts.get(region, []),
    }


def places_to_visit(activities: Tuple[Activity, ...], destination: str) -> Dict[str, List[str]]:
    d = destination.strip()
    picks: Dict[str, List[str]] = {
        "Ideas based on your interests": [],
        "Easy wins anywhere": [
            "Do a walking tour on day 1 (fast orientation).",
            "Pick one neighborhood to wander with no agenda.",
            "Bookmark 2–3 indoor options for bad weather.",
        ],
        "Destination prompts": [
            f"Search: “best neighborhoods in {d}” and save 2–3 to explore.",
            f"Search: “day trips from {d}” and pick one that matches your pace.",
            f"Search: “local events in {d} during your dates”.",
        ],
    }

    if "Museums/Art" in activities:
        picks["Ideas based on your interests"] += ["One flagship museum + one small gallery.", "Check late-night openings/free entry windows."]
    if "Food tour" in activities:
        picks["Ideas based on your interests"] += ["Market visit early in the trip.", "Street-food area with high turnover + visible cooking."]
    if "Hiking" in activities:
        picks["Ideas based on your interests"] += ["Half-day hike first; then full-day route.", "Download offline trail maps; check daylight hours."]
    if "Beach" in activities:
        picks["Ideas based on your interests"] += ["One calm beach (morning) + one lively beach (afternoon).", "Pick a sunset spot."]
    if "Theme parks" in activities:
        picks["Ideas based on your interests"] += ["Buy timed-entry tickets early if needed.", "Arrive before opening for first rides."]
    if not picks["Ideas based on your interests"]:
        picks["Ideas based on your interests"] = ["Each day: 1 landmark, 1 local experience, 1 nature break (park/river)."]
    return picks


def transport_guide(region: str, destination: str) -> Dict[str, List[str]]:
    region_apps = RIDE_HAILING_BY_REGION.get(region, RIDE_HAILING_BY_REGION["unknown"])
    d = destination.strip()
    safety = [
        "Prefer official taxi ranks or app-dispatched rides.",
        "If street taxis: confirm meter or agree price before starting.",
        "Share trip details; sit in back if solo.",
        "At airports: use official/prepaid counters or hotel transfers.",
    ]
    return {
        "Taxi / ride options": region_apps,
        "Safety checklist": safety,
        "Destination prompts": [
            f"Search: “official taxi number in {d}” and save it.",
            f"Search: “airport to city center transport {d}” (compare train/bus/taxi).",
            "Download the local public transit app.",
        ],
    }


def food_guide(destination: str, dietary_notes: str) -> Dict[str, List[str]]:
    key = extract_country_key(destination)
    starter = FOOD_STARTERS_BY_COUNTRY.get(key or "", [])
    prompts = [
        "Ask locals: “What’s the one dish this city does best?”",
        "Try: one market meal, one street snack, one sit-down specialty.",
    ]
    if dietary_notes.strip():
        prompts.append(f"Diet note: {dietary_notes.strip()}")

    if starter:
        return {"Local foods (starter list)": starter, "Food game plan": prompts}

    return {
        "Local foods (starter list)": [
            "Signature stew/soup of the region",
            "Famous street-food item",
            "Local dessert/pastry",
            "Common breakfast item",
            "Seasonal specialty (ask what’s best right now)",
        ],
        "Food game plan": prompts,
    }


def generate_plan(inp: TravelInput) -> GeneratedPlan:
    region = infer_region(inp.destination)
    days = trip_length_days(inp.start_date, inp.end_date)

    packing = base_packing()
    for k, v in weather_module(inp.weather, inp.rain_likelihood).items():
        packing.setdefault(k, []).extend(v)
    for k, v in activity_modules(inp.activities, days, inp.trip_style).items():
        packing.setdefault(k, []).extend(v)

    if inp.luggage == "Carry-on only":
        packing.setdefault("Carry-on strategy", []).extend(
            [
                ChecklistItem("Solid toiletries (or <100ml liquids)", "Avoid liquid limits issues", ("luggage",)),
                ChecklistItem("Wear bulkiest shoes on travel day", "Saves bag space", ("luggage",)),
                ChecklistItem("One versatile jacket", "Reduces overpacking", ("luggage",)),
            ]
        )

    if inp.mobility_notes.strip():
        packing.setdefault("Accessibility", []).append(
            ChecklistItem("Any mobility aids / supports you rely on", "Consistency and comfort", ("accessibility",))
        )

    for cat in list(packing.keys()):
        packing[cat] = uniq_items(packing[cat])

    return GeneratedPlan(
        packing=packing,
        health=health_checklist(region),
        places=places_to_visit(inp.activities, inp.destination),
        transport=transport_guide(region, inp.destination),
        food=food_guide(inp.destination, inp.dietary_notes),
        reminders=[
            "Download offline maps + save key addresses (hotel, embassy, venues).",
            "Set up roaming/eSIM plan before departure.",
            "Enable contactless payments; consider notifying your bank.",
            "Save local emergency number + key contacts.",
        ],
    )


# ----------------------------
# API Enrichment (optional)
# ----------------------------
GOOGLE_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"

OTM_BASE = "https://api.opentripmap.com/0.1/en/places"


def _http_json(method: str, url: str, *, headers: Optional[dict] = None, params: Optional[dict] = None, body: Optional[dict] = None) -> dict:
    try:
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=20)
        else:
            r = requests.post(url, headers=headers, json=body, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP error calling {url}: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Non-JSON response from {url}") from e


@st.cache_data(show_spinner=False, ttl=3600)
def google_text_search(api_key: str, text_query: str, field_mask: str) -> List[dict]:
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": field_mask}
    data = _http_json("POST", GOOGLE_TEXT_SEARCH_URL, headers=headers, body={"textQuery": text_query})
    return list(data.get("places", []) or [])


@st.cache_data(show_spinner=False, ttl=3600)
def google_nearby_search(api_key: str, lat: float, lon: float, radius_m: int, included_types: Sequence[str], max_results: int, field_mask: str) -> List[dict]:
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": field_mask}
    body = {
        "includedTypes": list(included_types),
        "maxResultCount": int(max_results),
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": float(radius_m)}},
    }
    data = _http_json("POST", GOOGLE_NEARBY_SEARCH_URL, headers=headers, body=body)
    return list(data.get("places", []) or [])


@st.cache_data(show_spinner=False, ttl=3600)
def otm_geoname(api_key: str, name: str) -> dict:
    # Example uses: /geoname?name=...&apikey=... :contentReference[oaicite:6]{index=6}
    return _http_json("GET", f"{OTM_BASE}/geoname", params={"name": name, "apikey": api_key})


@st.cache_data(show_spinner=False, ttl=3600)
def otm_radius(api_key: str, lat: float, lon: float, radius_m: int, kinds: str, limit: int) -> List[dict]:
    # Example uses: /radius?radius=1000&limit=...&lon=...&lat=...&rate=2&format=json :contentReference[oaicite:7]{index=7}
    params = {
        "radius": int(radius_m),
        "limit": int(limit),
        "offset": 0,
        "lon": float(lon),
        "lat": float(lat),
        "rate": 2,
        "format": "json",
        "apikey": api_key,
    }
    if kinds:
        params["kinds"] = kinds
    data = _http_json("GET", f"{OTM_BASE}/radius", params=params)
    return list(data or [])


@st.cache_data(show_spinner=False, ttl=3600)
def otm_xid(api_key: str, xid: str) -> dict:
    # Example uses: /xid/{xid}?apikey=... :contentReference[oaicite:8]{index=8}
    return _http_json("GET", f"{OTM_BASE}/xid/{xid}", params={"apikey": api_key})


def _suggestion_from_google_place(p: dict, category: str) -> PlaceSuggestion:
    return PlaceSuggestion(
        name=_safe_get_text_field(p.get("displayName")) or _safe_get_text_field(p.get("name")) or "Unknown",
        address=str(p.get("formattedAddress") or ""),
        url=str(p.get("googleMapsUri") or ""),
        rating=(float(p["rating"]) if isinstance(p.get("rating"), (int, float)) else None),
        rating_count=(int(p["userRatingCount"]) if isinstance(p.get("userRatingCount"), int) else None),
        category=category,
    )


def _suggestion_from_otm_details(d: dict, fallback_name: str, category: str) -> PlaceSuggestion:
    name = str(d.get("name") or fallback_name or "Unknown")
    url = str(d.get("otm") or d.get("wikipedia") or "")
    address = ""
    if isinstance(d.get("address"), dict):
        a = d["address"]
        address = ", ".join([str(x) for x in [a.get("road"), a.get("city"), a.get("state"), a.get("country")] if x])
    return PlaceSuggestion(name=name, address=address, url=url, category=category)


def build_interest_queries(inp: TravelInput) -> Dict[str, dict]:
    """
    Returns mapping:
      key -> {"google_nearby_types": [...], "google_text": "...", "otm_kinds": "..."}
    """
    d = inp.destination.strip()
    m: Dict[str, dict] = {}

    m["Top attractions"] = {
        "google_nearby_types": ["tourist_attraction", "park"],
        "google_text": f"top attractions in {d}",
        "otm_kinds": "interesting_places,architecture,cultural,historic",
    }
    if "Museums/Art" in inp.activities:
        m["Museums & art"] = {
            "google_nearby_types": ["museum", "art_gallery"],
            "google_text": f"best museums in {d}",
            "otm_kinds": "museums,cultural",
        }
    if "Nightlife" in inp.activities:
        m["Nightlife"] = {
            "google_nearby_types": ["bar", "night_club"],
            "google_text": f"best nightlife in {d}",
            "otm_kinds": "nightlife,foods",
        }
    if "Food tour" in inp.activities:
        m["Food spots"] = {
            "google_nearby_types": ["restaurant"],
            "google_text": f"best local restaurants in {d}",
            "otm_kinds": "foods",
        }
    if "Beach" in inp.activities:
        m["Beaches"] = {
            "google_nearby_types": [],
            "google_text": f"best beaches near {d}",
            "otm_kinds": "natural,beaches",
        }
    if "Hiking" in inp.activities:
        m["Hikes & nature"] = {
            "google_nearby_types": ["park"],
            "google_text": f"best hikes near {d}",
            "otm_kinds": "natural",
        }

    m["Taxi services"] = {
        "google_nearby_types": [],  # text search works better
        "google_text": f"taxi service in {d}",
        "otm_kinds": "",
    }

    return m


def enrich_plan_with_api(
    plan: GeneratedPlan,
    inp: TravelInput,
    provider: Provider,
    api_key: str,
    radius_km: int,
    max_results: int,
    cost_saver: bool,
) -> GeneratedPlan:
    if provider == "Offline (no API)":
        return plan

    radius_m = clamp(radius_km, 1, 50) * 1000
    max_results = clamp(max_results, 3, 20)

    enriched: Dict[str, List[PlaceSuggestion]] = {}
    queries = build_interest_queries(inp)
    d = inp.destination.strip()

    if provider == "Google Places (New)":
        if not api_key.strip():
            raise RuntimeError("Google Places API key is required.")
        field_mask = "places.displayName,places.formattedAddress,places.googleMapsUri"
        if not cost_saver:
            field_mask = "places.displayName,places.formattedAddress,places.googleMapsUri,places.rating,places.userRatingCount"

        # Get a lat/lon by Text Search on the destination itself.
        loc_field_mask = "places.displayName,places.formattedAddress,places.location"
        loc_places = google_text_search(api_key, d, loc_field_mask)
        if not loc_places:
            raise RuntimeError("Google Places Text Search returned no results for destination; try a more specific destination string.")
        loc = loc_places[0].get("location") or {}
        lat = float(loc.get("latitude"))
        lon = float(loc.get("longitude"))

        for section, spec in queries.items():
            suggestions: List[PlaceSuggestion] = []
            nearby_types = list(spec.get("google_nearby_types") or [])
            text_q = str(spec.get("google_text") or "").strip()

            if nearby_types:
                nearby = google_nearby_search(api_key, lat, lon, radius_m, nearby_types, max_results, field_mask)
                suggestions.extend([_suggestion_from_google_place(p, section) for p in nearby])

            # Always add a text search fallback; helps for "Beaches", "Taxi services", etc.
            if text_q:
                text = google_text_search(api_key, text_q, field_mask)
                suggestions.extend([_suggestion_from_google_place(p, section) for p in text[:max_results]])

            enriched[section] = _dedupe_suggestions(suggestions)[:max_results]

    elif provider == "OpenTripMap":
        if not api_key.strip():
            raise RuntimeError("OpenTripMap API key is required.")
        geo = otm_geoname(api_key, d)
        if str(geo.get("status")) != "OK":
            raise RuntimeError("OpenTripMap geoname failed; try a more specific destination string (e.g., 'Paris, France').")
        lat = float(geo.get("lat"))
        lon = float(geo.get("lon"))

        for section, spec in queries.items():
            kinds = str(spec.get("otm_kinds") or "")
            if section == "Taxi services":
                # OTM isn't strong for taxi companies; keep it offline.
                enriched[section] = []
                continue

            hits = otm_radius(api_key, lat, lon, radius_m, kinds=kinds, limit=max_results)
            # Fetch details for top N for better URLs/descriptions.
            suggestions: List[PlaceSuggestion] = []
            for h in hits[:max_results]:
                xid = str(h.get("xid") or "")
                fallback_name = str(h.get("name") or "")
                if not xid:
                    continue
                details = otm_xid(api_key, xid)
                suggestions.append(_suggestion_from_otm_details(details, fallback_name, section))
            enriched[section] = _dedupe_suggestions(suggestions)[:max_results]

    else:
        return plan

    plan.enriched = enriched
    return plan


def _dedupe_suggestions(items: List[PlaceSuggestion]) -> List[PlaceSuggestion]:
    seen: Set[str] = set()
    out: List[PlaceSuggestion] = []
    for it in items:
        key = normalize_text(f"{it.category}::{it.name}::{it.address}")
        if key not in seen and it.name.strip():
            seen.add(key)
            out.append(it)
    return out


# ----------------------------
# Export
# ----------------------------
def _plan_to_jsonable(plan: GeneratedPlan) -> dict:
    def item_to_dict(it: ChecklistItem) -> dict:
        return {"item": it.item, "why": it.why, "tags": list(it.tags)}

    def place_to_dict(p: PlaceSuggestion) -> dict:
        return {
            "name": p.name,
            "address": p.address,
            "url": p.url,
            "rating": p.rating,
            "rating_count": p.rating_count,
            "category": p.category,
        }

    return {
        "packing": {k: [item_to_dict(i) for i in v] for k, v in plan.packing.items()},
        "health": {k: [item_to_dict(i) for i in v] for k, v in plan.health.items()},
        "places": plan.places,
        "transport": plan.transport,
        "food": plan.food,
        "enriched": {k: [place_to_dict(p) for p in v] for k, v in plan.enriched.items()},
        "reminders": plan.reminders,
    }


def plan_to_markdown(inp: TravelInput, plan: GeneratedPlan) -> str:
    lines: List[str] = []
    lines.append(f"# Travel Plan: {inp.departure} → {inp.destination}")
    lines.append("")
    lines.append(f"- Dates: {inp.start_date.isoformat()} to {inp.end_date.isoformat()}")
    lines.append(f"- Travelers: {inp.travelers}")
    lines.append(f"- Style: {inp.trip_style} | Accommodation: {inp.accommodation} | Luggage: {inp.luggage}")
    lines.append(f"- Weather: {inp.weather} | Rain likelihood: {inp.rain_likelihood}%")
    lines.append(f"- Activities: {', '.join(inp.activities) if inp.activities else 'None selected'}")
    if inp.dietary_notes.strip():
        lines.append(f"- Dietary notes: {inp.dietary_notes.strip()}")
    if inp.health_notes.strip():
        lines.append(f"- Health notes: {inp.health_notes.strip()}")
    lines.append("")

    if plan.enriched:
        lines.append("## Quick picks (from API)")
        for section, places in plan.enriched.items():
            if not places:
                continue
            lines.append(f"### {section}")
            for p in places:
                meta = []
                if p.address:
                    meta.append(p.address)
                if p.rating is not None and p.rating_count is not None:
                    meta.append(f"⭐ {p.rating} ({p.rating_count})")
                tail = " — ".join(meta) if meta else ""
                if p.url:
                    lines.append(f"- [{p.name}]({p.url}){(' — ' + tail) if tail else ''}")
                else:
                    lines.append(f"- {p.name}{(' — ' + tail) if tail else ''}")
            lines.append("")

    lines.append("## Packing List")
    for cat, items in plan.packing.items():
        lines.append(f"### {cat}")
        for it in items:
            lines.append(f"- [ ] **{it.item}** — {it.why}")
        lines.append("")

    lines.append("## Health & Vaccines (Verify)")
    for cat, items in plan.health.items():
        lines.append(f"### {cat}")
        for it in items:
            lines.append(f"- [ ] **{it.item}** — {it.why}")
        lines.append("")

    lines.append("## Places to Visit (Prompts)")
    for cat, items in plan.places.items():
        lines.append(f"### {cat}")
        for s in items:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## Transport / Taxi (Prompts)")
    for cat, items in plan.transport.items():
        lines.append(f"### {cat}")
        for s in items:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## Food Not to Miss")
    for cat, items in plan.food.items():
        lines.append(f"### {cat}")
        for s in items:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## Reminders")
    for r in plan.reminders:
        lines.append(f"- [ ] {r}")
    lines.append("")

    return "\n".join(lines)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Travel Helper", page_icon="🧳", layout="wide")
st.title("🧳 Travel Helper")

with st.sidebar:
    st.header("API enrichment")
    provider: Provider = st.selectbox("Provider", ["Offline (no API)", "OpenTripMap", "Google Places (New)"], index=0)
    api_key = st.text_input("API key", type="password", help="Optional if Offline. Required for OpenTripMap / Google Places.")
    radius_km = st.slider("Search radius (km)", 1, 25, 5)
    max_results = st.slider("Max results per section", 3, 20, 8)
    cost_saver = st.toggle("Cost-saver mode (Google: fewer fields)", value=True)
    st.caption("Tip: Start Offline. Switch to an API when you want real POIs.")

with st.expander("What this is (and isn’t)"):
    st.write("Generates planning prompts. Health/vaccine items must be verified with official sources + a clinician.")

col1, col2, col3 = st.columns(3)
with col1:
    departure = st.text_input("Departure (city/country)", placeholder="e.g., London, UK")
    travelers = st.number_input("Travelers", min_value=1, max_value=20, value=1, step=1)
    trip_style = st.selectbox("Trip style", ["Budget", "Mid-range", "Luxury"], index=1)
    luggage = st.selectbox("Luggage", ["Backpack", "Carry-on only", "Checked bag"], index=1)

with col2:
    destination = st.text_input("Destination (city/country)", placeholder="e.g., Tokyo, Japan")
    start_date = st.date_input("Start date", value=date.today())
    end_date = st.date_input("End date", value=date.today())
    accommodation = st.selectbox(
        "Accommodation", ["Hotel", "Hostel", "Airbnb/Apartment", "Resort", "Visiting friends/family", "Other"], index=0
    )

with col3:
    weather = st.selectbox("Weather expectation", ["Cold", "Mild", "Hot"], index=1)
    rain_likelihood = st.slider("Rain likelihood (%)", 0, 100, 30, 5)
    activities = st.multiselect(
        "Activities",
        [
            "City exploring",
            "Business",
            "Beach",
            "Hiking",
            "Ski/Snow",
            "Nightlife",
            "Museums/Art",
            "Food tour",
            "Theme parks",
            "Road trip",
            "Camping",
            "Water sports",
        ],
        default=["City exploring"],
    )

dietary_notes = st.text_input("Dietary notes (optional)", placeholder="e.g., vegetarian, gluten-free")
mobility_notes = st.text_input("Mobility/accessibility notes (optional)", placeholder="e.g., avoid stairs, knee support")
health_notes = st.text_input("Health notes (optional)", placeholder="e.g., asthma meds, allergies")
budget_notes = st.text_input("Budget notes (optional)", placeholder="e.g., prefer free attractions, mid-price restaurants")

generate = st.button("Generate plan", type="primary")

if generate:
    if not departure.strip() or not destination.strip():
        st.error("Please enter both departure and destination.")
        st.stop()
    if end_date < start_date:
        st.error("End date must be on/after start date.")
        st.stop()

    inp = TravelInput(
        departure=departure.strip(),
        destination=destination.strip(),
        start_date=start_date,
        end_date=end_date,
        travelers=int(travelers),
        trip_style=trip_style,  # type: ignore[assignment]
        accommodation=accommodation,  # type: ignore[assignment]
        luggage=luggage,  # type: ignore[assignment]
        weather=weather,  # type: ignore[assignment]
        rain_likelihood=int(rain_likelihood),
        activities=tuple(activities),  # type: ignore[arg-type]
        dietary_notes=dietary_notes.strip(),
        mobility_notes=mobility_notes.strip(),
        health_notes=health_notes.strip(),
        budget_notes=budget_notes.strip(),
    )

    plan = generate_plan(inp)

    with st.spinner("Enriching with real places (if enabled)..."):
        try:
            plan = enrich_plan_with_api(
                plan=plan,
                inp=inp,
                provider=provider,
                api_key=api_key,
                radius_km=int(radius_km),
                max_results=int(max_results),
                cost_saver=bool(cost_saver),
            )
        except Exception as e:
            st.warning(str(e))

    tab_picks, tab_pack, tab_health, tab_places, tab_transport, tab_food, tab_export = st.tabs(
        ["Quick picks", "Packing", "Health", "Places", "Transport", "Food", "Export"]
    )

    with tab_picks:
        st.subheader("Quick picks (API results)")
        if not plan.enriched:
            st.info("No API enrichment (or provider set to Offline).")
        else:
            for section, items in plan.enriched.items():
                if not items:
                    continue
                st.markdown(f"### {section}")
                for p in items:
                    meta = []
                    if p.address:
                        meta.append(p.address)
                    if p.rating is not None and p.rating_count is not None:
                        meta.append(f"⭐ {p.rating} ({p.rating_count})")
                    tail = " — ".join(meta)
                    if p.url:
                        st.markdown(f"- [{p.name}]({p.url}){(' — ' + tail) if tail else ''}")
                    else:
                        st.write(f"- {p.name}{(' — ' + tail) if tail else ''}")

    with tab_pack:
        st.subheader("Packing list")
        for cat, items in plan.packing.items():
            st.markdown(f"### {cat}")
            for it in items:
                st.checkbox(f"{it.item} — {it.why}", value=False, key=f"pack::{cat}::{it.item}")

    with tab_health:
        st.subheader("Health & vaccines (verify)")
        st.info("Not medical advice. Verify requirements with official sources and a clinician.")
        for cat, items in plan.health.items():
            st.markdown(f"### {cat}")
            for it in items:
                st.checkbox(f"{it.item} — {it.why}", value=False, key=f"health::{cat}::{it.item}")

    with tab_places:
        st.subheader("Places prompts (offline)")
        for cat, items in plan.places.items():
            st.markdown(f"### {cat}")
            for s in items:
                st.write(f"- {s}")

    with tab_transport:
        st.subheader("Transport / taxi prompts")
        for cat, items in plan.transport.items():
            st.markdown(f"### {cat}")
            for s in items:
                st.write(f"- {s}")

        if plan.enriched.get("Taxi services"):
            st.markdown("### Taxi services (API)")
            for p in plan.enriched["Taxi services"]:
                if p.url:
                    st.markdown(f"- [{p.name}]({p.url}){(' — ' + p.address) if p.address else ''}")
                else:
                    st.write(f"- {p.name}{(' — ' + p.address) if p.address else ''}")

    with tab_food:
        st.subheader("Local food not to miss")
        for cat, items in plan.food.items():
            st.markdown(f"### {cat}")
            for s in items:
                st.write(f"- {s}")

        if plan.enriched.get("Food spots"):
            st.markdown("### Food spots (API)")
            for p in plan.enriched["Food spots"]:
                meta = []
                if p.address:
                    meta.append(p.address)
                if p.rating is not None and p.rating_count is not None:
                    meta.append(f"⭐ {p.rating} ({p.rating_count})")
                tail = " — ".join(meta)
                if p.url:
                    st.markdown(f"- [{p.name}]({p.url}){(' — ' + tail) if tail else ''}")
                else:
                    st.write(f"- {p.name}{(' — ' + tail) if tail else ''}")

        st.markdown("### Customize your food list")
        custom_food = st.text_area("Add your own items (one per line)", value="", placeholder="Dish name\nAnother dish\nA dessert")
        if custom_food.strip():
            added = [line.strip() for line in custom_food.splitlines() if line.strip()]
            if added:
                st.success(f"Added {len(added)} custom items (export will include them).")
                plan.food.setdefault("Your custom additions", []).extend(added)

    with tab_export:
        st.subheader("Export")
        md = plan_to_markdown(inp, plan)

        st.download_button("Download Markdown", data=md.encode("utf-8"), file_name="travel_plan.md", mime="text/markdown")
        json_blob = json.dumps({"input": asdict(inp), "plan": _plan_to_jsonable(plan)}, indent=2, ensure_ascii=False)
        st.download_button("Download JSON", data=json_blob.encode("utf-8"), file_name="travel_plan.json", mime="application/json")

        st.markdown("### Preview (Markdown)")
        st.code(md, language="markdown")
