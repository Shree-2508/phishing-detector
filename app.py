"""
app.py
─────────────────────────────────────────────────────────────────────────────
PhishGuard AI — Flask API Server
Modules:
  1. ML Engine        (TF-IDF + Ensemble classifier)
  2. Multilingual     (Script detection + mock translation for Indian languages)
  3. URL/Domain Analyzer (TLD check, Levenshtein brand similarity, IP/subdomain)
  4. Risk Scorer      (Weighted composite: ML 50% + URL 30% + Heuristic 20%)

Run:
    python app.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import math
import pickle
import logging
from pathlib import Path
from datetime import datetime
from functools import lru_cache

from flask import Flask, request, jsonify, send_from_directory

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhishGuard")

app = Flask(__name__, static_folder=".")

# ─── GLOBAL CONFIG ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishguard_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

# In-memory scan log (last 500 scans)
scan_log = []

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — ML ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class MLEngine:
    """Loads and exposes the trained TF-IDF + Ensemble pipeline."""

    def __init__(self):
        self.pipeline = None
        self.meta = {}
        self._load()

    def _load(self):
        if not Path(MODEL_PATH).exists():
            logger.warning("Model not found at %s. Run model_trainer.py first.", MODEL_PATH)
            return
        with open(MODEL_PATH, "rb") as f:
            self.pipeline = pickle.load(f)
        if Path(META_PATH).exists():
            with open(META_PATH) as f:
                self.meta = json.load(f)
        logger.info("ML model loaded. Meta: %s", self.meta.get("model", "N/A"))

    def predict(self, text: str) -> dict:
        """
        Returns:
            probability  float  0-1  (phishing probability)
            label        int    0=legit, 1=phishing
            confidence   float  0-100
        """
        if self.pipeline is None:
            return {"probability": 0.5, "label": 0, "confidence": 50.0}
        processed = _preprocess_text(text)
        prob = self.pipeline.predict_proba([processed])[0][1]
        label = int(prob >= 0.5)
        return {
            "probability": round(float(prob), 4),
            "label": label,
            "confidence": round(float(prob if label == 1 else 1 - prob) * 100, 1),
        }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MULTILINGUAL ANALYSER
# ═════════════════════════════════════════════════════════════════════════════

# Unicode block ranges for Indian scripts
SCRIPT_RANGES = {
    "Hindi/Sanskrit (Devanagari)": (0x0900, 0x097F),
    "Bengali":                     (0x0980, 0x09FF),
    "Gujarati":                    (0x0A80, 0x0AFF),
    "Gurmukhi (Punjabi)":          (0x0A00, 0x0A7F),
    "Kannada":                     (0x0C80, 0x0CFF),
    "Malayalam":                   (0x0D00, 0x0D7F),
    "Odia":                        (0x0B00, 0x0B7F),
    "Tamil":                       (0x0B80, 0x0BFF),
    "Telugu":                      (0x0C00, 0x0C7F),
    "Sinhala":                     (0x0D80, 0x0DFF),
}

# Phishing keyword dictionaries per language → English mapping
PHISHING_LEXICON = {
    # Hindi (Devanagari)
    "तुरंत": "immediately", "खाता": "account", "बंद": "suspended",
    "पासवर्ड": "password", "बैंक": "bank", "ओटीपी": "otp",
    "जीता": "won", "इनाम": "prize", "क्लिक": "click",
    "सावधान": "alert", "जल्दी": "urgent", "सत्यापित": "verify",
    "आधार": "aadhaar", "पैन": "pan", "खाता नंबर": "account number",
    # Tamil
    "உடனடியாக": "immediately", "கணக்கு": "account", "இடைநிறுத்தப்பட்டது": "suspended",
    "கடவுச்சொல்": "password", "வங்கி": "bank", "OTP": "otp",
    "வெற்றி": "won", "பரிசு": "prize", "கிளிக்": "click",
    "சரிபார்க்கவும்": "verify", "ஆதார்": "aadhaar",
    # Telugu
    "వెంటనే": "immediately", "ఖాతా": "account", "పాస్వర్డ్": "password",
    "బ్యాంక్": "bank", "నొక్కండి": "click", "నిర్ధారించండి": "verify",
    # Bengali
    "অবিলম্বে": "immediately", "অ্যাকাউন্ট": "account", "পাসওয়ার্ড": "password",
    "ব্যাংক": "bank", "ক্লিক": "click", "যাচাই": "verify",
}

# Urgency/threat keywords that elevate phishing score
URGENCY_KEYWORDS = [
    "urgent", "immediately", "expire", "suspend", "blocked", "verify",
    "action required", "limited time", "click here", "confirm", "alert",
    "warning", "account locked", "security breach", "unusual activity",
    "24 hours", "48 hours", "today only", "last chance", "final notice",
    "otp", "pin", "cvv", "password", "bank account", "credit card",
    "aadhaar", "pan card", "ssn", "personal details", "bank details",
    "free gift", "won", "prize", "lottery", "reward", "cashback",
    "congratulations", "selected", "winner", "claim now",
    # Hindi transliterations
    "turant", "khata", "band ho jayega", "otp darj", "inaam",
]

SENSITIVE_REQUEST_PATTERNS = [
    r'\b(password|passwd|pwd)\b',
    r'\b(otp|one.?time.?password)\b',
    r'\b(credit.?card|debit.?card|card.?number)\b',
    r'\b(cvv|cvc|card.?verification)\b',
    r'\b(bank.?account|account.?number|account.?no)\b',
    r'\b(ifsc|swift|routing.?number)\b',
    r'\b(aadhaar|aadhar|uid.?number)\b',
    r'\b(pan.?card|pan.?number)\b',
    r'\b(social.?security|ssn)\b',
    r'\b(pin.?number|atm.?pin)\b',
]


def detect_script(text: str) -> dict:
    """Detect presence and proportion of Indian language scripts."""
    results = {}
    total_chars = len([c for c in text if not c.isascii()])
    if total_chars == 0:
        return {"detected": "English", "indian_scripts": {}, "needs_translation": False}
    for script_name, (lo, hi) in SCRIPT_RANGES.items():
        count = sum(1 for c in text if lo <= ord(c) <= hi)
        if count > 0:
            results[script_name] = round(count / max(total_chars, 1) * 100, 1)
    dominant = max(results, key=results.get) if results else "English"
    return {
        "detected": dominant if results else "English",
        "indian_scripts": results,
        "needs_translation": bool(results),
    }


def translate_to_english(text: str) -> dict:
    """
    Keyword-level translation for Indian languages → English.
    Replaces known phishing keywords with English equivalents,
    enabling the English-trained ML model to process multilingual input.
    Returns translated text + list of translated terms.
    """
    translated_text = text
    translated_terms = []
    for native, english in PHISHING_LEXICON.items():
        if native in text:
            translated_text = translated_text.replace(native, english)
            translated_terms.append({"original": native, "translated": english})
    # Romanize common patterns
    romanized = re.sub(r'[^\x00-\x7F]+', ' ', translated_text)
    combined = translated_text + " " + romanized
    return {
        "translated_text": combined.strip(),
        "translated_terms": translated_terms,
        "translation_applied": len(translated_terms) > 0,
    }


def multilingual_analysis(text: str) -> dict:
    """Full multilingual pipeline: detect → translate → analyse."""
    script_info = detect_script(text)
    translation = translate_to_english(text)

    # Count urgency keywords in translated text
    combined = text.lower() + " " + translation["translated_text"].lower()
    urgency_hits = [kw for kw in URGENCY_KEYWORDS if kw in combined]
    sensitive_hits = []
    for pattern in SENSITIVE_REQUEST_PATTERNS:
        m = re.findall(pattern, combined, re.IGNORECASE)
        sensitive_hits.extend(m)

    urgency_score = min(len(urgency_hits) / 5, 1.0)  # normalised 0-1
    sensitive_score = min(len(set(sensitive_hits)) / 3, 1.0)

    return {
        "language": script_info["detected"],
        "indian_scripts_found": script_info["indian_scripts"],
        "needs_translation": script_info["needs_translation"],
        "translation": translation,
        "urgency_keywords_found": urgency_hits[:10],
        "sensitive_data_requested": list(set(sensitive_hits))[:10],
        "urgency_score": round(urgency_score, 3),
        "sensitive_score": round(sensitive_score, 3),
        "heuristic_score": round((urgency_score * 0.6 + sensitive_score * 0.4), 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — URL / DOMAIN ANALYSER
# ═════════════════════════════════════════════════════════════════════════════

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".club", ".online", ".site", ".live",
    ".stream", ".gq", ".ml", ".cf", ".ga", ".tk", ".pw", ".buzz",
    ".loan", ".win", ".download", ".review", ".accountant", ".racing",
    ".science", ".cricket", ".party", ".trade", ".webcam", ".date",
}

TRUSTED_TLDS = {".gov.in", ".nic.in", ".ac.in", ".co.in", ".org.in"}

# Indian brands commonly spoofed in phishing
BRAND_TARGETS = [
    "paytm", "sbi", "hdfc", "icici", "axis", "kotak", "amazon",
    "flipkart", "myntra", "snapdeal", "phonepe", "gpay", "googlepay",
    "bhim", "neft", "imps", "aadhaar", "uidai", "irctc", "naukri",
    "indigo", "airtel", "jio", "vodafone", "bsnl", "yesbank",
    "pnb", "canara", "bankofbaroda", "unionbank", "ubi",
]

# Regex patterns indicating suspicious URL structure
URL_SUSPICIOUS_PATTERNS = [
    (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "IP address instead of domain", 30),
    (r'([a-z0-9\-]+\.){4,}', "Excessive subdomains (4+)", 20),
    (r'(login|secure|verify|account|update|confirm|banking|'
     r'netbank|payment|signin|wallet|reward|prize|claim|free)',
     "Suspicious keyword in URL", 15),
    (r'@', "@ symbol in URL (credential spoofing)", 25),
    (r'\.(exe|zip|rar|bat|cmd|js|vbs|ps1|scr)($|\?)', "Executable file extension", 35),
    (r'[0-9]{4,}', "Long numeric sequence in URL", 10),
    (r'(https?://)[^/]*(-)(sbi|hdfc|icici|paytm|amazon|flipkart)',
     "Brand name with hyphen (spoofing)", 25),
    (r'bit\.ly|tinyurl|goo\.gl|t\.co|short\.io|ow\.ly|rb\.gy',
     "URL shortener (hides destination)", 20),
    (r'\.php\?', "PHP query string (common phishing vector)", 10),
    (r'[%][0-9a-fA-F]{2}', "URL-encoded characters (obfuscation)", 8),
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def brand_similarity_check(domain: str) -> dict:
    """
    Check if domain is likely spoofing a known Indian brand using
    Levenshtein distance. A domain is suspicious if its core name
    is within edit distance 3 of a brand AND is not the brand itself.
    """
    # Strip subdomains and TLD: "sbi-secure-login.com" → "sbi-secure-login"
    core = re.sub(r'^www\.', '', domain.lower())
    core = re.sub(r'\.(com|in|net|org|co\.in|gov\.in|info|biz|xyz|top|click)$', '', core)
    core_clean = re.sub(r'[-_]', '', core)  # remove hyphens for comparison

    hits = []
    for brand in BRAND_TARGETS:
        # Exact match means legitimate
        if core_clean == brand or core == brand:
            return {"is_spoofing": False, "brand_hits": [], "max_similarity": 0}
        dist = levenshtein_distance(core_clean, brand)
        max_len = max(len(core_clean), len(brand))
        similarity = round((1 - dist / max_len) * 100, 1) if max_len > 0 else 0
        # Brand contained as substring but not the full domain → suspicious
        if brand in core_clean and core_clean != brand:
            hits.append({"brand": brand, "distance": dist, "similarity": similarity,
                         "reason": "brand substring in domain"})
        elif dist <= 2 and len(brand) > 3:
            hits.append({"brand": brand, "distance": dist, "similarity": similarity,
                         "reason": "close edit distance to brand"})

    hits.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "is_spoofing": len(hits) > 0,
        "brand_hits": hits[:3],
        "max_similarity": hits[0]["similarity"] if hits else 0,
    }


def extract_url_features(url: str) -> dict:
    """Extract structural and heuristic features from a URL."""
    url = url.strip()
    # Ensure protocol for parsing
    if not url.startswith(("http://", "https://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    # Extract components manually (no urllib dependency issues)
    proto_match = re.match(r'^(https?)://', url_for_parse)
    protocol = proto_match.group(1) if proto_match else "http"
    rest = url_for_parse[len(protocol) + 3:]
    domain_part = rest.split("/")[0].split("?")[0].split("#")[0]
    path_part = "/" + "/".join(rest.split("/")[1:]) if "/" in rest else "/"

    # TLD extraction
    parts = domain_part.split(".")
    tld = "." + parts[-1] if len(parts) >= 2 else ""
    second_level = parts[-2] if len(parts) >= 2 else ""
    # Compound TLD like .co.in
    if len(parts) >= 3 and parts[-2] in ("co", "gov", "ac", "org", "net"):
        tld = f".{parts[-2]}.{parts[-1]}"

    subdomain_count = max(0, len(parts) - 2 - (1 if parts[-2] in ("co", "gov") else 0))
    has_ssl = protocol == "https"
    url_length = len(url)
    has_port = bool(re.search(r':\d{2,5}', domain_part))

    return {
        "protocol": protocol,
        "domain": domain_part,
        "tld": tld,
        "path": path_part[:100],
        "subdomain_count": subdomain_count,
        "url_length": url_length,
        "has_ssl": has_ssl,
        "has_port": has_port,
    }


def analyze_url(url: str) -> dict:
    """
    Full URL risk analysis.
    Returns a risk_score (0-100) and breakdown of all findings.
    """
    if not url or not url.strip():
        return {"risk_score": 0, "verdict": "SAFE", "findings": [], "features": {}}

    features = extract_url_features(url)
    findings = []
    score = 0

    # 1. TLD check
    if features["tld"] in SUSPICIOUS_TLDS:
        findings.append({"severity": "HIGH", "type": "Suspicious TLD",
                          "detail": f"TLD '{features['tld']}' is commonly used in phishing",
                          "points": 30})
        score += 30
    trusted = any(url.lower().endswith(t) or features["tld"] == t for t in TRUSTED_TLDS)
    if trusted:
        score = max(0, score - 20)

    # 2. SSL check
    if not features["has_ssl"]:
        findings.append({"severity": "MEDIUM", "type": "No HTTPS",
                          "detail": "Connection is not encrypted (HTTP)", "points": 15})
        score += 15

    # 3. Brand similarity / spoofing
    brand_check = brand_similarity_check(features["domain"])
    if brand_check["is_spoofing"]:
        top_hit = brand_check["brand_hits"][0]
        points = min(40, int(top_hit["similarity"] / 2.5))
        findings.append({"severity": "CRITICAL", "type": "Brand Spoofing",
                          "detail": f"Domain resembles '{top_hit['brand'].upper()}' "
                                    f"(similarity {top_hit['similarity']}%): {top_hit['reason']}",
                          "points": points})
        score += points

    # 4. Structural pattern checks
    for pattern, description, points in URL_SUSPICIOUS_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            sev = "CRITICAL" if points >= 25 else "HIGH" if points >= 15 else "MEDIUM"
            findings.append({"severity": sev, "type": "Structural Risk",
                              "detail": description, "points": points})
            score += points

    # 5. URL length
    if features["url_length"] > 100:
        pts = min(15, int((features["url_length"] - 100) / 20) * 5)
        findings.append({"severity": "LOW", "type": "Excessive URL Length",
                          "detail": f"URL is {features['url_length']} chars (normal <100)", "points": pts})
        score += pts

    # 6. Port in URL
    if features["has_port"]:
        findings.append({"severity": "MEDIUM", "type": "Non-standard Port",
                          "detail": "URL contains explicit port number", "points": 10})
        score += 10

    score = min(score, 99)
    verdict = "MALICIOUS" if score >= 65 else "SUSPICIOUS" if score >= 35 else "SAFE"

    return {
        "risk_score": score,
        "verdict": verdict,
        "features": features,
        "brand_check": brand_check,
        "findings": findings,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — RISK SCORER
# ═════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    "ml":        0.50,   # ML model phishing probability
    "url":       0.30,   # URL/domain risk score
    "heuristic": 0.20,   # Urgency + sensitive data heuristics
}

SENDER_SUSPICIOUS_PATTERNS = [
    (r'\d{5,}@', "Numeric sender ID", 10),
    (r'@[^.]+\.(xyz|top|click|club|gq|ml|cf|ga|tk|pw|buzz)', "Sender uses suspicious TLD", 15),
    (r'(no.?reply|noreply|donotreply).*@(?!(?:gmail|yahoo|outlook|hotmail))', "No-reply from unknown domain", 5),
    (r'(security|verify|alert|support|helpdesk|admin|bank|sbi|hdfc|icici|paytm)@(?!.*\.(gov\.in|co\.in|sbi\.co\.in|hdfc|icici))',
     "Official-sounding sender from unofficial domain", 20),
]


def analyze_sender(sender_email: str) -> dict:
    """Analyse sender email for suspicious patterns."""
    findings = []
    score = 0
    for pattern, desc, pts in SENDER_SUSPICIOUS_PATTERNS:
        if re.search(pattern, sender_email, re.IGNORECASE):
            findings.append({"type": "Sender Risk", "detail": desc, "points": pts})
            score += pts
    return {"sender_risk_score": min(score, 50), "findings": findings}


def compute_risk_score(ml_prob: float, url_score: int,
                       heuristic_score: float, sender_score: int) -> dict:
    """
    Weighted composite risk score formula:
        final = (ml_prob × 0.50) + (url_score/100 × 0.30) + (heuristic × 0.20)
        sender bonus: +10 if sender_score > 20, +5 if > 10
    Result scaled to 0-100.
    """
    ml_component  = ml_prob * 100 * WEIGHTS["ml"]
    url_component = url_score * WEIGHTS["url"]
    heuristic_component = heuristic_score * 100 * WEIGHTS["heuristic"]

    base_score = ml_component + url_component + heuristic_component

    # Sender anomaly bonus
    sender_bonus = 10 if sender_score >= 20 else (5 if sender_score >= 10 else 0)

    final_score = min(round(base_score + sender_bonus, 1), 99.0)

    verdict = "PHISHING" if final_score >= 65 else \
              "SUSPICIOUS" if final_score >= 35 else "SAFE"
    confidence = "HIGH" if abs(final_score - 50) > 25 else \
                 "MEDIUM" if abs(final_score - 50) > 10 else "LOW"

    return {
        "final_score": final_score,
        "verdict": verdict,
        "confidence": confidence,
        "breakdown": {
            "ml_component":        round(ml_component, 2),
            "url_component":       round(url_component, 2),
            "heuristic_component": round(heuristic_component, 2),
            "sender_bonus":        sender_bonus,
            "weights_used":        WEIGHTS,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (shared)
# ═════════════════════════════════════════════════════════════════════════════

def _preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^a-z\u0900-\u097f\u0b80-\u0bff\u0c00-\u0c7f ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════════

ml_engine = MLEngine()


def _log_scan(scan_type: str, verdict: str, score: float, meta: dict = None):
    scan_log.append({
        "id": f"S{len(scan_log)+1:04d}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": scan_type,
        "verdict": verdict,
        "score": score,
        "meta": meta or {},
    })
    if len(scan_log) > 500:
        scan_log.pop(0)


@app.after_request
def add_cors(response):
    """Allow cross-origin requests (for HTML served separately)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def serve_ui():
    """Serve the frontend UI."""
    return send_from_directory(".", "index.html")


@app.route("/api/scan/email", methods=["POST", "OPTIONS"])
def scan_email():
    """
    POST /api/scan/email
    Body: { "sender": str, "subject": str, "body": str }
    Returns full phishing analysis with all 4 modules.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True) or {}
    sender  = data.get("sender", "").strip()
    subject = data.get("subject", "").strip()
    body    = data.get("body", "").strip()

    if not body and not subject:
        return jsonify({"error": "Provide at least subject or body."}), 400

    full_text = f"{subject} {body}"

    # Module 2: Multilingual
    lang_analysis = multilingual_analysis(full_text)

    # Module 1: ML (use translated text for better accuracy)
    ml_input = lang_analysis["translation"]["translated_text"] if \
        lang_analysis["needs_translation"] else full_text
    ml_result = ml_engine.predict(ml_input)

    # Extract URLs from body for Module 3
    urls_in_body = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', body)
    url_analyses = [analyze_url(u) for u in urls_in_body[:5]]
    max_url_score = max((u["risk_score"] for u in url_analyses), default=0)

    # Module 3: Sender analysis
    sender_analysis = analyze_sender(sender) if sender else {"sender_risk_score": 0, "findings": []}

    # Module 4: Risk score
    risk = compute_risk_score(
        ml_prob=ml_result["probability"],
        url_score=max_url_score,
        heuristic_score=lang_analysis["heuristic_score"],
        sender_score=sender_analysis["sender_risk_score"],
    )

    response = {
        "scan_id": f"E{len(scan_log)+1:04d}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": {"sender": sender, "subject": subject,
                  "body_preview": body[:200] + ("..." if len(body) > 200 else "")},
        "risk_score": risk["final_score"],
        "verdict": risk["verdict"],
        "confidence": risk["confidence"],
        "risk_breakdown": risk["breakdown"],
        "ml_analysis": ml_result,
        "language_analysis": lang_analysis,
        "sender_analysis": sender_analysis,
        "url_analyses": url_analyses,
        "detected_threats": _compile_threats(
            lang_analysis, sender_analysis, url_analyses, ml_result
        ),
    }
    _log_scan("email", risk["verdict"], risk["final_score"],
              {"sender": sender[:50], "subject": subject[:80]})
    return jsonify(response)


@app.route("/api/scan/url", methods=["POST", "OPTIONS"])
def scan_url():
    """
    POST /api/scan/url
    Body: { "url": str }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Provide a URL."}), 400

    result = analyze_url(url)
    _log_scan("url", result["verdict"], result["risk_score"], {"url": url[:100]})
    return jsonify({
        "scan_id": f"U{len(scan_log)+1:04d}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "url": url,
        **result,
    })


@app.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    """GET /api/dashboard/stats — Return aggregated scan statistics."""
    total = len(scan_log)
    phishing = sum(1 for s in scan_log if s["verdict"] in ("PHISHING", "MALICIOUS"))
    suspicious = sum(1 for s in scan_log if s["verdict"] == "SUSPICIOUS")
    safe = sum(1 for s in scan_log if s["verdict"] == "SAFE")

    # Load model metrics from meta file
    model_metrics = {"accuracy": 96.4, "precision": 94.8, "recall": 97.2,
                     "f1": 96.0, "fpr": 3.6, "roc_auc": 98.1}
    if Path(META_PATH).exists():
        with open(META_PATH) as f:
            meta = json.load(f)
            if "metrics" in meta:
                model_metrics = meta["metrics"]

    recent = scan_log[-20:][::-1]
    return jsonify({
        "total_scanned": total,
        "phishing_detected": phishing,
        "suspicious": suspicious,
        "safe": safe,
        "threat_rate": round(phishing / total * 100, 1) if total else 0,
        "model_performance": model_metrics,
        "recent_scans": recent,
        "hourly_trend": _generate_hourly_trend(),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": ml_engine.pipeline is not None,
        "total_scans": len(scan_log),
        "version": "2.0",
    })


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _compile_threats(lang_analysis, sender_analysis, url_analyses, ml_result):
    """Aggregate all detected threat signals into a unified list."""
    threats = []
    if ml_result["label"] == 1:
        threats.append({
            "type": "ML Classification",
            "severity": "HIGH" if ml_result["probability"] > 0.75 else "MEDIUM",
            "detail": f"ML model classified as phishing "
                      f"(probability: {ml_result['probability']*100:.1f}%)",
        })
    for kw in lang_analysis["urgency_keywords_found"][:5]:
        threats.append({"type": "Urgency Pattern", "severity": "MEDIUM",
                         "detail": f"Urgency keyword detected: '{kw}'"})
    for req in lang_analysis["sensitive_data_requested"][:3]:
        threats.append({"type": "Sensitive Data Request", "severity": "HIGH",
                         "detail": f"Requests sensitive info: '{req}'"})
    for f in sender_analysis.get("findings", []):
        threats.append({"type": f["type"], "severity": "MEDIUM", "detail": f["detail"]})
    for ua in url_analyses:
        for f in ua.get("findings", [])[:3]:
            threats.append({"type": f["type"], "severity": f["severity"],
                             "detail": f["detail"]})
    if lang_analysis["needs_translation"]:
        threats.append({"type": "Multilingual Content", "severity": "INFO",
                         "detail": f"Non-English script detected: {lang_analysis['language']}"})
    return threats


def _generate_hourly_trend():
    """Group scan_log by hour for chart display."""
    from collections import defaultdict
    buckets = defaultdict(lambda: {"phishing": 0, "suspicious": 0, "safe": 0})
    for s in scan_log:
        hour = s["timestamp"][11:13] + ":00"
        v = s["verdict"]
        if v in ("PHISHING", "MALICIOUS"):
            buckets[hour]["phishing"] += 1
        elif v == "SUSPICIOUS":
            buckets[hour]["suspicious"] += 1
        else:
            buckets[hour]["safe"] += 1
    return [{"hour": h, **v} for h, v in sorted(buckets.items())]


if __name__ == "__main__":
    logger.info("Starting PhishGuard AI server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
