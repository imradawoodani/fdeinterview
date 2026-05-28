"""Curated, freely-available online documentation sources for each category.

Each entry is fetched live during ingestion. We deliberately use authoritative,
stable, no-auth sources:
  - eCFR API for federal regulations (OSHA 29 CFR 1910, FDA 21 CFR 820)
  - Wikipedia REST/Action API for engineering & quality-control reference content
"""
from __future__ import annotations

# type "ecfr":      {title, part, section, name}
# type "wikipedia": {page, name}

SOURCES: dict[str, list[dict]] = {
    "safety": [
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.147",
         "name": "Control of Hazardous Energy (Lockout/Tagout)"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.212",
         "name": "General Requirements for All Machines (Machine Guarding)"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.132",
         "name": "Personal Protective Equipment - General Requirements"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.146",
         "name": "Permit-Required Confined Spaces"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.95",
         "name": "Occupational Noise Exposure"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.1200",
         "name": "Hazard Communication"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.157",
         "name": "Portable Fire Extinguishers"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.178",
         "name": "Powered Industrial Trucks (Forklifts)"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.303",
         "name": "Electrical - General Requirements"},
        {"type": "ecfr", "title": 29, "part": 1910, "section": "1910.23",
         "name": "Ladders"},
    ],
    "maintenance": [
        {"type": "wikipedia", "page": "Preventive maintenance",
         "name": "Preventive Maintenance"},
        {"type": "wikipedia", "page": "Predictive maintenance",
         "name": "Predictive Maintenance"},
        {"type": "wikipedia", "page": "Reliability-centered maintenance",
         "name": "Reliability-Centered Maintenance"},
        {"type": "wikipedia", "page": "Centrifugal pump",
         "name": "Centrifugal Pump Service & Operation"},
        {"type": "wikipedia", "page": "Electric motor",
         "name": "Electric Motor Maintenance"},
        {"type": "wikipedia", "page": "Rolling-element bearing",
         "name": "Rolling-Element Bearings"},
        {"type": "wikipedia", "page": "Lubrication",
         "name": "Lubrication"},
        {"type": "wikipedia", "page": "Vibration",
         "name": "Vibration (Condition Monitoring)"},
        {"type": "wikipedia", "page": "Belt (mechanical)",
         "name": "Mechanical Drive Belts"},
        {"type": "wikipedia", "page": "Gear",
         "name": "Gears & Gearboxes"},
    ],
    "quality": [
        {"type": "wikipedia", "page": "Statistical process control",
         "name": "Statistical Process Control (SPC)"},
        {"type": "wikipedia", "page": "Control chart",
         "name": "Control Charts"},
        {"type": "wikipedia", "page": "Process capability index",
         "name": "Process Capability Index (Cp/Cpk)"},
        {"type": "wikipedia", "page": "Acceptance sampling",
         "name": "Acceptance Sampling"},
        {"type": "wikipedia", "page": "Six Sigma",
         "name": "Six Sigma"},
        {"type": "wikipedia", "page": "Quality control",
         "name": "Quality Control Fundamentals"},
        {"type": "ecfr", "title": 21, "part": 820, "section": "820.30",
         "name": "Design Controls (FDA QSR)"},
        {"type": "ecfr", "title": 21, "part": 820, "section": "820.70",
         "name": "Production and Process Controls (FDA QSR)"},
        {"type": "ecfr", "title": 21, "part": 820, "section": "820.100",
         "name": "Corrective and Preventive Action / CAPA (FDA QSR)"},
        {"type": "ecfr", "title": 21, "part": 820, "section": "820.250",
         "name": "Statistical Techniques (FDA QSR)"},
    ],
}


def ecfr_xml_url(title: int, part: int, section: str, date: str) -> str:
    return (
        f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
        f"?part={part}&section={section}"
    )


def ecfr_public_url(title: int, section: str) -> str:
    return f"https://www.ecfr.gov/current/title-{title}/section-{section}"


def osha_public_url(part: int, section: str) -> str:
    return f"https://www.osha.gov/laws-regs/regulations/standardnumber/{part}/{section}"


def wikipedia_extract_url(page: str) -> str:
    return (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
        "&explaintext=1&format=json&redirects=1&titles=" + page.replace(" ", "_")
    )


def wikipedia_public_url(page: str) -> str:
    return "https://en.wikipedia.org/wiki/" + page.replace(" ", "_")
