#!/usr/bin/env python3
"""Phase 1 데이터 수집: 공식 raspberrypilearning repos + awesome-raspberry-pi README를
계획서의 JSON 스키마로 정규화해서 data/rpi_projects_dataset.json 으로 저장한다."""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXCLUDE_KEYWORDS = [
    "template", "glossary", "cheat sheet", "worksheet", "deploy resources",
    "competition finalist", "finalists", "bebras", "pedagogy", "wysiwyg",
    "test resource", "testing new resource", "translation tidy", "dev activity",
    "quick start guide", "troubleshooting guide", "frequently asked questions",
    "teachers guide", "teachers classroom guide", "parents guide",
    "addons guide", "guide for using raspberry pis in the classroom",
    "set up files for", "resource style", "hardware guide", "software guide",
    "physical computing guide", "noobs install", "edu image",
    "recruiting mentors", "staging", "hj test", "demo programs",
    "cambridgegcsecomputing", "redirect only", "patrick all blocks test",
    "test rfm translations", "indieweb",
]

EXCLUDE_TITLE_PREFIXES = ("editor ", "ada cs anvil ")

CATEGORY_RULES = [
    ("네트워크/서버", ["dns", "vpn", "server", "hosting", "nas", "proxy", "kubernetes",
                     "docker", "cluster", "wordpress", "web server", "reverse proxy",
                     "cloud", "database", "self-host"]),
    ("홈오토메이션", ["home assistant", "smart home", "automation", "thermostat",
                    "garage", "irrigation", "smart lamp", "smart mirror"]),
    ("보안/모니터링", ["security", "surveillance", "alarm", "detect", "monitor",
                    "intrusion", "presence detector", "baby monitor"]),
    ("미디어/엔터테인먼트", ["media", "kodi", "plex", "music", "video", "camera",
                        "streaming", "retro", "arcade", "game", "emulat", "synth",
                        "drum", "midi", "tweet", "sound", "light", "sparkle",
                        "pet", "buggy", "3d print"]),
    ("로봇/자동화기기", ["robot", "arm", "motor", "servo", "drone", "car computer",
                     "bartender", "cocktail", "buggy"]),
    ("IoT/센서", ["iot", "sensor", "weather", "temperature", "gps", "bluetooth",
                "ble ", "gpio", "sense hat", "data logger", "hat "]),
    ("교육/프로그래밍", ["python", "scratch", "learn", "code", "programming",
                     "tutorial", "lesson", "scheme of work", "mathematica",
                     "html", "css", "javascript", "jquery"]),
]

DIFFICULTY_HARD = ["cluster", "kubernetes", "kernel", "bare metal", "ceph",
                    "distributed", "machine learning", "neural", "ai-native",
                    "k8s", "hailo"]
DIFFICULTY_EASY = ["simple", "beginner", "easy", "quick", "basic", "intro",
                    "getting started", "for kids"]

BADGE_MAP = {
    "rpi-0": "RPi Zero", "rpi-2": "RPi 2", "rpi-2+": "RPi 2 이상",
    "rpi-3": "RPi 3", "rpi-4": "RPi 4", "rpi-5": "RPi 5",
}


def infer_category(text: str) -> str:
    low = text.lower()
    for label, keywords in CATEGORY_RULES:
        if any(kw in low for kw in keywords):
            return label
    return "기타"


def infer_difficulty(text: str) -> str:
    low = text.lower()
    if any(kw in low for kw in DIFFICULTY_HARD):
        return "어려움"
    if any(kw in low for kw in DIFFICULTY_EASY):
        return "쉬움"
    return "보통"


def infer_hardware(text: str, badge_ids=None) -> list:
    hw = set()
    if badge_ids:
        for b in badge_ids:
            if b in BADGE_MAP:
                hw.add(BADGE_MAP[b])
    low = text.lower()
    if "zero" in low:
        hw.add("RPi Zero")
    if "pico" in low:
        hw.add("RPi Pico")
    if not hw:
        hw.add("RPi 3B+ 이상")
    return sorted(hw)


def infer_use_case(title: str, description: str) -> list:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", f"{title} {description}".lower())
    stop = {"with", "your", "this", "that", "from", "using", "into", "the",
            "raspberry", "which", "have", "make", "makes", "learn", "project",
            "will", "able", "when", "than", "also", "more", "other"}
    seen, use_case = [], []
    for w in words:
        if w in stop or w in seen:
            continue
        seen.append(w)
        use_case.append(w)
        if len(use_case) == 4:
            break
    return use_case


def make_id(source: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{source}-{slug}"[:80]


def make_entry(source, title, url, description, badge_ids=None):
    text = f"{title} {description}"
    entry = {
        "id": make_id(source, title),
        "title": title,
        "category": infer_category(text),
        "difficulty": infer_difficulty(text),
        "hardware": infer_hardware(text, badge_ids),
        "description": description,
        "use_case": infer_use_case(title, description),
        "source": source,
        "url": url,
        "human_query": f"{title} 같은 걸 라즈베리파이로 만들어보고 싶은데 어떻게 시작하면 될까?",
        "ai_expected_output": (
            f"{title}를(을) 추천합니다. {description} "
            f"(참고: {url})"
        ),
    }
    return entry


def parse_official_tsv():
    path = DATA_DIR / "raspberrypilearning_repos_all.tsv"
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, archived, description = line.split("\t", 2)
        description = description.strip()
        if not description or len(description) < 15:
            continue
        name_spaced = name.replace("-", " ").lower()
        low = f"{name_spaced} {description}".lower()
        if any(kw in low for kw in EXCLUDE_KEYWORDS):
            continue
        if name_spaced.startswith(EXCLUDE_TITLE_PREFIXES):
            continue
        title = name.replace("-", " ").title()
        url = f"https://github.com/raspberrypilearning/{name}"
        entries.append(make_entry("official", title, url, description))
    return entries


OS_IMAGE_EXCLUDE = ["not maintained", "discontinued"]


def parse_os_images():
    path = DATA_DIR / "awesome_raspberry_pi_raw.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    section = None
    entries = []
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if section != "OS Images":
            continue
        m = LIST_ITEM_RE.match(line.strip())
        if not m:
            continue
        title, url, rest = m.groups()
        badge_ids = BADGE_RE.findall(rest)
        description = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", rest).strip()
        description = description.rstrip(".") + "."
        entry = make_entry("awesome-list-os", title, url, description, badge_ids)
        entry["category"] = "OS/펌웨어"
        entry["human_query"] = f"{title}는 어떤 용도로 쓰기 좋은 라즈베리파이 OS야?"
        entry["ai_expected_output"] = (
            f"{title}를(을) 추천합니다. {description} (참고: {url})"
        )
        entries.append(entry)
    return entries


LIST_ITEM_RE = re.compile(r"^-\s+\[([^\]]+)\]\(([^)]+)\)\s*-\s*(.+)$")
BADGE_RE = re.compile(r"badges/(rpi-[0-9+]+)\.png")


def parse_awesome_list():
    path = DATA_DIR / "awesome_raspberry_pi_raw.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    section = None
    entries = []
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if section not in ("Projects", "Tools"):
            continue
        m = LIST_ITEM_RE.match(line.strip())
        if not m:
            continue
        title, url, rest = m.groups()
        badge_ids = BADGE_RE.findall(rest)
        description = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", rest).strip()
        description = description.rstrip(".") + "."
        entries.append(make_entry("awesome-list", title, url, description, badge_ids))
    return entries


def main():
    official = parse_official_tsv()
    awesome = parse_awesome_list()
    os_images = parse_os_images()
    all_entries = official + awesome + os_images

    seen_ids = set()
    deduped = []
    for e in all_entries:
        if e["id"] in seen_ids:
            continue
        seen_ids.add(e["id"])
        deduped.append(e)

    out_path = DATA_DIR / "rpi_projects_dataset.json"
    out_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"official: {len(official)}")
    print(f"awesome-list: {len(awesome)}")
    print(f"os-images: {len(os_images)}")
    print(f"total (deduped): {len(deduped)}")
    from collections import Counter
    cat_counts = Counter(e["category"] for e in deduped)
    diff_counts = Counter(e["difficulty"] for e in deduped)
    print("category distribution:", dict(cat_counts))
    print("difficulty distribution:", dict(diff_counts))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
