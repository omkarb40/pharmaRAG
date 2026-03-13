"""
PharmaRAG — Data Ingestion Demo
Fetches FDA drug labels from DailyMed, parses XML, chunks into searchable text.

What it does:
  1. Reads drug list from CSV
  2. Searches DailyMed API for each drug → gets set_id
  3. Downloads full SPL XML for each drug label
  4. Parses XML to extract safety-relevant sections
  5. Chunks sections into retrieval-sized pieces
  6. Saves everything as JSONL (one JSON per line)
"""

import csv
import json
import hashlib
import re
import time
from pathlib import Path

import requests
from lxml import etree


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
DRUG_LIST = Path("data/drug_lists/ms_drugs.csv")
RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

# SPL sections we care about (LOINC code → readable name)
# These are the standard FDA section identifiers
SECTION_MAP = {
    "34067-9":  "indications_and_usage",
    "34070-3":  "contraindications",
    "43685-7":  "warnings_and_precautions",
    "42232-9":  "boxed_warning",
    "34084-4":  "adverse_reactions",
    "34068-7":  "dosage_and_administration",
    "34073-7":  "drug_interactions",
    "43684-0":  "use_in_specific_populations",
    "34071-1":  "warnings",
}

CHUNK_SIZE_CHARS = 2000    # ~500 tokens
CHUNK_OVERLAP_CHARS = 200  # ~50 tokens overlap

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# STEP 1: Fetch from DailyMed API
# ──────────────────────────────────────────────

def search_drug(drug_name: str) -> str | None:
    """Search DailyMed for a drug, return the set_id of the first result."""
    url = f"{DAILYMED_API}/spls.json"
    params = {"drug_name": drug_name, "pagesize": 1}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            set_id = data["data"][0]["setid"]
            title = data["data"][0]["title"]
            print(f"  ✓ Found: {title[:80]}...")
            return set_id
    except Exception as e:
        print(f"  ✗ API error: {e}")
    return None


def download_spl_xml(set_id: str, drug_name: str) -> Path | None:
    """Download the full SPL XML document for a drug."""
    out_path = RAW_DIR / f"{drug_name.replace(' ', '_')}.xml"

    if out_path.exists():
        print(f"  → Already downloaded: {out_path.name}")
        return out_path

    # The /spls/{SETID}.xml endpoint returns the raw SPL XML
    url = f"{DAILYMED_API}/spls/{set_id}.xml"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"  → Saved XML: {out_path.name} ({len(resp.content):,} bytes)")
        return out_path
    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return None


def fetch_all_drugs() -> list[dict]:
    """Read drug list CSV, fetch SPL XML for each drug."""
    fetched = []
    with open(DRUG_LIST, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["generic_name"]
            brand = row["drug_name"]
            category = row["category"]
            print(f"\n[{brand}] Searching for '{name}'...")

            set_id = search_drug(name)
            if not set_id:
                print(f"  ✗ Not found on DailyMed, skipping.")
                continue

            xml_path = download_spl_xml(set_id, name)
            if xml_path:
                fetched.append({
                    "drug_name": brand,
                    "generic_name": name,
                    "category": category,
                    "set_id": set_id,
                    "xml_path": str(xml_path),
                })

            time.sleep(1)  # Be polite to the API

    return fetched


# ──────────────────────────────────────────────
# STEP 2: Parse SPL XML → Extract Sections
# ──────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Strip HTML/XML tags, normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_spl_xml(xml_path: str, drug_info: dict) -> list[dict]:
    try:
        tree = etree.parse(xml_path)
    except Exception as e:
        print(f"  ✗ XML parse error for {xml_path}: {e}")
        return []

    root = tree.getroot()
    # SPL uses the HL7 namespace
    ns = {"hl7": "urn:hl7-org:v3"}

    sections = []

    # Find all <section> elements that have a <code> child
    for section_el in root.iter("{urn:hl7-org:v3}section"):
        code_el = section_el.find("hl7:code", ns)
        if code_el is None:
            continue

        loinc_code = code_el.get("code", "")
        section_name = SECTION_MAP.get(loinc_code)
        if not section_name:
            continue  # Skip sections outside our taxonomy

        # Extract ALL text content from this section (including nested subsections)
        # etree.tostring gets the full XML subtree as string
        raw_xml = etree.tostring(section_el, encoding="unicode", method="text")
        clean = clean_text(raw_xml)

        if len(clean) < 30:
            continue  # Skip trivially short sections

        sections.append({
            "drug_name": drug_info["drug_name"],
            "generic_name": drug_info["generic_name"],
            "set_id": drug_info["set_id"],
            "section_name": section_name,
            "loinc_code": loinc_code,
            "text": clean,
            "char_count": len(clean),
        })

    return sections


# ──────────────────────────────────────────────
# STEP 3: Chunk Sections into Retrieval Pieces
# ──────────────────────────────────────────────

def make_chunk_id(drug: str, section: str, idx: int) -> str:
    """Deterministic chunk ID based on drug, section, and index."""
    raw = f"{drug}::{section}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def parse_spl_xml(xml_path: str, drug_info: dict) -> list[dict]:
    """
    Parse SPL XML and extract text from safety-relevant sections.
    Only captures top-level matching sections (avoids duplicate nested text).
    """
    try:
        tree = etree.parse(xml_path)
    except Exception as e:
        print(f"  ✗ XML parse error for {xml_path}: {e}")
        return []

    root = tree.getroot()
    ns = {"hl7": "urn:hl7-org:v3"}

    sections = []
    seen_loinc_codes = set()  # Track which section types we've already captured

    for section_el in root.iter("{urn:hl7-org:v3}section"):
        code_el = section_el.find("hl7:code", ns)
        if code_el is None:
            continue

        loinc_code = code_el.get("code", "")
        section_name = SECTION_MAP.get(loinc_code)
        if not section_name:
            continue

        # Skip duplicate section types (parent already captured child text)
        if loinc_code in seen_loinc_codes:
            continue
        seen_loinc_codes.add(loinc_code)

        # Extract text content
        raw_xml = etree.tostring(section_el, encoding="unicode", method="text")
        clean = clean_text(raw_xml)

        if len(clean) < 30:
            continue

        sections.append({
            "drug_name": drug_info["drug_name"],
            "generic_name": drug_info["generic_name"],
            "set_id": drug_info["set_id"],
            "section_name": section_name,
            "loinc_code": loinc_code,
            "text": clean,
            "char_count": len(clean),
        })

    return sections


def chunk_section(section: dict) -> list[dict]:
    """Split a section into overlapping chunks, breaking at sentence boundaries."""
    text = section["text"]
    drug = section["drug_name"]
    sec = section["section_name"]

    if len(text) <= CHUNK_SIZE_CHARS:
        return [{
            "chunk_id": make_chunk_id(drug, sec, 0),
            "drug_name": drug,
            "generic_name": section["generic_name"],
            "set_id": section["set_id"],
            "section_name": sec,
            "loinc_code": section["loinc_code"],
            "chunk_index": 0,
            "total_chunks": 1,
            "text": text,
        }]

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE_CHARS, len(text))

        # Try to break at a sentence boundary, but ONLY look forward
        # from a safe minimum position to prevent backward jumps
        if end < len(text):
            # Only search the last 300 chars, but never go before
            # the midpoint of the chunk to prevent tiny chunks
            min_end = start + (CHUNK_SIZE_CHARS // 2)
            search_start = max(end - 300, min_end)
            if search_start < end:
                search_zone = text[search_start:end]
                last_period = search_zone.rfind(". ")
                if last_period != -1:
                    end = search_start + last_period + 2

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": make_chunk_id(drug, sec, idx),
                "drug_name": drug,
                "generic_name": section["generic_name"],
                "set_id": section["set_id"],
                "section_name": sec,
                "loinc_code": section["loinc_code"],
                "chunk_index": idx,
                "text": chunk_text,
            })
            idx += 1

        # Ensure start always advances by at least half the chunk size
        new_start = end - CHUNK_OVERLAP_CHARS
        if new_start <= start:
            new_start = start + (CHUNK_SIZE_CHARS // 2)
        start = new_start

    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PharmaRAG — Phase 1.0 Ingestion Demo")
    print("=" * 60)

    # Step 1: Fetch from DailyMed
    print("\n📡 STEP 1: Fetching drug labels from DailyMed API...")
    drug_infos = fetch_all_drugs()
    print(f"\n✓ Fetched {len(drug_infos)} drug labels")

    # Step 2: Parse XML sections
    print("\n📄 STEP 2: Parsing SPL XML sections...")
    all_sections = []
    for info in drug_infos:
        sections = parse_spl_xml(info["xml_path"], info)
        all_sections.extend(sections)
        print(f"  [{info['drug_name']}] → {len(sections)} sections extracted")

    print(f"\n✓ Total sections: {len(all_sections)}")

    # Print section distribution
    from collections import Counter
    sec_counts = Counter(s["section_name"] for s in all_sections)
    print("\n  Section distribution:")
    for sec, count in sec_counts.most_common():
        print(f"    {sec}: {count}")

    # Step 3: Chunk
    print("\n✂️  STEP 3: Chunking sections...")
    all_chunks = []
    for section in all_sections:
        all_chunks.extend(chunk_section(section))

    print(f"\n✓ Total chunks: {len(all_chunks)}")

    # Step 4: Save to JSONL
    print(f"\n💾 STEP 4: Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n{'=' * 60}")
    print(f"✅ DONE! {len(all_chunks)} chunks saved to {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    # Show a sample chunk
    print(f"\n📋 Sample chunk:")
    sample = all_chunks[0]
    print(f"  Drug:    {sample['drug_name']}")
    print(f"  Section: {sample['section_name']}")
    print(f"  ID:      {sample['chunk_id']}")
    print(f"  Text:    {sample['text'][:200]}...")


if __name__ == "__main__":
    main()