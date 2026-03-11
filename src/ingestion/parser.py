"""
SPL XML/JSON parser.
Extracts relevant sections from DailyMed SPL data.
"""

import json
import re
from pathlib import Path
from typing import Optional
from lxml import etree


class SPLParser:
    """Parse DailyMed SPL data and extract labeled sections."""

    SECTION_LOINC_MAP = {
        "34067-9": "indications_and_usage",
        "34070-3": "contraindications",
        "43685-7": "warnings_and_precautions",
        "34084-4": "adverse_reactions",
        "34068-7": "dosage_and_administration",
        "34073-7": "drug_interactions",
        "43684-0": "use_in_specific_populations",
        "34071-1": "warnings",
        "42232-9": "boxed_warning",
    }

    @staticmethod
    def clean_html(raw_html: str) -> str:
        """Strip HTML tags and normalize whitespace."""
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def parse_spl_json(self, spl_path: Path) -> list[dict]:
        """
        Parse a saved DailyMed JSON file, extract sections.
        Returns list of section dicts with metadata.
        """
        with open(spl_path, "r") as f:
            data = json.load(f)

        drug_name = data.get("drug_name", "unknown")
        set_id = data.get("set_id", "unknown")
        spl_data = data.get("spl_data", {})

        sections = []
        raw_sections = spl_data.get("sections", [])

        if not raw_sections:
            print(f"[Parser] No sections found in {spl_path.name}")
            return sections

        for sec in raw_sections:
            loinc_code = sec.get("loinc_code", "")
            section_name = self.SECTION_LOINC_MAP.get(loinc_code)

            if not section_name:
                continue  # Skip sections outside our taxonomy

            raw_text = sec.get("text", "")
            clean_text = self.clean_html(raw_text)

            if len(clean_text) < 20:
                continue  # Skip empty/trivial sections

            sections.append({
                "drug_name": drug_name,
                "set_id": set_id,
                "section_name": section_name,
                "loinc_code": loinc_code,
                "text": clean_text,
            })

        return sections

    def parse_all(self, raw_dir: Path) -> list[dict]:
        """Parse all raw JSON files in a directory."""
        all_sections = []
        for json_file in sorted(raw_dir.glob("*.json")):
            sections = self.parse_spl_json(json_file)
            all_sections.extend(sections)
            print(f"[Parser] {json_file.name}: {len(sections)} sections")

        print(f"[Parser] Total sections extracted: {len(all_sections)}")
        return all_sections