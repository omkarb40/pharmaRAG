'''
DailyMed API client for fetching drug information. This module provides functions to interact with the DailyMed API, allowing you to retrieve drug labels, active ingredients, and other relevant information. The client is designed to handle API requests, parse responses, and manage errors effectively. It serves as a crucial component for integrating DailyMed data into the application, enabling users to access up-to-date drug information for various use cases such as research, analysis, and decision-making.
Fetches Structured Product Label (SPL) for drugs in our List.
'''
import requests
import json
import time
from pathlib import Path
from typing import Optional

from configs.settings import settings

class DailyMedClient:
    """Client for the DailyMed REST API v2."""

    # Relevant SPL sections for drug safety QA
    SECTION_TAXONOMY = {
        "34067-9": "indications_and_usage",
        "34070-3": "contraindications",
        "43685-7": "warnings_and_precautions",
        "34084-4": "adverse_reactions",
        "34068-7": "dosage_and_administration",
        "34073-7": "drug_interactions",
        "43684-0": "use_in_specific_populations",
        "34071-1": "warnings",              # older format
        "42232-9": "boxed_warning",
    }

    def __init__(self):
        self.base_url = settings.dailymed_base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def search_drug(self, drug_name: str) -> Optional[dict]:
        """Search DailyMed for a drug by name, return first SPL result."""
        url = f"{self.base_url}/spls.json"
        params = {"drug_name": drug_name, "page": 1, "pagesize": 1}

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("data"):
                spl = data["data"][0]
                return {
                    "set_id": spl.get("setid"),
                    "title": spl.get("title"),
                    "published_date": spl.get("published_date"),
                }
        except requests.RequestException as e:
            print(f"[DailyMed] Error searching for '{drug_name}': {e}")

        return None

    def fetch_spl_sections(self, set_id: str) -> dict:
        """Fetch all sections of an SPL by set_id."""
        url = f"{self.base_url}/spls/{set_id}.json"

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[DailyMed] Error fetching SPL '{set_id}': {e}")
            return {}

    def fetch_drug_label(self, drug_name: str) -> Optional[dict]:
        """
        End-to-end: search for drug, fetch its SPL, return structured data.
        """
        result = self.search_drug(drug_name)
        if not result:
            print(f"[DailyMed] No results for '{drug_name}'")
            return None

        set_id = result["set_id"]
        print(f"[DailyMed] Found '{drug_name}' → set_id: {set_id}")

        spl_data = self.fetch_spl_sections(set_id)
        if not spl_data:
            return None

        return {
            "drug_name": drug_name,
            "set_id": set_id,
            "title": result["title"],
            "published_date": result["published_date"],
            "spl_data": spl_data,
        }

    def fetch_all_drugs(self, drug_list_path: Path, output_dir: Path,
                        delay: float = 1.0):
        """
        Fetch SPL data for all drugs in a CSV file.
        Saves raw JSON per drug to output_dir.
        """
        import csv

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(drug_list_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_name = row["generic_name"]
                out_file = output_dir / f"{drug_name.replace(' ', '_')}.json"

                if out_file.exists():
                    print(f"[DailyMed] Skipping '{drug_name}' (already fetched)")
                    continue

                label = self.fetch_drug_label(drug_name)
                if label:
                    with open(out_file, "w") as fout:
                        json.dump(label, fout, indent=2)
                    print(f"[DailyMed] Saved: {out_file}")
                else:
                    print(f"[DailyMed] FAILED: {drug_name}")

                time.sleep(delay)  # Rate limiting