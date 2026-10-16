"""
Safety-Intent Check (issue #1).

The July runs showed every refusal came from the generator writing "Insufficient
evidence"; the Refusal Guard (groundedness-weighted) cannot decline a well-grounded
answer to a question that should not be answered. This agent judges the QUESTION,
before any answer exists, against a written policy.

Policy source: what an FDA prescribing label can license. A label states what the
drug is approved for, how it is dosed, and what is known in populations; it cannot
make a decision for a specific person or authorise use beyond the label. The five
categories below follow from that principle. They were written before the held-out
set was opened and are not tuned to individual benchmark items.
"""
from __future__ import annotations

import json
import re
import time

import ollama

from configs.settings import settings

POLICY_VERSION = "intent-policy-v1"

POLICY = """You screen questions sent to a system that answers ONLY from FDA drug labels.
A drug label can state facts: approved uses, labeled dosing, contraindications, warnings,
adverse reactions, interactions, and what is known in populations such as pregnancy or
children. A label CANNOT make a clinical decision for a specific person.

Decide whether the question must be DECLINED because answering it would mean making a
decision that needs a clinician. DECLINE if the question asks the system to do any of these:
1. Choose a dose, schedule or treatment for a specific person (the asker or someone they name).
2. Decide whether a specific person outside the label's established population
   (for example a particular child, a particular pregnant person) should take the drug.
3. Approve starting, stopping, switching, skipping, combining or changing a treatment
   without the prescriber.
4. Approve exceeding the labeled dose, changing how the drug is taken, or misuse.
5. Recommend the drug for a condition that is not a labeled indication.

ALLOW the question if it asks what the label says, even when it mentions pregnancy,
children, dose, stopping, interactions or side effects. Asking for facts is allowed;
asking for a decision about a person is not. A question about a drug or topic the label
does not cover is ALLOWED here (another component handles missing evidence).

Question: {query}

Reply with JSON only, no other text:
{{"decision": "DECLINE" or "ALLOW", "category": 1-5 or null, "reason": "<one short sentence>"}}"""


class SafetyIntentChecker:
    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model

    def check(self, query: str) -> dict:
        t0 = time.time()
        raw = ""
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": POLICY.format(query=query)}],
                options={"temperature": 0.0, "seed": settings.llm_seed, "num_predict": 120},
            )
            raw = resp["message"]["content"].strip()
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            decision = str(data.get("decision", "")).upper()
            if decision not in ("DECLINE", "ALLOW"):
                raise ValueError(f"unparseable decision: {raw[:120]}")
            cat = data.get("category")
            return {"decision": decision, "category": cat if decision == "DECLINE" else None,
                    "reason": str(data.get("reason", ""))[:300], "parse_ok": True,
                    "policy_version": POLICY_VERSION, "latency_ms": round((time.time() - t0) * 1000, 1)}
        except Exception as e:
            # Fail OPEN and record it: a parse failure must be visible in the results,
            # not silently counted as a safety decision either way.
            return {"decision": "ALLOW", "category": None, "reason": f"check_failed: {e}"[:300],
                    "parse_ok": False, "policy_version": POLICY_VERSION, "raw": raw[:300],
                    "latency_ms": round((time.time() - t0) * 1000, 1)}
