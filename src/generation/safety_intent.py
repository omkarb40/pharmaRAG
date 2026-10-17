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

POLICY_VERSION = "intent-policy-v2"
# Change log (dev set only; held-out set unopened):
#   v1 -> v2: v1 declined 12/58 answerable dev questions. All 12 were INFORMATION requests
#   misread as decisions: comparisons of two labels, population-level questions
#   ("safe in pregnancy?"), the labeled dose, and first-person questions the label answers
#   with a general fact. v2 states the information-vs-permission distinction explicitly
#   and lists the allowed forms first. No drug- or item-specific rules were added.

POLICY = """You screen questions sent to a system that answers ONLY from FDA drug labels.
The system may give INFORMATION that a label states. It must not give PERMISSION or make a
CHOICE that only a clinician can make for a particular person.

ALLOW (information) - the question asks what is known or stated, for example:
- the labeled or recommended dose, schedule, route or titration of a drug
- whether a drug has a contraindication, warning or interaction for a condition or another drug
- what is known in a population in general (pregnancy, breastfeeding, children, older adults,
  liver or kidney impairment), even when phrased as "is it safe during pregnancy?"
- what happens if a dose is missed or a drug is stopped, as the label describes it
- a comparison of what two drugs' labels state (risks, monitoring, route, requirements)
- any of the above phrased in the first person ("Can I take this if I have liver problems?"),
  when a general labeled fact answers it

DECLINE (permission or choice) - the question asks the system to decide for a particular person:
1. choose or adjust a dose or schedule for that person beyond the labeled dosing
2. decide whether that particular person (a named child, a particular pregnancy, a relative
   with stated circumstances) should take or continue the drug
3. approve stopping, skipping, switching, combining or changing treatment without the prescriber
   ("Can I just stop...", "Is it okay if I switch myself...")
4. approve exceeding the labeled dose, taking it differently than labeled, or misuse
5. recommend the drug for a condition that is not a labeled indication

Test: if a correct answer could be a quotation from the label, ALLOW. If a correct answer
would have to be "yes, you should" or "take this amount" for that person, DECLINE.
Questions about drugs or topics the label does not cover are ALLOWED here; another
component handles missing evidence.

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
