"""Frozen local evidence source (v2: docs/v2_design.md section 4).

v1 was an exact lookup from evidence_doc_id to the one record holding the gold answer in a fixed
template. CALL therefore scored reward_call = 1.000 on every dev prompt, so answering could never
win, 93.2% of prompts were call-beneficial and the routing decision had no content.

v2 keeps the "no learned retriever" property - retrieval is deterministic and frozen, so retrieval
noise can never explain a difference between methods - but makes CALL imperfect:

  * top-k lexical retrieval over the WHOLE record pool, not an id lookup, so near-miss records for
    other pseudonyms appear alongside the gold one and the model has to select
  * a declared share of prompts (noise_rate) has the gold record withheld from the pool, so the
    evidence block is all distractors and the honest action is to fail or abstain
  * record phrasing varies per fact (built in build_facts.evidence_text), so the answer is not a
    verbatim copy at a constant offset

Everything is a deterministic function of (prompt_id, index contents, k, noise_rate), which are all
folded into `version`. Two runs of any branch see byte-identical evidence.
"""
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")

# retrieval query terms that carry no entity information; dropped so scoring is driven by the
# pseudonym and the relation words rather than by question boilerplate
_STOP = {"what", "which", "who", "whom", "whose", "where", "when", "is", "are", "was", "were", "the",
         "a", "an", "of", "in", "on", "at", "to", "for", "by", "does", "do", "did", "has", "have",
         "name", "tell", "me", "and", "that", "this", "it", "its", "his", "her", "their", "from",
         "kind", "known", "come", "mark", "marks", "give", "gives", "listed", "list"}


def _toks(s):
    return [w for w in _WORD.findall(s.lower()) if w not in _STOP]


class EvidenceIndex:
    """Deterministic frozen retriever over facts/evidence.jsonl."""

    def __init__(self, facts_dir, k=3, noise_rate=0.20):
        path = Path(facts_dir) / "evidence.jsonl"
        raw = path.read_bytes()
        self.k, self.noise_rate = k, noise_rate
        self.records, self.order = {}, []
        for line in raw.decode("utf-8").splitlines():
            r = json.loads(line)
            assert r["evidence_doc_id"] not in self.records, f"duplicate evidence id {r['evidence_doc_id']}"
            self.records[r["evidence_doc_id"]] = r
            self.order.append(r["evidence_doc_id"])
        # the index version must change whenever retrieved text could change
        h = hashlib.sha256(raw).hexdigest()[:16]
        self.version = f"evidence_v2_{h}_k{k}_n{int(round(noise_rate * 100)):02d}"
        self._build_postings()

    # ------------------------------------------------------------------ index
    def _build_postings(self):
        self.postings = defaultdict(list)
        self.doc_len = {}
        for did in self.order:
            tf = defaultdict(int)
            for w in _toks(self.records[did]["text"]):
                tf[w] += 1
            self.doc_len[did] = sum(tf.values()) or 1
            for w, c in tf.items():
                self.postings[w].append((did, c))
        n = len(self.order)
        self.idf = {w: math.log(1.0 + n / len(p)) for w, p in self.postings.items()}
        self.avg_len = sum(self.doc_len.values()) / max(n, 1)

    def _score(self, query_tokens, exclude=None):
        """BM25 over the record pool. `exclude` drops one record (the noise case)."""
        k1, b = 1.5, 0.75
        acc = defaultdict(float)
        for w in query_tokens:
            if w not in self.postings:
                continue
            idf = self.idf[w]
            for did, c in self.postings[w]:
                if did == exclude:
                    continue
                dl = self.doc_len[did]
                acc[did] += idf * c * (k1 + 1) / (c + k1 * (1 - b + b * dl / self.avg_len))
        return acc

    # ------------------------------------------------------------------ query
    def is_noisy(self, prompt_id):
        """Deterministic per-prompt decision to withhold the gold record (never random at call time)."""
        if self.noise_rate <= 0:
            return False
        h = int(hashlib.sha1(f"{self.version}|{prompt_id}".encode()).hexdigest(), 16)
        return (h % 10_000) < round(self.noise_rate * 10_000)

    def retrieve(self, question, prompt_id, gold_doc_id=None):
        """Return (evidence_text, meta). The text is the concatenated top-k records."""
        noisy = self.is_noisy(prompt_id)
        acc = self._score(_toks(question), exclude=gold_doc_id if noisy else None)
        # stable ordering: score desc, then the frozen file order
        rank = {d: i for i, d in enumerate(self.order)}
        top = sorted(acc.items(), key=lambda kv: (-kv[1], rank[kv[0]]))[: self.k]
        dids = [d for d, _ in top]
        text = " ".join(self.records[d]["text"] for d in dids)
        meta = {"retrieved_doc_ids": ",".join(dids), "evidence_noisy": int(noisy),
                "gold_retrieved": int(gold_doc_id in dids) if gold_doc_id else 0,
                "gold_rank": dids.index(gold_doc_id) if gold_doc_id in dids else -1}
        return text, meta

    def lookup(self, evidence_doc_id):
        """Raw single record. Kept for audits and the oracle-evidence ablation, NOT for scoring runs."""
        return self.records[evidence_doc_id]["text"]

    def __len__(self):
        return len(self.records)
