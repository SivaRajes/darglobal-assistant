"""Evaluate hybrid retrieval against fixed source-level cases (no LLM call)."""
from app.main import lexical_candidates, retrieve
from app.metadata import classify_query


CASES = [
    {
        "question": "What does DarGlobal say about its partnership with OMRAN Group?",
        "expected": ("omran-group-announce-partnership-to-develop-aida",),
    },
    {
        "question": "What is the AIDA project and where is it located?",
        "expected": ("omran-group-announce-partnership-to-develop-aida",),
    },
    {
        "question": "What information does Wasalt provide about Shams Al Riyadh?",
        "expected": ("shams-al-riyadh-100325",),
    },
    {
        "question": "What property details are listed for Raseen Khuzam on Wasalt?",
        "expected": ("raseen-khuzam-villas-for-sale-100548",),
    },
    {
        "question": "Which DarGlobal projects are in Dubai?",
        "expected": (
            "trump-international-hotel-tower-dubai",
            "w-residences-dubai-downtown",
            "urban-oasis-water-front-residential-tower-in-dubai",
            "partners/missoni",
        ),
        "forbidden_top3": ("qetaifan-projects-unveil-premium-sea-front-residences-with-interiors-designed-by-elie-saab-at-expo-2020-dubai",),
    },
]


def main():
    totals = {"BM25": {"hits": 0, "rr": 0.0}, "Hybrid": {"hits": 0, "rr": 0.0}}
    for case in CASES:
        intent = classify_query(case["question"])
        print(f"\n{case['question']} | intent={intent}")
        for label, results in (
            ("BM25", lexical_candidates(case["question"], limit=10)),
            ("Hybrid", retrieve(case["question"], limit=10)),
        ):
            urls = [item.get("url", "").lower() for item in results]
            rank = next((i + 1 for i, url in enumerate(urls)
                         if any(expected in url for expected in case["expected"])), None)
            forbidden = [fragment for fragment in case.get("forbidden_top3", ())
                         if any(fragment in url for url in urls[:3])]
            passed = rank is not None and not forbidden
            totals[label]["hits"] += int(passed)
            totals[label]["rr"] += 1 / rank if rank else 0.0
            print(f"  {label}: {'PASS' if passed else 'FAIL'} | rank={rank or '-'}")
            for item in results[:3]:
                meta = item.get("page_metadata", {})
                print(f"    {item.get('title')} | {meta.get('page_type')} | project_city={meta.get('project_city')} | {item.get('url')}")
            if forbidden:
                print(f"    False-positive event-location result in top 3: {forbidden}")
    for label, stats in totals.items():
        print(f"{label}: Recall@10 {stats['hits']}/{len(CASES)}; MRR {stats['rr'] / len(CASES):.3f}")
    # BM25 is the baseline being compared, so it may miss the target case; the
    # evaluation succeeds when the hybrid system passes the fixed set.
    return 0 if totals["Hybrid"]["hits"] == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
