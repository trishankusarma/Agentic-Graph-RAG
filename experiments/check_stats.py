import json
from collections import Counter

rows = [json.loads(l) for l in open("data/reports.jsonl")]

hop_type_cross = Counter()
bridge_cross = Counter()
comparison_cross = Counter()


def classify(bucket, t):
    status = t["status"]

    if status == "entity_reached" and not t["is_connected"]:
        bucket["reached_but_disconnected"] += 1
        if t["is_genuine"]:
            bucket["reached_but_disconnected_genuine"] += 1
        return  # excluded from every count below — not a real success

    bucket[status] += 1
    if status == "entity_reached" and t["is_genuine"]:
        bucket["genuine+reached"] += 1
    elif status == "entity_reached" and t["is_trivial"]:
        bucket["trivial+reached"] += 1
    elif status == "entity_unreached" and t["is_trivial"]:
        bucket["trivial+unreached"] += 1
    elif status == "entity_unreached" and not t["is_trivial"]:
        bucket["genuine-attempt+unreached"] += 1


for r in rows:
    hop_type_cross[r["hop_type"]] += 1

    if r["hop_type"] == "bridge":
        bucket = bridge_cross
        classify(bucket, r["chain_terminals"][0])

    else:
        bucket = comparison_cross
        t0, t1 = r["chain_terminals"][0], r["chain_terminals"][1]

        classify(bucket, t0)
        classify(bucket, t1)

        # comparison chains are built from a single gold title (one segment),
        # so is_connected() and terminal.status "reached" should always agree
        # — this should stay at 0. If it isn't, something upstream changed.
        if t0["status"] == "entity_reached" and not t0["is_connected"]:
            bucket["UNEXPECTED_disconnected_reached_t0"] += 1
        if t1["status"] == "entity_reached" and not t1["is_connected"]:
            bucket["UNEXPECTED_disconnected_reached_t1"] += 1

        status0, status1 = t0["status"], t1["status"]
        bucket[f"{status0} × {status1}"] += 1

        extended = (
            f"{status0} | G={'Y' if t0['is_genuine'] else 'N'} "
            f"T={'Y' if t0['is_trivial'] else 'N'}"
            "  <->  "
            f"{status1} | G={'Y' if t1['is_genuine'] else 'N'} "
            f"T={'Y' if t1['is_trivial'] else 'N'}"
        )
        bucket[extended] += 1


def print_counter(title, counter):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in counter.most_common():
        print(f"{k:<55} {v:>4}")


print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)
print(f"Total samples           : {len(rows)}")

print("\nHop Type Distribution")
print("---------------------")
for k, v in hop_type_cross.items():
    print(f"{k:<12}: {v:>4} ({v/len(rows):6.2%})")

print_counter("Bridge Statistics", bridge_cross)
print_counter("Comparison Statistics", comparison_cross)

text_grounded_not_connected = sum(
    1 for r in rows
    if r["text_grounded"] and r["failure_mode"] != "connected"
)

print("\nOther Statistics")
print("----------------")
print(f"Text-grounded but not connected     : {text_grounded_not_connected}")
print(
    "Reached-but-disconnected (bridge)   :",
    bridge_cross["reached_but_disconnected"],
    "(genuine:", bridge_cross["reached_but_disconnected_genuine"], ")"
)
print("=" * 70)