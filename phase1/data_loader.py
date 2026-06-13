from datasets import load_dataset

def load_webqsp(split="train"):
    ds = load_dataset("rmanluo/RoG-webqsp", split=split)
    samples = []
    for item in ds:
        triples = [tuple(t) for t in item.get("graph", []) if len(t) == 3]
        samples.append({
            "question": item["question"],
            "answers":  item["answer"] if isinstance(item["answer"], list) else [item["answer"]],
            "triples":  triples,
        })
    return samples

if __name__ == "__main__":
    data = load_webqsp()
    print(f"Total samples: {len(data)}\n")

    # 1. Look at one full sample
    sample = data[0]
    print("Question:", sample["question"])
    print("Answers  :", sample["answers"])
    print("Num triples:", len(sample["triples"]))
    print("First 10 triples:")
    for t in sample["triples"][:10]:
        print("  ", t)