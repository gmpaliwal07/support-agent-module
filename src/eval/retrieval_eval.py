import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from query import HistoricalCaseRetriever  # noqa: E402
 
TEST_CASES = [
    {
        "query": "My driver cancelled the ride and I still got charged a fee",
        "relevant_in_top3": 3,
        "relevant_in_top10": 10,
    },
    {
        "query": "I left my phone in the car",
        "relevant_in_top3": 3,
        "relevant_in_top10": 9,
    },
    {
        "query": "The driver was really rude to me",
        "relevant_in_top3": 3,
        "relevant_in_top10": 8,
    },
    {
        "query": "I ordered food and never got a refund for missing items",
        "relevant_in_top3": 3,
        "relevant_in_top10": 10,
    },
    {
        "query": "my credit card keeps getting declined when I try to pay",
        "relevant_in_top3": 3,
        "relevant_in_top10": 9,
    },
    {
        "query": "I can't leave a tip for my driver, the option is missing",
        "relevant_in_top3": 3,
        "relevant_in_top10": 5,
    },
    {
        "query": "I'm getting an error saying my phone number is already registered",
        "relevant_in_top3": 2,
        "relevant_in_top10": 9,
    },
]
 
def main() -> None:
    retriever = HistoricalCaseRetriever()
 
    precisions, recalls = [], []
    for case in TEST_CASES:
        precision_at_3 = case["relevant_in_top3"] / 3
        recall_at_3 = case["relevant_in_top3"] / case["relevant_in_top10"]
        precisions.append(precision_at_3)
        recalls.append(recall_at_3)
        print(f"{case['query'][:60]:<62} P@3={precision_at_3:.2f}  approx_R@3={recall_at_3:.2f}")
 
    print(f"\nmean precision@3: {sum(precisions)/len(precisions):.2f}")
    print(f"mean approx recall@3: {sum(recalls)/len(recalls):.2f}")

if __name__ == "__main__":
    main()