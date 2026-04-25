import json
import random

def investigate():
    with open('/home/ai-faculty/workspace/Q-RAG/runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_planner_eval.json', 'r') as f:
        data = json.load(f)

    empty_hop_cases = []
    
    for item in data:
        hops = item.get("hop_answers", [])
        if any(h == "" for h in hops):
            empty_hop_cases.append(item)

    print(f"Total Empty Hop cases found: {len(empty_hop_cases)}")
    
    # Print 3 random examples to understand WHY the model answered ""
    print("\n--- Why did it output empty? (3 Examples) ---")
    random.seed(42) # For reproducible analysis
    
    for item in random.sample(empty_hop_cases, 3):
        print("="*60)
        print(f"Original Q : {item['question']}")
        print(f"Sub-Qs     : {item['sub_questions']}")
        print(f"Hops Ans   : {item['hop_answers']}")
        print(f"GT         : {item['ground_truth']}")

if __name__ == "__main__":
    investigate()
