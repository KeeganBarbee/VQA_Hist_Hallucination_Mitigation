import os
import json
from PIL import Image
from Dentist.model.my_llava.my_llava_verifier import LLaVA_Verifier
from Dentist.model.Qwen2VL.Qwen2VL_Verifier import Qwen2VL_Verifier

BASE_DIR = "/home/kbarbee2/.cache/huggingface/hub/datasets--ReadingTimeMachine--visual_qa_histograms/snapshots/f7ba35f3751cf2b27c943ccb1d313424897c913b/example_hists"
IMG_DIR = os.path.join(BASE_DIR, "imgs/imgs")
JSON_DIR = os.path.join(BASE_DIR, "jsons")

def extract_vqa(sample):
    qa_list = []
    for level in sample["VQA"].values():
        for section in level.values():
            for qtype in section:
                for plot in section[qtype].values():
                    question = plot["question"]
                    answer_dict = plot["A"]
                    answer = list(answer_dict.values())[0]
                    if isinstance(answer, dict):
                        answer = list(answer.values())[0]
                    qa_list.append({
                        "question": question,
                        "answer": answer,
                        "type": qtype
                    })
    return qa_list

def build_dataset():
    dataset = []
    for json_file in os.listdir(JSON_DIR):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join(JSON_DIR, json_file)
        img_path = os.path.join(IMG_DIR, json_file.replace(".json", ".jpeg"))
        if not os.path.exists(img_path):
            continue
        with open(json_path) as f:
            sample_json = json.load(f)
        if isinstance(sample_json, str):
            sample_json = json.loads(sample_json)
        qa_pairs = extract_vqa(sample_json)
        dataset.append({
            "image_path": img_path,
            "qa_pairs": qa_pairs
        })
    return dataset

def run_experiment(verifier, image, question):
    # FIX: no results reference here — just return answers
    try:
        baseline_answer = verifier.ask_model(image, question, use_image=True)
        revised_answer = verifier.verify_loop(
            original_image=image,
            original_q=question,
            original_a=baseline_answer
        )
        return baseline_answer, revised_answer
    except Exception as e:
        print(f"Error: {e}")
        return "ERROR", "ERROR"

def main():
    limited_cnt = 3
    max_samples = None  # set to None for full run

    qwen_verifier = Qwen2VL_Verifier(limited_cnt=limited_cnt)
    llava_verifier = LLaVA_Verifier(limited_cnt=limited_cnt)

    dataset = build_dataset()
    if max_samples:
        dataset = dataset[:max_samples]

    results = []  # FIX: initialize results here

    for sample in dataset:
        image = Image.open(sample["image_path"]).convert("RGB")
        for qa in sample["qa_pairs"]:
            question = qa["question"]
            ground_truth = qa["answer"]
            qtype = qa["type"]
            for name, verifier in [("Qwen2VL", qwen_verifier), ("LLaVA", llava_verifier)]:
                baseline, revised = run_experiment(verifier, image, question)
                results.append({
                    "model": name,
                    "image_path": sample["image_path"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "type": qtype,
                    "baseline_answer": baseline,
                    "revised_answer": revised
                })
                print(f"\n--- {name} | {qtype} ---")
                print(f"Q:        {question}")
                print(f"GT:       {ground_truth}")
                print(f"Baseline: {baseline}")
                print(f"Revised:  {revised}")

                # FIX: incremental save here in main() where results exists
                with open("results.json", "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} results saved to results.json")

if __name__ == '__main__':
    main()
"""
Inspired by Chang, Yue, et al. "A unified hallucination mitigation framework for large vision-language models." arXiv preprint arXiv:2409.16494 (2024).
"""
