import torch
from PIL import Image
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor
import re
from collections import defaultdict

class LLaVAFrontierSelector:
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", device="cuda"):
        self.device = device
        print(f"Loading model from {model_path} ...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None
        )
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, revision='a272c74')
        print("llava-1.5-7b-hf loaded.")

    def compare(self, img1: np.ndarray, img2: np.ndarray, question: str, max_new_tokens=48) -> str:
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)
        prompt = (
            "You are an intelligent agent exploring an indoor environment. "
            "Given two possible frontiers (A and B)—each representing an observation of an unexplored region that could potentially provide new information for answering the question—choose the frontier you would prefer to explore further, and explain why you selected that direction."
            "Always use the following response format:\n"
            "Answer: [A or B]\nReason: <your explanation>\n"
            "\n"
            "A: (shows a pillow with a floral pattern on a black couch)\n"
            "B: (shows a pillow with a plain color on a brown couch)\n"
            "Question: What color pattern is on the pillow on the black couch?\n"
            "Answer: A\nReason: Direction A directly shows a patterned pillow on a black couch, which helps answer the question.\n"
            "---\n"
            "A: (shows a hallway with a large mirror mounted on the wall)\n"
            "B: (shows a bedroom with no visible mirror)\n"
            "Question: Where is the full body mirror?\n"
            "Answer: A\nReason: Direction A reveals a full body mirror in the hallway, making it the best direction for answering the question.\n"
            "---\n"
            "A: (shows a dining table with lots of clutter: plates, cups, food)\n"
            "B: (shows a clean dining table with only a vase in the center)\n"
            "Question: Is there space on the dining table to work on my laptop?\n"
            "Answer: B\nReason: Direction B shows a clear table with enough space for a laptop.\n"
            "---\n"
            "A: (shows a staircase leading upwards to a second floor)\n"
            "B: (shows only the ground floor with no stairs in sight)\n"
            "Question: How many stories does this house have?\n"
            "Answer: A\nReason: Direction A reveals a staircase, suggesting the house has more than one story.\n"
            "---\n"
            "A:\n<image>\n"
            "B:\n<image>\n"
            f"Question: {question.strip()}\n"
            "Please answer in the following format:\n"
            "Answer: [A or B]\nReason: <your explanation>\n"
            "Answer:"
        )

        inputs = self.processor(
            images=[pil_img1, pil_img2],
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        output = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return output.strip()

    @staticmethod
    def parse_answer_reason(output: str):
        answers = re.findall(r"Answer:\s*([AB])", output, re.IGNORECASE)
        reasons = re.findall(r"Reason:\s*([^\n]*)", output, re.IGNORECASE)
        answer = answers[-1].strip().upper() if answers else None
        reason = reasons[-1].strip() if reasons else None
        return answer, reason

    def select(self, img1: np.ndarray, img2: np.ndarray, question: str):
        result = self.compare(img1, img2, question)
        answer, reason = self.parse_answer_reason(result)
        if answer == 'A':
            idx = 0
        elif answer == 'B':
            idx = 1
        else:
            idx = -1
        return idx, answer, reason, result

def pairwise_voting_frontier(frontier_dict, question, selector):
    """
    All-pair (round robin) voting with both A/B and B/A.
    3 points if one frontier wins both orders, 1-1 if split win, 0 for both lose.
    """
    indexes = sorted(list(frontier_dict.keys()))
    score = defaultdict(int)
    log_trace = []

    n = len(indexes)
    for i in range(n):
        for j in range(i+1, n):
            idx_i = indexes[i]
            idx_j = indexes[j]
            img_i = frontier_dict[idx_i]
            img_j = frontier_dict[idx_j]

            # idx_i as A, idx_j as B
            print(f"\n[Compare] {idx_i} (A) vs {idx_j} (B)")
            win_1, ans_1, reason_1, _ = selector.select(img_i, img_j, question)
            if win_1 == 0:
                print(f"Winner: {idx_i} (A), Reason: {reason_1}")
            elif win_1 == 1:
                print(f"Winner: {idx_j} (B), Reason: {reason_1}")
            else:
                print("Unclear winner.")
            log_trace.append((idx_i, idx_j, 'A', win_1, ans_1, reason_1))

            # idx_j as A, idx_i as B
            print(f"[Compare] {idx_j} (A) vs {idx_i} (B)")
            win_2, ans_2, reason_2, _ = selector.select(img_j, img_i, question)
            if win_2 == 0:
                print(f"Winner: {idx_j} (A), Reason: {reason_2}")
            elif win_2 == 1:
                print(f"Winner: {idx_i} (B), Reason: {reason_2}")
            else:
                print("Unclear winner.")
            log_trace.append((idx_j, idx_i, 'A', win_2, ans_2, reason_2))

            # Scoring
            # idx_i wins both directions
            if win_1 == 0 and win_2 == 1:
                score[idx_i] += 3
                print(f"--> {idx_i} wins both rounds: +3")
            # idx_j wins both directions
            elif win_1 == 1 and win_2 == 0:
                score[idx_j] += 3
                print(f"--> {idx_j} wins both rounds: +3")
            # tie: each wins one (or any unclear result)
            else:
                score[idx_i] += 1
                score[idx_j] += 1
                print(f"--> Tie: both {idx_i} and {idx_j} +1")

    max_score = max(score.values())
    best_idxs = [idx for idx, s in score.items() if s == max_score]

    print(f"\n=== Final voting result ===")
    for idx in indexes:
        print(f"Frontier {idx}: score = {score[idx]}")
    print(f"Best index(s): {best_idxs}")

    return best_idxs, score, log_trace

if __name__ == "__main__":
    img_paths = {
        0: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/4o/results_41_4o/exp_eval_aeqa/4dbd213e-56cd-481a-8ff5-ed9a8d636dbc/frontier/8_4.png",
        1: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/4o/results_41_4o/exp_eval_aeqa/4dbd213e-56cd-481a-8ff5-ed9a8d636dbc/frontier/8_5.png",
        2: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/4o/results_41_4o/exp_eval_aeqa/4dbd213e-56cd-481a-8ff5-ed9a8d636dbc/frontier/8_6.png"}
    question = "Is the light above the sink turned on?"

    frontier_dict = {idx: np.array(Image.open(path).convert("RGB")) for idx, path in img_paths.items()}

    selector = LLaVAFrontierSelector(device="cuda")
    best_idxs, score_dict, log_trace = pairwise_voting_frontier(frontier_dict, question, selector)

    print("\n==== Final selected winner(s): ====")
    print(best_idxs)