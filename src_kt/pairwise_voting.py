import torch
from PIL import Image
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor
import re
from collections import defaultdict

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

    def generate_caption(self, img: np.ndarray, max_new_tokens=80):
        def extract_last_caption(text):
            splits = text.strip().split("Caption:")
            if len(splits) > 1:
                return splits[-1].strip()
            else:
                return text.strip()
        pil_img = Image.fromarray(img)
        prompt = (
            "You are an embodied agent exploring an indoor environment. "
            "Given the following image, your task is to generate a detailed and structured caption. "
            "Mention key objects, their spatial relationships, the overall layout, and any unique or salient details. "
            "Use concise sentences. Follow the examples below.\n\n"
            "Examples:\n"
            "Image: <example image>\n"
            "Caption: A living room with a brown sofa facing a TV on a wooden stand. A small coffee table is in front of the sofa. There is a large window with curtains on the right. A floor lamp stands next to the sofa.\n\n"
            "Image: <example image>\n"
            "Caption: A kitchen area with a white refrigerator to the left and a microwave oven on the countertop. A dining table with four chairs is positioned in the center. There are cabinets above the counter. A door leads to another room in the background.\n\n"
            "Image: <image>\n"
            "Caption:"
        )
        inputs = self.processor(
            images=pil_img,
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return extract_last_caption(caption)


    def compare(self, caption1, img1: np.ndarray, caption2, img2: np.ndarray, question: str, max_new_tokens=48) -> str:

        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)

        # 2. 把caption和图片一起写入prompt
        prompt = (
            "You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose a direction to further explore."
            "Given two possible frontiers (A and B)—each representing an observation of an unexplored region that could potentially provide new information for answering the question—choose the frontier you would prefer to explore further, and explain why you selected that direction. When making your decision, carefully consider both the provided caption and the corresponding image for each direction."
            "Always use the following response format:\n"
            "Answer: [A or B]\nReason: <your explanation>\n"
            "\n"
            "A: A brown couch with a plain color pillow. A wooden side table is next to the couch. The background shows a white wall and a small plant. <example image>\n"
            "B: A black couch with a pillow that has a floral pattern. A glass coffee table sits in front of the couch. The living room has a large window with curtains on the right. <example image>\n"
            "Question: What color pattern is on the pillow on the black couch?\n"
            "Answer: B\nReason: Direction B describes a floral patterned pillow on a black couch, directly addressing the question.\n"
            "---\n"
            "A: A hallway with a large mirror mounted on the wall. A shoe rack is below the mirror, and the hallway leads to another room. <example image>\n"
            "B: A bedroom with a bed, a bedside table, and no visible mirror. The walls are painted light gray. <example image>\n"
            "Question: Where is the full body mirror?\n"
            "Answer: A\nReason: Direction A clearly mentions a large mirror in the hallway, making it the best choice.\n"
            "---\n"
            "A: A dining table cluttered with plates, cups, and various food items. There are several chairs around the table and a pendant lamp above. <example image>\n"
            "B: A clean dining table with only a vase placed in the center. The table is empty otherwise, with four neatly arranged chairs. <example image>\n"
            "Question: Is there space on the dining table to work on my laptop?\n"
            "Answer: B\nReason: Direction B shows a clear, uncluttered table with enough space for a laptop.\n"
            "---\n"
            "A: A staircase leading upward to a second floor. There is a handrail on the right, and the steps are carpeted. <example image>\n"
            "B: The ground floor area with no stairs in sight. The space is open with a rug and a small table. <example image>\n"
            "Question: How many stories does this house have?\n"
            "Answer: A\nReason: Direction A mentions a staircase, indicating the house has more than one story.\n"
            "---\n"
            f"A: {caption1}\n<image>\n"
            f"B: {caption2}\n<image>\n"
            f"Question: {question.strip()}\n"
            "Please answer in the following format:\n"
            "Answer: [A or B]\nReason: <your explanation>\n"
            "Answer:"
        )


        # 3. 依然喂入两张图片
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

    def select(self, caption1, img1: np.ndarray, caption2, img2: np.ndarray, question: str):


        result = self.compare(caption1, img1, caption2, img2, question)
        answer, reason = self.parse_answer_reason(result)
        if answer == 'A':
            idx = 0
        elif answer == 'B':
            idx = 1
        else:
            idx = -1
        return idx, answer, reason, result

def pairwise_voting_frontier(frontier_dict: dict, question: str, selector: LLaVAFrontierSelector):
    """
    All-pair voting: each pair compared in both orders.
    Win both orders -> +3, split -> +1 each, otherwise +1 each.
    Returns (winner_idx, scores, log).
    自动为所有frontier批量生成caption，后续对比直接复用。
    """
    indexes = sorted(frontier_dict.keys())
    scores = defaultdict(int)
    log = []

    # 1. 批量生成所有frontier的caption
    print("[INFO] Generating captions for all frontiers ...")
    captions = {}
    for idx in indexes:
        captions[idx] = selector.generate_caption(frontier_dict[idx])
        print(f"Frontier {idx}: {captions[idx]}")
    print("[INFO] Caption generation completed.\n")

    # 2. 两两比较时直接用caption，不重复生成
    for i in range(len(indexes)):
        for j in range(i+1, len(indexes)):
            idx_i, idx_j = indexes[i], indexes[j]
            img_i, img_j = frontier_dict[idx_i], frontier_dict[idx_j]
            cap_i, cap_j = captions[idx_i], captions[idx_j]

            print(f"[Compare] {idx_i} (A) vs {idx_j} (B)")
            win1, ans1, rea1, _ = selector.select(cap_i, img_i, cap_j, img_j, question)
            log.append((idx_i, idx_j, 'A', win1, ans1, rea1))

            print(f"[Compare] {idx_j} (A) vs {idx_i} (B)")
            win2, ans2, rea2, _ = selector.select(cap_j, img_j, cap_i, img_i, question)
            log.append((idx_j, idx_i, 'A', win2, ans2, rea2))

            # scoring
            if win1 == 0 and win2 == 1:
                scores[idx_i] += 3
            elif win1 == 1 and win2 == 0:
                scores[idx_j] += 3
            else:
                scores[idx_i] += 1
                scores[idx_j] += 1

    max_score = max(scores.values())
    winners = [idx for idx, sc in scores.items() if sc == max_score]
    final_winner = winners[0] if len(winners) == 1 else -1

    print("Final scores:", dict(scores))
    print("Winner:", final_winner if final_winner >= 0 else "tie")
    return final_winner, scores, log







if __name__ == "__main__":
    img_paths = {
        0: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/result_184/"
           "exp_eval_aeqa/f776a834-1e21-4442-8834-18b6f9d6cfad/frontier/0_1.png",
        1: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/result_184/"
           "exp_eval_aeqa/f776a834-1e21-4442-8834-18b6f9d6cfad/frontier/0_2.png",
        2: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/result_184/"
           "exp_eval_aeqa/f776a834-1e21-4442-8834-18b6f9d6cfad/frontier/0_0.png",
        3: "/home/wiss/zhang/projects/openeqa/aeqa/baseline/result_184/"
           "exp_eval_aeqa/f776a834-1e21-4442-8834-18b6f9d6cfad/frontier/0_3.png",
    }
    question = "Where is the orange painting?"

    # frontier_dict = {
    #     idx: np.array(Image.open(path).convert("RGB"))
    #     for idx, path in img_paths.items()
    # }

    # selector = LLaVAFrontierSelector(device="cuda")
    # best_idx, score_dict, log_trace = pairwise_voting_frontier(frontier_dict, question, selector)

    # print("\n==== Final selected winner(s): ====")
    # print(best_idx)
    # print("log_trace:", log_trace)
    # print("score_dict:", score_dict)
