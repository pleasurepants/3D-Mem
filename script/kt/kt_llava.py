import torch
from PIL import Image
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor
import re

class LLaVAFrontierSelector:
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """
        Initialize the LLaVA model and multimodal processor.
        """
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
        # Few-shot: only use (example image) for demonstration, main question uses <image>
        prompt = (
            # Example 1
            "A: (example image)\n"
            "B: (example image)\n"
            "Question: If I don't want to use the dining table, where can I sit and eat?\n"
            "Please answer in the following format:\n"
            "Answer: [A or B]\nReason: <your explanation>\n"
            "Answer: A\nReason: The bar counter is suitable for eating meals.\n"
            "---\n"
            # Example 2
            "A: (example image)\n"
            "B: (example image)\n"
            "Question: If I want to take a rest, which place should I choose?\n"
            "Please answer in the following format:\n"
            "Answer: [A or B]\nReason: <your explanation>\n"
            "Answer: B\nReason: The sofa is more comfortable for resting.\n"
            "---\n"
            # Main question
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
        """
        提取最后一组Answer/Reason作为最终答案。
        """
        # 找到所有出现的Answer:和Reason:
        answers = re.findall(r"Answer:\s*([AB])", output, re.IGNORECASE)
        reasons = re.findall(r"Reason:\s*([^\n]*)", output, re.IGNORECASE)
        answer = answers[-1].strip().upper() if answers else None
        reason = reasons[-1].strip() if reasons else None
        return answer, reason

    def select(self, img1: np.ndarray, img2: np.ndarray, question: str):
        """
        Choose between two frontier RGB images and a question, return index, Answer, Reason, and raw output.
        """
        result = self.compare(img1, img2, question)
        # print(f"LLaVA raw output:\n{result}\n")
        answer, reason = self.parse_answer_reason(result)
        if answer == 'A':
            idx = 0
        elif answer == 'B':
            idx = 1
        else:
            idx = -1
        return idx, answer, reason, result

if __name__ == "__main__":
    img_path1 = "/home/wiss/zhang/projects/openeqa/aeqa/baseline/result_184/exp_eval_aeqa/f776a834-1e21-4442-8834-18b6f9d6cfad/frontier/0_3.png"
    img_path2 = "/home/wiss/zhang/projects/openeqa/aeqa/baseline/result_184/exp_eval_aeqa/f776a834-1e21-4442-8834-18b6f9d6cfad/frontier/0_2.png"

    question = "Where is the orange painting?"

    img1 = np.array(Image.open(img_path1).convert("RGB"))
    img2 = np.array(Image.open(img_path2).convert("RGB"))

    selector = LLaVAFrontierSelector(device="cuda")
    idx, answer, reason, result_text = selector.select(img1, img2, question)

    print("-" * 30)
    if idx == 0:
        print("Final choice: Frontier 0 (A)")
    elif idx == 1:
        print("Final choice: Frontier 1 (B)")
    else:
        print("Model output unclear, please check manually.")
    print(f"Model Answer: {answer}")
    print(f"Model Reason: {reason}")
    # print(f"Raw Output: {result_text}")
