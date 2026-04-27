import re
from PIL import Image
import numpy as np

debug_flag = False


class Verifier:
    def __init__(self, model=None, processor=None, client=None,
                 device=None, limited_cnt=3):
        self.model = model
        self.processor = processor
        self.client = client
        self.device = device
        self.limited_cnt = limited_cnt

    def ask_model(self, image, prompt, use_image=True):
        if use_image and isinstance(image, (Image.Image, np.ndarray)):
            image = self.processor["eval"](image).unsqueeze(0).to(self.device)
            return self.model.generate({"image": image, "prompt": prompt})[0]
        else:
            return self.model.generate({"prompt": prompt})[0]

    def verify_loop(self, original_image, original_q, original_a):
        revised_answer = self.verify(original_image, original_q, original_a)
        first_revised = revised_answer

        conflict_prompt_template = (
            '''Do these two answers conflict with each other?
Answer only "yes" or "no".
 
Answer 1: {answer1}
Answer 2: {answer2}
 
Conflict (yes/no):
'''
        )

        count = 1
        while count < self.limited_cnt:
            temp = self.verify(original_image, original_q, revised_answer)

            full_prompt = conflict_prompt_template.format(
                answer1=revised_answer,
                answer2=temp
            )
            judgement = self.ask_model(
                None, full_prompt, use_image=False
            ).strip().lower()

            if "yes" in judgement:
                revised_answer = temp
            elif "no" in judgement:
                return revised_answer
            else:
                return revised_answer

            count += 1

        return first_revised

    def verify(self, original_image, original_q, original_a):
        # 1. Classify
        classification_prompt = (
            'Role:\n'
            'You are a question classification assistant.\n'
            'Classify the question as either "perception" or "reasoning".\n\n'
            'Rules:\n'
            '1. Output only "perception" or "reasoning" — nothing else.\n'
            '2. Perception = directly reading a visual value from the histogram '
            '(counting bars, reading axis values, identifying the x-axis range).\n'
            '3. Reasoning = interpreting or inferring from the histogram '
            '(distribution shape, number of gaussians, mean, median).\n\n'
            'Examples for histogram questions:\n'
            '1. "How many bars are there in the histogram?" → "perception"\n'
            '2. "What is the minimum value shown on the x-axis?" → "perception"\n'
            '3. "What is the maximum value of the data?" → "perception"\n'
            '4. "How many gaussians were used to generate the data?" → "reasoning"\n'
            '5. "What is the mean value of the data?" → "reasoning"\n'
            '6. "Is this distribution unimodal or bimodal?" → "reasoning"\n\n'
            'Now classify this question — output only "perception" or "reasoning":\n'
            + original_q
        )
        classification = self.ask_model(
            None, classification_prompt, use_image=False
        ).strip().lower()

        # 2. Purify
        if "perception" in classification:
            classification = "perception"
        elif "reasoning" in classification:
            classification = "reasoning"
        else:
            classification = "perception"

        # 3. Route
        if "reasoning" in classification:
            return self.verify_reasoning(original_image, original_q, original_a)
        else:
            return self.verify_perception(original_image, original_q, original_a)

    def verify_perception(self, original_image, original_q, original_a):
        # 1. Generate sub-questions
        sub_q_prompt = (
            'You are analyzing a histogram image.\n'
            'Generate up to 3 short verification questions about what is '
            'directly visible in the histogram.\n\n'
            'Rules:\n'
            '1. Maximum 3 questions.\n'
            '2. Each question must be answerable by reading the histogram directly.\n'
            '3. Focus only on: bar count, x-axis range, y-axis range, bar heights.\n'
            '4. Do NOT ask about statistics, distributions, or interpretations.\n'
            '5. Return a numbered list only — no explanations.\n\n'
            f'Original question: {original_q}\n'
            f'Original answer: {original_a}\n\n'
            'Verification questions:'
        )
        sub_q_text = self.ask_model(None, sub_q_prompt, use_image=False)

        # 2. Parse sub-questions
        sub_questions = re.findall(r'(?:^|\n)\s*\d+[.)]\s*(.+)', sub_q_text)
        sub_questions = [q.strip() for q in sub_questions if len(q.strip()) > 3]
        if not sub_questions:
            sub_questions = [original_q]

        # 3. Answer sub-questions WITH IMAGE — force concise answers
        qa_results = []
        for q in sub_questions:
            q_with_instruction = (
                q + '\n\nIMPORTANT: Reply with ONLY the answer — '
                'a number or short value. No explanation.'
            )
            ans = self.ask_model(
                original_image, q_with_instruction, use_image=True
            )
            qa_results.append({"question": q, "answer": ans})

        # 4. Aggregate
        qa_text = "\n".join(
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in qa_results
        )
        agg_prompt = (
            'You are correcting an answer about a histogram.\n\n'
            'Rules:\n'
            '1. Use the Q&A pairs as ground truth to correct the original answer.\n'
            '2. Output ONLY the corrected answer — no explanation, no extra text.\n'
            '3. If the question asks for a number, output ONLY the number.\n'
            '4. If the question asks for a value, output ONLY the value.\n\n'
            f'Original question: {original_q}\n'
            f'Original answer: {original_a}\n\n'
            f'Q&A verification pairs:\n{qa_text}\n\n'
            'Corrected answer (number or short value only):'
        )
        corrected = self.ask_model(None, agg_prompt, use_image=False)
        return corrected

    def verify_reasoning(self, original_image, original_q, original_a):
        # Step 1: CoT generation with explicit axis clarification
        cot_prompt = (
            'You are analyzing a histogram image. Answer the question step by step.\n\n'
            'CRITICAL — understand the histogram axes:\n'
            '- The X-AXIS shows DATA VALUES (the range of the actual data).\n'
            '- The Y-AXIS shows FREQUENCY (how many data points fall in each bin).\n'
            '- The mean and median refer to X-AXIS values, NOT bar heights.\n'
            '- The maximum and minimum are also referring to X-AXIS VALUES, NOT BAR HEIGHTS, so respond with minimum or maximum on X-AXIS.\n'
            '- Count the number of bars by counting vertical bars from left to right.\n\n'
            f'Question: {original_q}\n\n'
            'Think step by step, then end your response with:\n'
            'Answer: [your final answer — number or short phrase only]'
        )
        cot_output = self.ask_model(original_image, cot_prompt, use_image=True)

        # Step 2: Extract corrected answer
        correction_prompt = (
            'Correct the original answer using the chain-of-thought reasoning below.\n\n'
            'Rules:\n'
            '1. Output ONLY the corrected answer — a number or short phrase.\n'
            '2. No explanation. No sentences. Just the answer value.\n'
            '3. If the reasoning ends with "Answer: X", output only X.\n\n'
            f'Original answer: {original_a}\n\n'
            f'Chain-of-thought reasoning:\n{cot_output}\n\n'
            'Corrected answer:'
        )
        corrected = self.ask_model(None, correction_prompt, use_image=False)
        return corrected

    def vqa_model_evaluation(self, original_image, questions):
        if isinstance(original_image, (Image.Image, np.ndarray)):
            image = self.processor["eval"](
                original_image
            ).unsqueeze(0).to(self.device)
        else:
            image = original_image
        result = []
        for question in questions:
            answer = self.model.generate(
                {"image": image, "prompt": question}
            )[0]
            result.append({"question": question, "answer": answer})
        return result