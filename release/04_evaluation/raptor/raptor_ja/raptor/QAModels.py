import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer

from typing import Optional, List, Tuple, Set  # CHANGED
import torch  # CHANGED
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)  # CHANGED


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]

### ADD ###
with open("raptor/prompt/prompt_strict.txt", "r", encoding="utf-8") as f:
    prompt = f.read()
CLASSIFICATION_SYSTEM_PROMPT = prompt

def build_allowed_token_seqs(tokenizer, labels: Tuple[str, ...] = ("real", "fake")) -> List[List[int]]:
    # Build token sequences for each label
    allowed = []
    for lab in labels:
        ids = tokenizer(lab, add_special_tokens=False).input_ids
        allowed.append(ids)
    return allowed

def _get_eos_id(model, tokenizer) -> int:
    # Choose eos token id robustly
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
        return int(model.config.eos_token_id)
    # Fallback: use pad_token_id if eos is missing
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise ValueError("Cannot determine eos_token_id")

class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len: int, allowed_token_seqs: List[List[int]], eos_token_id: int):
        super().__init__()
        self.prompt_len = prompt_len
        self.allowed = allowed_token_seqs
        self.eos = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Single-sample assumption (your pipeline is sequential)
        gen = input_ids[0, self.prompt_len:].tolist()

        allowed_next = set()
        for seq in self.allowed:
            if len(gen) <= len(seq) and gen == seq[: len(gen)]:
                if len(gen) == len(seq):
                    allowed_next.add(self.eos)  # completed label -> allow EOS only
                else:
                    allowed_next.add(seq[len(gen)])

        if not allowed_next:
            allowed_next = {self.eos}  # safety: avoid infinite loop

        mask = torch.full_like(scores, float("-inf"))
        mask[0, list(allowed_next)] = 0.0
        return scores + mask

class StopOnAnyTarget(StoppingCriteria):
    def __init__(self, prompt_len: int, tokenizer, target_texts_norm: Set[str]):
        super().__init__()
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        self.target_texts_norm = target_texts_norm

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
        return text in self.target_texts_norm

class LocalHFClassificationQAModel(BaseQAModel):
    """
    Local HF chat model classifier for {real,fake} (calm3-22b-chat).
    - Uses apply_chat_template() for chat formatting.
    - Forces output to one of {'real','fake'} using constrained decoding.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.model_path = model_path
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Prebuild constraints
        self.allowed = build_allowed_token_seqs(self.tokenizer, labels=("real", "fake"))  # CHANGED
        self.eos_id = _get_eos_id(self.model, self.tokenizer)

    def build_messages(self, title: str, description: str, context_text: str):
        # Build chat messages
        user_content = (
            f"Title: {title}\n"
            f"Description: {description}\n\n"
            f"Transcript (RAPTOR Root Summary + Retrieved Excerpts):\n{context_text}\n\n"
            f"Based on the definitions provided, classify this content."
        )

        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages

    def answer_question(
        self,
        context,
        question=None,
        *,
        title: str = "",
        description: str = "",
        max_new_tokens: int = 10,
    ):
        """
        Compatibility:
        - `context` is used as context_text (RAPTOR root + retrieved excerpts).
        - `question` is ignored by default, but kept for interface compatibility.

        Additional inputs:
        - title, description can be injected later.
        """
        context_text = str(context)

        messages = self.build_messages(title=title, description=description, context_text=context_text)  # CHANGED

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Move to model device
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(self.model.device)

        prompt_len = input_ids.shape[1]

        lp = LogitsProcessorList([
            PrefixConstrainedLogitsProcessor(prompt_len, self.allowed, self.eos_id)
        ])

        sc = StoppingCriteriaList([
            StopOnAnyTarget(prompt_len, self.tokenizer, {"real", "fake"})
        ])

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_processor=lp,
                stopping_criteria=sc,
                eos_token_id=self.eos_id,
            )

        # Decode only generated part
        gen_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()  # CHANGED

        # Final safety normalization
        if text.startswith("real"):
            return "real"
        if text.startswith("fake"):
            return "fake"

        # If something goes wrong, fall back to a strict error
        raise ValueError(f"Model output is not 'real' or 'fake': {text}")

    def close(self) -> None:
        # Explicitly release GPU memory
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
