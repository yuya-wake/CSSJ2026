import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

import torch  # CHANGED
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # CHANGED

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    """
    OpenAI chat-completions based summarizer.
    NOTE: You will enforce "no default OpenAI" at the config level.
    This class remains for compatibility only.  # CHANGED
    """
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    """
    (Legacy) OpenAI completions based summarizer.
    NOTE: You will enforce "no default OpenAI" at the config level.
    This class remains for compatibility only.  # CHANGED
    """
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e

### ADD ###
class LocalHFSummarizationModel(BaseSummarizationModel):
    """
    Local HuggingFace causal LM summarizer (e.g., calm3-22b-chat).
    Designed to run on GPU, with explicit close() to free VRAM.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens_default: int = 500,
    ):
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for LocalHFSummarizationModel"
            )

        self.model_path = model_path
        self.device = device
        self.max_new_tokens_default = max_new_tokens_default

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # CHANGED

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def summarize(self, context: str, max_tokens: int = 500) -> str:
        """
        max_tokens here is treated as "budget hint".
        We map it to max_new_tokens to keep original signature stable.  # CHANGED
        """
        context = str(context).strip()

        # ADD: Japanese-friendly summarization instruction
        system_prompt = (
            "あなたは要約アシスタントです。"
            "入力テキストの主要な主張・手順・用量・注意点・結論を保持しつつ、"
            "冗長さを削って簡潔に要約してください。"
            "新しい主張を追加したり、判断・評価・反論を加えたりしないでください。"
        )
        user_prompt = f"テキスト:\n{context}\n\n要約:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_new_tokens = min(max_tokens, self.max_new_tokens_default)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if torch is not None and hasattr(input_ids, "to"):
            input_ids = input_ids.to(self.model.device)

        with torch.no_grad() if torch is not None else _nullcontext():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gen_ids = out[0][input_ids.shape[-1]:]

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return text

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# ADD: small helper for no-torch environments
class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False