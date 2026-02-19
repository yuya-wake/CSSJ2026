import logging
from abc import ABC, abstractmethod
import numpy as np
import gc  # CHANGED
from typing import List, Union, Optional  # CHANGED

from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel  # CHANGED
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:  # CHANGED
        """Create an embedding for a single text."""
        raise NotImplementedError

    # Optional cleanup hook (CHANGED)
    def close(self) -> None:
        """Release resources (e.g., GPU memory) if needed."""
        return  # CHANGED


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI embeddings wrapper.
    NOTE: This class exists for compatibility, but you will enforce "no default OpenAI"
    at the config level.  # CHANGED
    """
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text: str) -> List[float]:  # CHANGED
        text = text.replace("\n", " ")
        emb = (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
        # Ensure python list[float] (CHANGED)
        return list(map(float, emb))  # CHANGED


class SBertEmbeddingModel(BaseEmbeddingModel):
    """
    Sentence-Transformers embedding model wrapper.
    Works with multilingual models including "BAAI/bge-m3".  # CHANGED
    """
    def __init__(
            self,
            model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1",
            device: str = "cuda",  # CHANGED: allow explicit device
            normalize: bool = True,  # CHANGED: cosine-friendly embeddings
            ):

        self.model = SentenceTransformer(model_name)
        self.device = device  # CHANGED
        self.normalize = normalize  # CHANGED
        self.model = SentenceTransformer(model_name, device=device)  # CHANGED

    def create_embedding(self, text: str) -> List[float]:
        vec = self.model.encode(
            text,
            convert_to_numpy=True,  # CHANGED
            normalize_embeddings=self.normalize,  # CHANGED
            show_progress_bar=False,
        )
        return vec.astype(np.float32).tolist()  # CHANGED: return list[float]

    def close(self) -> None:
        # Explicitly release GPU memory (CHANGED)
        try:
            del self.model  # CHANGED
        except Exception:
            pass

    if torch is not None and torch.cuda.is_available():  # CHANGED
            torch.cuda.empty_cache()  # CHANGED
            torch.cuda.synchronize()  # CHANGED

### ADD ###
class JapaneseSBERTMeanTokensEmbeddingModel(BaseEmbeddingModel):
    """
    Japanese Sentence-BERT style embedding model for clustering (e.g., sonoisa/sentence-bert-base-ja-mean-tokens-v2).
    Uses mean pooling over token embeddings with attention mask.
    """
    def __init__(
            self,
            model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
            device: Optional[str] = None,
            normalize: bool = True,
            batch_size: int = 8,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        # 属性が消えても復活できるようにする
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if not hasattr(self, "model") or self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        texts = [str(t) for t in texts]

        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            out = self.model(**encoded)
            vecs = self._mean_pooling(out, encoded["attention_mask"])

            if self.normalize:
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

            vecs = vecs.detach().cpu()
            all_vecs.extend(vecs.tolist())

        return all_vecs

    def create_embedding(self, text: str) -> List[float]:
        return self.create_embeddings([text])[0]

    def close(self) -> None:
        # Explicit VRAM release
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()