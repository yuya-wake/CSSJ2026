import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import gc
import csv
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

import tiktoken
from transformers import AutoTokenizer

# =========================
# User config
# =========================
MODEL_PATH = "/home/wake/local_models/cyberagent-calm3-22b-chat"
retr_emb    = "BAAI/bge-m3"
cluster_emb = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

MNT_PATH = os.getenv("MNT_PATH") or ""
INPUT_CSV_PATH = os.path.join(MNT_PATH, "complete_data", "real_and_fake_w_summary.csv")

# Retrieval queries (as requested)
RETRIEVAL_QUERIES = [
    "言及されている具体的な医療上の主張・作用機序・治療法は何か？",
    "健康効果の誇張、有害な助言、または危険な指導は含まれているか？",
    "確立された医学的事実や医学的コンセンサスに反する記述はあるか？",
]

QUERY_TEMPLATE = "Represent this sentence for searching relevant passages: {q}"

# Retrieval mode: "collapsed" or "traversal"
RETRIEVAL_MODE = "collapsed"

# Output
OUT_DIR = os.path.join(MNT_PATH, "results_run_japanese")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PRED_CSV = os.path.join(OUT_DIR, f"preds_{RETRIEVAL_MODE}_strict_real_fake.csv")
OUT_GENLOG_CSV = os.path.join(OUT_DIR, f"{RETRIEVAL_MODE}_strict_generation_inputs.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# Imports from your codebase
# =========================
# NOTE: keep imports inside functions where possible to avoid importing heavy deps early.

from .tree_structures import Node, Tree
from .utils import split_text


# =========================
# ModelManager: load/unload to satisfy VRAM constraint
# =========================
class ModelManager:
    """
    Manages mutually-exclusive GPU residency:
    - Embedding model (bge-m3 via SBertEmbeddingModel)
    - Generation model (calm3) used for summarization / classification
    You can load one, use it, then unload to free VRAM.
    """

    def __init__(self, model_path: str, cluster_embedding_model_name: str, retr_embedding_model_name: str):
        self.model_path = model_path
        self.cluster_embedding = cluster_embedding_model_name
        self.retr_embedding = retr_embedding_model_name

        self._cluster_emb = None
        self._retr_emb = None
        self._summarizer = None
        self._classifier = None

    # ---------- Embedding ----------
    def load_cluster_embedding(self):
        if self._cluster_emb is not None:
            return self._cluster_emb

        # Lazy import
        from .EmbeddingModels import JapaneseSBERTMeanTokensEmbeddingModel

        # Create embedding model on GPU
        self._cluster_emb = JapaneseSBERTMeanTokensEmbeddingModel(
            model_name=self.cluster_embedding,
            device="cuda",
            normalize=True,
            batch_size=8,
        )
        return self._cluster_emb

    def unload_cluster_embedding(self):
        if self._cluster_emb is None:
            return
        try:
            self._cluster_emb.close()
        except Exception:
            pass
        self._cluster_emb = None
        self._gc_cuda()

    def load_retr_embedding(self):
        if self._retr_emb is not None:
            return self._retr_emb

        # Lazy import
        from .EmbeddingModels import SBertEmbeddingModel

        # Create embedding model on GPU
        self._retr_emb = SBertEmbeddingModel(
            model_name=self.retr_embedding,
            device="cuda",
            normalize=True,
        )
        return self._retr_emb

    def unload_retr_embedding(self):
        if self._retr_emb is None:
            return
        try:
            self._retr_emb.close()
        except Exception:
            pass
        self._retr_emb = None
        self._gc_cuda()

    # ---------- Summarization ----------
    def load_summarizer(self):
        if self._summarizer is not None:
            return self._summarizer

        from .SummarizationModels import LocalHFSummarizationModel

        # Create calm3 summarizer on GPU
        self._summarizer = LocalHFSummarizationModel(
            model_path=self.model_path,
        )
        return self._summarizer

    def unload_summarizer(self):
        if self._summarizer is None:
            return
        try:
            self._summarizer.close()
        except Exception:
            pass
        self._summarizer = None
        self._gc_cuda()

    # ---------- Classifier ----------
    def load_classifier(self):
        if self._classifier is not None:
            return self._classifier

        from .QAModels import LocalHFClassificationQAModel

        self._classifier = LocalHFClassificationQAModel(
            model_path=self.model_path,
            load_in_4bit=True,
        )
        return self._classifier

    def unload_classifier(self):
        if self._classifier is None:
            return
        try:
            self._classifier.close()
        except Exception:
            pass
        self._classifier = None
        self._gc_cuda()

    # ---------- Utility ----------
    def _gc_cuda(self):
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

from .SummarizationModels import BaseSummarizationModel

class LazySummarizationProxy(BaseSummarizationModel):
    """
    TreeBuilderConfig の 'summarization_model 必須' 制約を満たしつつ、
    summarize() 呼び出しのたびに GPU 上で calm3 を load/unload する代理。
    """
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def summarize(self, context: str, max_tokens: int = 500) -> str:
        summ = self.manager.load_summarizer()
        try:
            return summ.summarize(context, max_tokens=max_tokens)
        finally:
            self.manager.unload_summarizer()

### Build RAPTOR TREE ###
def build_raptor_tree(transcript: str, manager: ModelManager, tokenizer) -> Tree:
    """
    Build RAPTOR tree using UMAP+GMM(BIC) clustering, then summarization.
    VRAM policy:
      1) load cluster embedding -> build leaf embeddings + clustering (UMAP/GMM) on CPU -> unload cluster emb
      2) load summarizer -> build summaries -> unload summarizer
    """

    # 1) CLUSTER embedding on GPU
    cluster_emb = manager.load_cluster_embedding()

    # Build tree structure (clustering happens here)
    from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
    from .cluster_utils import RAPTOR_Clustering

    proxy_summarizer = LazySummarizationProxy(manager)

    cfg = ClusterTreeConfig(
        tokenizer=tokenizer,
        max_tokens=100,
        num_layers=5,
        top_k=5,                 # unused for GMM clustering path (safe to leave)
        threshold=0.5,           # unused for GMM clustering path (safe to leave)
        selection_mode="top_k",  # unused for GMM clustering path (safe to leave)
        summarization_length=100,
        summarization_model=proxy_summarizer,
        embedding_models={"CLUSTER": cluster_emb},   # CHANGED: only CLUSTER here
        cluster_embedding_model="CLUSTER",
        model_manager=manager,

        # RAPTOR clustering params (UMAP+GMM+BIC+recursive)
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,
        clustering_params={
            "threshold": 0.1,
            "max_length_in_cluster": 3500,
            "tokenizer": tokenizer,
            # "verbose": False,
        },
    )

    builder = ClusterTreeBuilder(cfg)
    tree = builder.build_from_text(transcript, use_multithreading=False)

    # Free CLUSTER embedding after build
    manager.unload_cluster_embedding()

    return tree


### RETRIEVAL ###
def add_retrieval_embeddings(tree: Tree, retr_emb, key: str = "RETR") -> None:
    """
    Add retrieval embeddings to all nodes after the tree is built.
    This allows you to keep CLUSTER embeddings during build, then free GPU, then load retrieval model.
    """
    for node_id, node in tree.all_nodes.items():
        # Ensure dict exists (original code always sets dict, but be safe)
        if node.embeddings is None:
            node.embeddings = {}

        # Compute and store retrieval embedding
        node.embeddings["RETR"] = retr_emb.create_embedding(node.text)


def attach_retrieval_embeddings(tree: Tree, manager: ModelManager) -> None:
    """
    Add retrieval embeddings (e.g., bge-m3) to all nodes after tree is built.
    VRAM policy: load retr model -> embed all nodes -> unload retr model.
    """
    retr_emb = manager.load_retr_embedding()
    add_retrieval_embeddings(tree, retr_emb, key="RETR")
    manager.unload_retr_embedding()

def retrieve_context_text_existing(
    tree: Tree,
    manager: ModelManager,
    tokenizer,
    queries: List[str],
    retrieval_mode: str,
    query_template: str,
    traversal_top_k: int = 9,
    collapsed_max_tokens: int = 2000,
) -> str:
    """
    Use the existing TreeRetriever implementation.
    Assumes node.embeddings has retrieval vectors under key "RETR".
    VRAM policy:
      - load retr embedding model (GPU) to embed the query
      - run TreeRetriever.retrieve (CPU-heavy; uses stored node embeddings)
      - unload retr embedding
    """
    # 1) Ensure retrieval embeddings exist on nodes
    attach_retrieval_embeddings(tree, manager)

    # 2) Load retrieval embedding model (needed to embed queries inside TreeRetriever)
    retr_emb_model = manager.load_retr_embedding()

    # 3) Build TreeRetriever with existing config/classes
    #    (IMPORTANT: use RELATIVE imports to avoid dual-loading bugs)
    from .tree_retriever import TreeRetriever, TreeRetrieverConfig

    tr_cfg = TreeRetrieverConfig(
        tokenizer=tokenizer,                 # HF tokenizer ok
        threshold=0.5,                       # keep default unless you tuned it
        top_k=traversal_top_k,               # used in traversal; harmless in collapsed
        selection_mode="top_k",              # TreeRetriever supports top_k/threshold
        context_embedding_model="RETR",      # <-- Node.embeddings["RETR"] を使う
        embedding_model=retr_emb_model,      # query embedding を作るために必要
        num_layers=None,
        start_layer=None,
        query_template=query_template,       # "Represent this sentence..." の prefix
    )
    retriever = TreeRetriever(tr_cfg, tree)

    # 4) Mode switch: collapsed vs traversal
    collapse_tree = (retrieval_mode == "collapsed")

    contexts: List[str] = []
    for q in queries:
        q_text = query_template.format(q=q) if query_template else q

        # 既存 retrieve のシグネチャに合わせる
        context, _layer_info = retriever.retrieve(
            query=q_text,
            start_layer=None,
            num_layers=None,
            top_k=traversal_top_k,                 # traversalで使う
            max_tokens=collapsed_max_tokens,       # collapsedで使う（token budget）
            collapse_tree=collapse_tree,
            return_layer_information=True,
        )
        contexts.append(context)

    # 5) Unload retr embedding model
    manager.unload_retr_embedding()

    return "\n\n".join([c for c in contexts if c])


### CLASSIFY ###
def classify_real_fake(title: str, description: str, context_text: str, manager: ModelManager) -> str:
    """
    Run calm3 classifier to output exactly 'real' or 'fake'.

    VRAM policy:
    - Load classifier (GPU)
    - Inference
    - Unload classifier
    """
    clf = manager.load_classifier()
    pred = clf.answer_question(
        context_text,
        None,
        title=title,
        description=description,
        max_new_tokens=10,
    )
    manager.unload_classifier()
    return pred


### CSV logging (generation inputs)
def ensure_csv_header(path: str, header: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_genlog(path: str, row: Dict[str, str]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(k, "") for k in ["video_id", "title", "description", "transcript","context_text", "pred", "label"]])


### Main
def main():
    df = pd.read_csv(INPUT_CSV_PATH, dtype=str, encoding="utf-8")
    df = df[df["is_long"] == "True"].copy()

    manager = ModelManager(MODEL_PATH, cluster_emb, retr_emb)

    ensure_csv_header(
        OUT_GENLOG_CSV,
        header=["video_id", "title", "description", "transcript", "context_text", "pred", "label"],
    )
    ensure_csv_header(
        OUT_PRED_CSV,
        header=["video_id", "pred", "label"],
    )

    preds = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = str(row.get("video_id") or "")
        title = str(row.get("title") or "")
        description = str(row.get("description") or "")
        transcript = str(row.get("transcript") or "")
        label = str(row.get("label") or "")

        # 1) Build tree (embedding -> unload -> summarization -> unload)
        tree = build_raptor_tree(transcript, manager, tokenizer)

        # 2) Retrieve context (embedding -> unload)
        context_text = retrieve_context_text_existing(
            tree=tree,
            manager=manager,
            tokenizer=tokenizer,
            queries=RETRIEVAL_QUERIES,
            retrieval_mode=RETRIEVAL_MODE,      # "collapsed" or "traversal"
            query_template=QUERY_TEMPLATE,
            traversal_top_k=9,                  # 論文通り
            collapsed_max_tokens=2000,          # 論文通り
        )

        # 3) Classify (calm3 -> unload)  # CHANGED
        pred = classify_real_fake(title, description, context_text, manager)

        preds.append({"video_id": video_id, "pred": pred, "label": label})

        append_genlog(
            OUT_GENLOG_CSV,
            {
                "video_id": video_id,
                "title": title,
                "description": description,
                "transcript": transcript,
                "context_text": context_text,
                "pred": pred,
                "label": label,
            },
        )

    # Save predictions
    out_df = pd.DataFrame(preds)
    out_df.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8")
    logging.info(f"Saved: {OUT_PRED_CSV}")
    logging.info(f"Saved: {OUT_GENLOG_CSV}")


if __name__ == "__main__":
    main()