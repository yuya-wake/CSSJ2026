import logging
import re
from typing import Dict, List, Set

import numpy as np
# import tiktoken
from scipy import spatial

### add ###
import spacy
from transformers import AutoTokenizer
from typing import List
import re
######

from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


# CHANGED
nlp = spacy.load("ja_ginza")

# CHANGED
def split_text(
    tokenizer, text: str, max_tokens: int, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    nlp(text) fails if text > ~49KB due to Sudachi limits.
    We pre-split text into safe chunks before passing to spaCy.

    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.

    Returns:
        List[str]: A list of text chunks.
    """
    raw_segments = re.split(r'([。！？\n])', text)

    safe_chunks = []
    current_chunk = []
    current_len = 0
    # Safe limit for Sudachi (approx < 49KB). 10k chars is usually safe for JP text.
    MAX_CHAR_LIMIT = 10000

    # Reconstruct segments and group into safe_chunks
    temp_buffer = ""

    for segment in raw_segments:
        temp_buffer += segment
        # If we hit a delimiter or the segment is getting long
        if segment in ['。', '！', '？', '\n'] or len(temp_buffer) > 1000:

            # Check if adding this segment exceeds limit
            if current_len + len(temp_buffer) > MAX_CHAR_LIMIT:
                if current_chunk:
                    safe_chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0

            # If a single buffer itself is huge (no punctuation), force split
            if len(temp_buffer) > MAX_CHAR_LIMIT:
                # Force split by character count
                for i in range(0, len(temp_buffer), MAX_CHAR_LIMIT):
                    safe_chunks.append(temp_buffer[i : i + MAX_CHAR_LIMIT])
                temp_buffer = ""
            else:
                current_chunk.append(temp_buffer)
                current_len += len(temp_buffer)
                temp_buffer = ""

    # Append remaining
    if temp_buffer:
        if current_len + len(temp_buffer) > MAX_CHAR_LIMIT:
            safe_chunks.append("".join(current_chunk))
            safe_chunks.append(temp_buffer)
        else:
            current_chunk.append(temp_buffer)
            safe_chunks.append("".join(current_chunk))
    elif current_chunk:
        safe_chunks.append("".join(current_chunk))

    # 2. Process safe chunks with spaCy
    sentences = []
    # nlp.pipe is more efficient and safe for batched inputs
    for doc in nlp.pipe(safe_chunks):
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

    # Token counting with calm3 tokenizer
    def count_tokens(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)

    sentence_token_counts = [count_tokens(sentence) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, sentence_token_counts):
        # Case 1: single sentence exceeds max_tokens
        if token_count > max_tokens:
            # Fallback: split by Japanese punctuation
            sub_sentences = re.split(r"[、，；；：:\n]", sentence)
            sub_sentences = [s.strip() for s in sub_sentences if s.strip()]

            sub_chunk = []
            sub_length = 0

            for sub_sentence in sub_sentences:
                sub_token_count = count_tokens(sub_sentence)

                if sub_length + sub_token_count > max_tokens:
                    if sub_chunk:
                        chunks.append("".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(count_tokens(s) for s in sub_chunk)

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append("".join(sub_chunk))

        # Case 2: adding sentence exceeds max_tokens
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append("".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(count_tokens(s) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += token_count

        # Case 3: safe to append
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)
