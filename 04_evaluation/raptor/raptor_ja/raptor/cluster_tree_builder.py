import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set
from dataclasses import dataclass  # CHANGED
from typing import Any, Optional  # CHANGED

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params: Optional[Dict[str, Any]] = None,  # CHANGED: avoid mutable default
        model_manager: Optional[Any] = None,  # CHANGED
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params or {}  # CHANGED: safe default
        self.model_manager = model_manager  # CHANGED

    def log_config(self):
        base_summary = super().log_config()
        algo_name = getattr(self.clustering_algorithm, "__name__", str(self.clustering_algorithm))  # CHANGED
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {algo_name}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params
        self.model_manager = getattr(config, "model_manager", None)  # CHANGED

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        if self.model_manager is None:  # CHANGED
            raise ValueError("ClusterTreeBuilder requires config.model_manager for GPU-exclusive loading")  # CHANGED

        next_node_index = len(all_tree_nodes)

        def summarize_cluster_text(cluster, summarization_length) -> str:  # CHANGED
            node_texts = get_text(cluster)
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )
            return summarized_text

        # def process_cluster(
        #     cluster, new_level_nodes, next_node_index, summarization_length, lock
        # ):
        #     node_texts = get_text(cluster)

        #     summarized_text = self.summarize(
        #         context=node_texts,
        #         max_tokens=summarization_length,
        #     )

        #     logging.info(
        #         f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
        #     )

        #     __, new_parent_node = self.create_node(
        #         next_node_index, summarized_text, {node.index for node in cluster}
        #     )

        #     with lock:
        #         new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            # Phase A: Clustering (needs CLUSTER embedding on GPU)  # CHANGED
            self.model_manager.load_cluster_embedding()  # CHANGED

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )
            self.model_manager.unload_cluster_embedding()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            # Phase B: Summarize clusters (needs calm3 on GPU)  # CHANGED
            summ_model = self.model_manager.load_summarizer()
            self.summarization_model = summ_model
            summarized_texts = []  # CHANGED: store texts only
            if use_multithreading:
                pass  # CHANGED
                # with ThreadPoolExecutor() as executor:
                #     for cluster in clusters:
                #         executor.submit(
                #             process_cluster,
                #             cluster,
                #             new_level_nodes,
                #             next_node_index,
                #             summarization_length,
                #             lock,
                #         )
                #         next_node_index += 1
                #     executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    summarized_text = summarize_cluster_text(cluster, summarization_length)  # CHANGED
                    summarized_texts.append((cluster, summarized_text))  # CHANGED

            self.summarization_model = None  # CHANGED
            self.model_manager.unload_summarizer()

            # Phase C: Create parent nodes (needs CLUSTER embedding on GPU)  # CHANGED
            self.model_manager.load_cluster_embedding()  # CHANGED

            new_level_nodes = {}
            for cluster, summarized_text in summarized_texts:  # CHANGED
                __, new_parent_node = self.create_node(
                    next_node_index,
                    summarized_text,
                    {node.index for node in cluster},
                )
                new_level_nodes[next_node_index] = new_parent_node
                next_node_index += 1

            self.model_manager.unload_cluster_embedding()  # CHANGED

            # Update layers
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
