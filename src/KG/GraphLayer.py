import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Any, Literal,Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


from src.schema import Entity, Relation, MMChunk
import logging
import torch



logger = logging.getLogger("lightrag")
class GraphLayer:
    def __init__(self, level=None):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.level: int | Literal["vert"] = level
        self.entity_index: Dict[str, Entity] = {}
        self.relation_index: Dict[str, Relation] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}  # {entity_id: embedding}
        self.relation_embeddings: Dict[str, np.ndarray] = {}  # {relation_id: embedding}
        self.valid_entries: List[int] = []
        #self.entity_names = set()

    def add_entity(self, entity: Entity) -> str:
        """Add an entity """

        if entity.id in self.entity_index:
            # logger.info(
            #     f"Entity {Entity} with id {entity.id} already exists in layer {self.level}"
            # )
            return False

        # Add to the finest layer
        self.graph.add_node(entity.id,
                            name=entity.name,
                            type=entity.type,
                            attributes=entity.attributes,
                            level=entity.level)

        # Update indexes
        self.entity_index[entity.id] = entity
        #self.entity_names.add(entity.name)

        return True

    def add_relation(self, relation: Relation) -> bool:
        """Add a relation"""

        # Verify both nodes exist
        if (relation.source_id
                not in self.entity_index) or (relation.target_id
                                              not in self.entity_index):
            logger.info(
                f"relation {relation} cannot be added because source or target entity does not exist in the crossponding layer"
            )
            #raise ValueError("Both source and target entities must exist")]
            return False

        # Add to the finest layer
        if relation.attributes["keywords"]:
            keywords = relation.attributes["keywords"][0]
        else:
            keywords = relation.type


        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            type=relation.type,
            key = relation.id, # track the relation id
            keywords=keywords,
            attributes=relation.attributes,
        )
        # Update indexes
        self.relation_index[relation.id] = relation
        # logger.info(
        #     f"Layer {self.level} added relation {source.name} -> {target.name}"
        # )
        return True

    def remove_relation(self, relation: Relation) -> None:
        src = relation.source_id
        tgt = relation.target_id
        rid = relation.id

        removed = False
         # 1. remove from graph
        if self.graph.has_edge(src, tgt, key=rid):
            self.graph.remove_edge(src, tgt, key=rid)
            removed = True

        # 2. remove from relation_index
        if rid in self.relation_index:
            del self.relation_index[rid]
            removed = True
        else:
            if removed:
                logger.warning(
                    f"Relation {rid} existed in graph but not in relation_index."
                )

        # 3. remove from embeddings
        if hasattr(self, "relation_embeddings") and rid in self.relation_embeddings:
            del self.relation_embeddings[rid]
            removed = True

        if not removed:
            logger.warning(f"Relation {rid} not found in layer {self.level}.")
            
        return removed

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by id"""
        return self.entity_index.get(entity_id)

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by id"""
        return self.relation_index.get(relation_id)

    def find_similar_entities(self, query_embedding: np.ndarray, top_k: int = 5,device = "cuda:0") -> List[Tuple[str, float]]:
        """
        Find most relevant entities using vector similarity search
        Returns: List of tuples [(entity_id, similarity_score), ...]
        """
        device = torch.device(device)
        query_tensor = torch.from_numpy(query_embedding).float().to(device)
        
        # Precompute all entity embeddings as a tensor
        entity_ids = []
        entity_embeddings = []
        for entity_id, embedding in self.entity_embeddings.items():
            entity_ids.append(entity_id)
            entity_embeddings.append(embedding)
        
        entity_tensor = torch.from_numpy(np.array(entity_embeddings)).float().to(device)
        
        # Batch cosine similarity calculation
        cos_sim = torch.nn.functional.cosine_similarity(
            query_tensor.unsqueeze(0),  # shape: (1, embedding_dim)
            entity_tensor,              # shape: (n_entities, embedding_dim)
            dim=1
        )
        
        # Get top_k results
        top_scores, top_indices = torch.topk(cos_sim, k=min(top_k, len(entity_ids)))
        
        # Convert back to CPU for output (regardless of input device)
        return [(entity_ids[i], score.item()) 
                for i, score in zip(top_indices.cpu(), top_scores.cpu())]
    
    def find_similar_relations(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most relevant relations using vector similarity search
        Returns: List of tuples [(relation_id, similarity_score), ...]
        """
        similarities = []
        for relation_id, relation_embedding in self.relation_embeddings.items(): 
            # Calculate cosine similarity between query and relation embeddings
            sim = cosine_similarity(query_embedding.reshape(1, -1), 
                                   relation_embedding.reshape(1, -1))[0][0]
            similarities.append((relation_id, sim))
        return similarities[:top_k]

    def traverse_from_entity(self, entity_id: str, max_hops: int = 2) -> Set[str]:
        """
        Perform graph traversal starting from given entity
        Args:
            entity_id: Starting entity ID
            max_hops: Maximum number of hops to traverse (default: 2)
        Returns: Set of reachable entity IDs
        """
        visited = set()
        queue = [(entity_id, 0)]  # (node, current_hop)
        
        while queue:
            current_node, current_hop = queue.pop(0)
            
            # Skip if already visited or exceeded max hops
            if current_node in visited or current_hop > max_hops:
                continue
                
            visited.add(current_node)
            
            # Add all neighbors to the queue
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, current_hop + 1))
                    
        return visited
    
    def visualize(self, with_labels: bool = True,
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None):
        G = self.graph
        if G.number_of_nodes() == 0:
            print("Graph is empty. Nothing to visualize.")
            return

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=figsize)

        # Draw nodes with different colors per type and sizes per level
        node_types = {G.nodes[n].get("type", "Unknown") for n in G.nodes()}
        type_color_map = {nt: f"C{i}" for i, nt in enumerate(node_types)}

        all_levels = sorted({G.nodes[n].get("level", 0) for n in G.nodes()})

        if self.level == "vert" and len(all_levels) > 1:
            # adjust the node size
            base_size = 300
            max_level = max(all_levels)

            size_multiplier = {
                lvl: 1.0 - (0.3 * (lvl / max_level))
                for lvl in all_levels
            }
        else:
            base_size = 300
            size_multiplier = {lvl: 1.0 for lvl in all_levels}

        for nt, color in type_color_map.items():
            nodes_of_type = [
                n for n in G.nodes() if G.nodes[n].get("type") == nt
            ]

            node_sizes = []
            for node in nodes_of_type:
                node_level = G.nodes[node].get("level", 0)
                multiplier = size_multiplier.get(node_level, 0.3)
                node_sizes.append(base_size * multiplier)

            nx.draw_networkx_nodes(G,
                                   pos,
                                   nodelist=nodes_of_type,
                                   node_color=color,
                                   label=nt,
                                   node_size=node_sizes,
                                   alpha=0.8)

        edge_labels = {
            (u, v): d.get("keywords", "")
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edges(G,
                               pos,
                               edge_color='gray',
                               arrows=True,
                               alpha=0.5)

        if with_labels:
            node_labels = {n: G.nodes[n].get("name", n) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

        title = "Knowledge Graph Visualization - Vertical Layer" if self.level == "vert" else f"Knowledge Graph Visualization - Layer {self.level}"
        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.legend(title="Node Types", loc="upper right")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Graph visualization saved to {save_path}")
        else:
            plt.show()