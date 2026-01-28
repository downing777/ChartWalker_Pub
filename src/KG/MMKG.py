import networkx as nx

from typing import List, Dict, Any, Literal,Callable
from dataclasses import dataclass, field


import numpy as np
from src.schema import Entity, Relation
from src.schema import MMChunk
import yaml
import uuid
import logging
import pickle
from .entity_resolver import EntityResolver
import asyncio
import copy
from .GraphLayer import GraphLayer
from collections import defaultdict



logger = logging.getLogger("lightrag")


prompts_path = "/share/project/tangning/MMGraph/src/KG/kg_prompts.yaml"

@dataclass
class EntityResolutionResult:
    """Entity resolution result class definition."""
    graph: nx.Graph

def semantic_relation_signature(relation: Relation):
        u, v = sorted([relation.source_id, relation.target_id])
        desc = relation.attributes.get("description", "").strip()
        chunk = relation.attributes.get("chunk_id")

        return (u,
                v,
                desc,
                chunk)


class HierarchicalMMKG:
    def __init__(self, args=None):
        self.layers: List[GraphLayer] = []
        self.vertical_layer = GraphLayer(level="vert")
        self._initialize_layer()

        self.id2entity: Dict[str, Entity] = {}
        self.id2relation: Dict[str, Relation] = {}
        self.entity2id: Dict[str, str] = {}
        #self.name2entity: List[Dict[str, Entity]] = []
        #entities share the same name
        self.name2entity: Dict[str, list[Entity]] = {}
        with open(prompts_path, "r") as f:
            self.PROMPTS = yaml.safe_load(f)
        self._lock = asyncio.Lock()
        self.valid_entries = []
        self.relation_signature_set = set()
        if args:
            self.vlm = args.vlm
        else:
            self.vlm = "qwen3-vl-8b"

    def _initialize_layer(self):
        """Initialize the Level1 Graph"""
        self._add_layer()

    def _add_layer(self):
        """Add a new layer to the graph"""
        self.layers.append(GraphLayer(level=len(self.layers)))

    def get_layer(self, level) -> GraphLayer:
        """
        Get a layer by its level
        #TODO 这里为什么会出现entity的layer 溢出的情况？
        """

        if level == "vert":
            return self.vertical_layer
        else:
            if 0 <= level < len(self.layers):
                return self.layers[level]
            else:
                logger.info(f"Invalid level: {level}")
                return self.layers[-1]

    async def _add_entity(self, entity: Entity) -> Entity:
        '''
        Return the entity if added else the same name entity
        '''
        if entity.name not in self.name2entity:
            level = entity.level

            while level >= len(self.layers):
                self._add_layer()

            layer = self.get_layer(entity.level)
            added = layer.add_entity(entity)
            if added:
                async with self._lock:  # lock the writing behavior
                    self.name2entity[entity.name] = [entity]
                    self.id2entity[entity.id] = entity
                    return entity
        else:
            ent_list = self.name2entity[entity.name]
            same_ent_idx = await EntityResolver(vlm = self.vlm)._resolve_sharename_entity(
                entity, ent_list)
            async with self._lock:
                if same_ent_idx is None:
                    self.name2entity[entity.name].append(entity)
                    self.id2entity[entity.id] = entity
                    return entity
                else:
                    same_ent = ent_list[same_ent_idx]
                    same_ent.source |= entity.source
                    # logging.warning(
                    #     f"Entity {entity.name} already exists, merge with {same_ent}"
                    # )
                    return same_ent

    async def _del_entity(self, entity_id: str):
        """
        Delete an entity from MMKG, including:
        - all its relations (both in id2relation and in corresponding layers)
        - its entry in id2entity / name2entity
        - its node in the corresponding GraphLayer
        """

        if entity_id not in self.id2entity:
            logger.warning(f"Entity {entity_id} does not exist, cannot delete")
            return False

        async with self._lock:
            entity = self.id2entity[entity_id]
            logger.info(f"Deleting entity {entity}")
            related_rels = [
                rel for rel in self.id2relation.values()
                if rel.source_id == entity_id or rel.target_id == entity_id
            ]

            # delete the relation index
            logger.info(f"Deleting {len(related_rels)} relations related to entity {entity_id}")
            for rel in related_rels:
                # remove the relation signature
                await self._remove_relation(rel)

            # delete the entity index
            del self.id2entity[entity_id]
            if entity.name in self.name2entity:
                logger.info(f"name2entity {self.name2entity[entity.name]}")
                self.name2entity[entity.name] = [
                    e for e in self.name2entity[entity.name]
                    if e.id != entity_id
                ]
                if not self.name2entity[entity.name]:
                    del self.name2entity[entity.name]

            layer = self.get_layer(entity.level)
            vert_layer = self.get_layer("vert")
            if entity_id in layer.entity_index:
                del layer.entity_index[entity_id]
                layer.graph.remove_node(entity_id)
            elif entity_id in vert_layer.entity_index:
                del vert_layer.entity_index[entity_id]
                vert_layer.graph.remove_node(entity_id)
            else:
                logger.info(f"Entity {entity_id} not found in any layer")

            logger.info(
                f"Deleted entity {entity_id} ({entity.name}) and all its relations"
            )

    def get_entity(self, id: str):
        return self.id2entity[id]

    def get_relation(self, id: str):
        return self.id2relation[id]

    def get_source_entity(self, file_source: List[str]):
        '''
        Given a query's ground truth, return the relevant entity found 
        For debug use
        '''
        if isinstance(file_source, str):
            file_source = [file_source]
        ent_list = []
        for file in file_source:
            for entity in self.id2entity.values():
                if file in entity.source:
                    ent_list.append(entity)
        return ent_list

    async def _add_relation(self, relation: Relation):
        async with self._lock:
            sig = semantic_relation_signature(relation)
            if sig in self.relation_signature_set:
                logger.info(f"Duplicate relation ignored: {sig}")
                return False

            if relation.source_id == relation.target_id:
                logger.warning(f"Relation add failed, no self-looping allowed")
                return False
            
            source_id = relation.source_id
            target_id = relation.target_id
            source = self.id2entity[source_id]
            target = self.id2entity[target_id]

            if source.level == target.level:
                layer = self.get_layer(source.level)
                added = layer.add_relation(relation)
            else:
                added = self._add_hierarchical_relations(relation)
            if added:
                self.id2relation[relation.id] = relation
                self.relation_signature_set.add(sig)

    def _add_hierarchical_relations(self, relation: Relation) -> bool:
        """
        Add hierarchical relations between coarse and fine nodes
        Hierarchical relations can also be stored in an nx.graph, which you should view it as vertical
        """
        source = self.id2entity[relation.source_id]
        target = self.id2entity[relation.target_id]
        self.vertical_layer.add_entity(source)
        self.vertical_layer.add_entity(target)

        return self.vertical_layer.add_relation(relation)


    async def _remove_relation(self, relation: Relation) -> None:
        # 1. 计算 signature（用于全局去重集）
        sig = semantic_relation_signature(relation)

        # 2. 定位 source / target
        source = self.get_entity(relation.source_id)
        target = self.get_entity(relation.target_id)

        if source is None or target is None:
            logger.warning(
                f"Remove relation failed: entity not found "
                f"{relation.source_id} -> {relation.target_id}"
            )
            return

        # 3. 判断所在 layer
        if source.level == target.level:
            layer = self.get_layer(source.level)
        else:
            layer = self.get_layer("vert")

        # 4. 从 layer 中删除
        removed = layer.remove_relation(relation)

        # 5. 清理全局索引
        self.id2relation.pop(relation.id, None)
        self.relation_signature_set.discard(sig)

        logger.info(f"Relation removed: {sig}")

    #TODO maybe deprecate later, this is too specific to maintain in the MMKG class
    async def init_graph(self, results):
        '''
        Initialize the graph based on the llm extraction
        results: parsed llm extraction results
            [{"chunk_id":..., "entities":..., "relationships":...}]
        '''
        for res in results:
            try:
                chunk_id = res.get("chunk_id")
                for ent in res.get("entities", []):
                    name = ent.get("entity_name", "").strip()
                    if not name:
                        logger.warning(
                            f"Skipping entity due to empty name in chunk {chunk_id}"
                        )
                        continue
                    level = int(ent.get("level", 1)) - 1

                    entity_id = uuid.uuid4().hex
                    entity = Entity(id=entity_id,
                                    name=name,
                                    type=ent.get("entity_type", ""),
                                    attributes={
                                        "description":
                                        ent.get("entity_description", "")
                                    },
                                    level=level,
                                    source=set([chunk_id]))
                    await self._add_entity(entity)
                logger.info("Adding relations")
                for rel in res.get("relationships", []):
                    source_name = rel.get("source_entity", "")
                    target_name = rel.get("target_entity", "")
                    # TODO Check duplicate
                    source = self.search_entity(source_name, chunk_id)
                    target = self.search_entity(target_name, chunk_id)

                    relation_type = rel.get("relation_type", "related")
                    relation_keywords = rel.get("relationship_keywords", [])
                    description = rel.get("relationship_description", "")

                    if not source or not target:

                        logger.info(
                            f"Entity list {self.name2entity.keys()}, Relationship {rel} not added due to missing source or target entity"
                        )
                        continue

                    relation = Relation(
                        source_id=source.id,
                        target_id=target.id,
                        id=uuid.uuid4().hex,
                        type=relation_type,
                        attributes={
                            "description": description,
                            "keywords": relation_keywords,
                            "chunk_id": chunk_id
                        },
                    )
                    await self._add_relation(relation)
            except Exception as e:
                logger.error(f"Error processing results: {e}")
                continue
        return

    def search_entity(self, entity_name: str, source: str) -> Entity:
        """_summary_

        seach the entity by name
        only one entity will be returned
        """
        ent_list = self.name2entity.get(entity_name, [])
        if len(ent_list) == 1:
            return ent_list[0]
        elif len(ent_list) > 1:
            for ent in ent_list:
                if source in ent.source:
                    return ent
        else:
            logger.warning(f"Entity {entity_name} not found")
            return None

    async def merge_entities(self, base_ent_id: str, other_ent_id: str):
        """
        Merge other_ent into base_ent globally:
            - Copy all relations of other_ent to base_ent
            - Add them via _add_relation
            - Remove other_ent and its original relations
        """
        if base_ent_id not in self.id2entity or other_ent_id not in self.id2entity:
            logger.warning(
                f"Cannot merge {base_ent_id} <- {other_ent_id}, one of them does not exist."
            )
            return

        base_ent = self.get_entity(base_ent_id)
        other_ent = self.get_entity(other_ent_id)

        related_rels = [
            rel for rel in self.id2relation.values()
            if rel.source_id == other_ent_id or rel.target_id == other_ent_id
        ]

        # deepcopy the relations to avoid modifying the original ones
        for rel in related_rels:
            new_rel = copy.deepcopy(rel)
            new_rel.id = uuid.uuid4().hex
            if new_rel.source_id == other_ent_id:
                new_rel.source_id = base_ent_id
            if new_rel.target_id == other_ent_id:
                new_rel.target_id = base_ent_id
            await self._add_relation(new_rel)

        await self._del_entity(other_ent_id)

        base_ent.source |= other_ent.source

        # logger.info(
        #     f"Globally merged entity {other_ent_id} ({other_ent.name}) into {base_ent_id} ({base_ent.name})"
        # )

    async def merge_graph(self, newkg: "HierarchicalMMKG"):
        '''
        Merge the new hierarchical knowledge graph to the main graph in place
        '''
        logger.info(
            "Merging new hierarchical knowledge graph to the main graph")
        candidate_entities = newkg.id2entity
        candidate_relations = newkg.id2relation
        logger.info(f"Number of candidate entities: {len(candidate_entities)}")
        logger.info(
            f"Number of candidate relations: {len(candidate_relations)}")

        for entity in candidate_entities.values():
            ori_entity = await self._add_entity(entity)
            if ori_entity != entity:
                ori_id = ori_entity.id
                new_id = entity.id
                for rel in candidate_relations.values():
                    if rel.source_id == new_id:
                        rel.source_id = ori_id
                    if rel.target_id == new_id:
                        rel.target_id = ori_id
        for rel in candidate_relations.values():
            await self._add_relation(rel)

    def emb(self, emb_model):
        entity_texts = []
        entity_ids = []
        for entity_id, entity in self.id2entity.items():
            # Create text representation of entity (customize as needed)
            entity_text = f"{entity.name} {entity.type} {entity.attributes.get('description', '')}"
            entity_texts.append(entity_text)
            entity_ids.append(entity_id)

        # Encode in batches to improve efficiency
        batch_size = 32  # Adjust based on your model's capacity
        for i in range(0, len(entity_texts), batch_size):
            batch_texts = entity_texts[i:i + batch_size]
            batch_embeddings = emb_model.encode(batch_texts)

            # Store embeddings in each layer where the entities exist
            for j, entity_id in enumerate(entity_ids[i:i + batch_size]):
                entity = self.id2entity[entity_id]
                layer = self.get_layer(entity.level)
                if entity_id in layer.entity_index:
                    layer.entity_embeddings[entity_id] = batch_embeddings[j]

        # Batch encode relations
        relation_texts = []
        relation_ids = []
        for relation_id, relation in self.id2relation.items():
            # Get source and target entities
            source = self.id2entity[relation.source_id]
            target = self.id2entity[relation.target_id]

            # Create text representation of relation (customize as needed)
            relation_text = (
                f"{source.name} {relation.type} {target.name} "
                f"{relation.attributes.get('description', '')} "
                f"{' '.join(relation.attributes.get('keywords', []))}")
            relation_texts.append(relation_text)
            relation_ids.append(relation_id)

        # Encode relations in batches
        for i in range(0, len(relation_texts), batch_size):
            batch_texts = relation_texts[i:i + batch_size]
            batch_embeddings = emb_model.encode(batch_texts)

            # Store relation embeddings in appropriate layers
            for j, relation_id in enumerate(relation_ids[i:i + batch_size]):
                relation = self.id2relation[relation_id]
                source = self.id2entity[relation.source_id]
                target = self.id2entity[relation.target_id]

                # Store in horizontal layer if same level
                if source.level == target.level:
                    layer = self.get_layer(source.level)
                    layer.relation_embeddings[relation_id] = batch_embeddings[
                        j]
                # Also store in vertical layer
                self.vertical_layer.relation_embeddings[
                    relation_id] = batch_embeddings[j]

        print(
            f"Successfully encoded {len(self.id2entity)} entities and {len(self.id2relation)} relations"
        )

    #TODO
    def graph_search(
        self,
        query: dict,
        #query_embedding: np.ndarray,
        top_k_similar: int = 10,
        max_hops: int = 3,
        max_sources: int = 20,
    ) -> Dict[str, Any]:
        """
        Enhanced hybrid graph search with two-phase strategy:
        1. Entity-first retrieval: Find core entities without consuming source quota
        2. Relation-aware expansion: Explore meaningful paths before collecting sources
        
        Args:
            query: The input query string
            query_embedding: Embedding vector of the query
            top_k_similar: Number of top similar entities to retrieve initially
            max_hops: Maximum relation hops to traverse
            max_sources: Maximum unique sources to collect
            rerank: Given the retrived paths, rerank them by similarity based on the meta data.
        
        Returns:
            Dictionary containing:
            - query: Original query
            - similar_entities: Top-k similar entities (Entity, score)
            - related_entities: All discovered entities
            - related_relations: All discovered relations
            - associated_chunks: Collected source chunks (up to max_sources)
            - paths: Sample exploration paths for explainability
        """
        # Initialize layer and retrieve initial entities
        # layer0 = self.get_layer(0)
        # similar_entities = layer0.find_similar_entities(query_embedding, top_k=top_k_similar)
        query_embedding = query.get("embedding")
        all_candis = []
        for layer in self.layers:  # Assuming self.layers contains all available layers
            all_candis.extend(
                layer.find_similar_entities(query_embedding,
                                            top_k=top_k_similar))

        top_items = sorted(all_candis, key=lambda x: x[1], reverse=True)[:top_k_similar]

        similar_entities = [(self.get_entity(ent_id), score) for ent_id, score in top_items]

        # Phase 1: Entity candidate collection (no source consumption)
        #candidate_entities = {ent_id: score for ent_id, score in similar_entities}
        associated_chunks = set()
        for item in similar_entities:
            ent = item[0]
            associated_chunks.update(ent.source)
        #gt_sources = query.get("gold")
        #gt_entities = self.get_source_entity(gt_sources)
        return {
            "similar_entities": similar_entities,
            "associated_chunks": list(associated_chunks),
            #"gt_entities": gt_entities
        }

    def _find_relation_chunk(self, ent1, ent2):
        """检查两个实体间的直接关系"""
        # 检查水平关系
        for layer in self.layers:
            if layer.graph.has_edge(ent1, ent2):
                edge_data = layer.graph.get_edge_data(ent1, ent2)
                if "chunk_id" in edge_data:
                    return edge_data["chunk_id"]

        # 检查垂直关系
        if self.vertical_layer.graph.has_edge(ent1, ent2):
            edge_data = self.vertical_layer.graph.get_edge_data(ent1, ent2)
            if "chunk_id" in edge_data:
                return edge_data["chunk_id"]
        return None
    
    async def validate_and_repair_relations(
        self,
        repair: bool = True,
        verbose: bool = True
    ):
        logger.info("[Relation Validation Started]")
        report = {
            "semntic_duplicates": 0,
            "structural_duplicates": 0,
            "invariant_violations": 0,
        }

        # =====================================================
        # 标准 1：语义重复（signature 相同，id 不同）
        # =====================================================
        sig2rels = defaultdict(list)
        for rel in self.id2relation.values():
            sig2rels[semantic_relation_signature(rel)].append(rel)

        for sig, rels in sig2rels.items():
            if len(rels) <= 1:
                continue

            report["semantic_duplicates"] += len(rels) - 1

            if verbose:
                logger.warning(f"[Semantic Duplicate] sig={sig}, ids={[r.id for r in rels]}")

            if repair:
                keep = rels[0]
                for rel in rels[1:]:
                    await self._remove_relation(rel)

        # =====================================================
        # 标准 2：结构重复（layer.graph 内）
        # =====================================================
        def check_layer(layer: GraphLayer):
            nonlocal report
            seen = defaultdict(list)

            for u, v, k in layer.graph.edges(keys=True):
                rel = self.id2relation.get(k)
                if not rel:
                    continue

                sig = semantic_relation_signature(rel)
                seen[sig].append(k)

            for sig, rel_ids in seen.items():
                if len(rel_ids) <= 1:
                    continue

                report["structural_duplicates"] += len(rel_ids) - 1

                if verbose:
                    logger.warning(
                        f"[Structural Duplicate] layer={layer.level}, "
                        f"{u} -> {v}, sig={sig}, rel_ids={rel_ids}"
                    )

                if repair:
                    for rid in rel_ids[1:]:
                        rel = self.id2relation.get(rid)
                        if rel:
                            layer.remove_relation(rel)
     
        for layer in self.layers:
            check_layer(layer)
        check_layer(self.vertical_layer)


        if verbose:
            logger.info(
                "[KG Validation Finished] "
                + ", ".join(f"{k}={v}" for k, v in report.items())
            )

        return report
    
    
    def save(self, path: str):
        data = {
            "layers": self.layers,
            "vertical_layer": self.vertical_layer,
            "id2entity": self.id2entity,
            "id2relation": self.id2relation,
            "entity2id": self.entity2id,
            "name2entity": self.name2entity,
            "PROMPTS": self.PROMPTS,
            "valid_entries": self.valid_entries,
            "signature_set": self.relation_signature_set
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"KG data saved to {path}")

    @staticmethod
    def load(path: str, args = None) -> "HierarchicalMMKG":
        with open(path, "rb") as f:
            data = pickle.load(f)
        kg = HierarchicalMMKG(args = args)
        kg.layers = data["layers"]
        kg.vertical_layer = data["vertical_layer"]
        kg.id2entity = data["id2entity"]
        kg.id2relation = data["id2relation"]
        kg.entity2id = data["entity2id"]
        kg.name2entity = data["name2entity"]
        kg.PROMPTS = data["PROMPTS"]
        kg.valid_entries = data["valid_entries"]
        kg.relation_signature_set = data["signature_set"]
        return kg
    
    
