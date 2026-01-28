import networkx as nx
import json
from typing import List, Dict, Tuple, Optional, Set, Any, Literal,Callable
from src.VLMs import LLM, LLMMessage, LLMRegistry
import logging
import editdistance
from src.utils import safe_parser, async_task_runner
import yaml
import asyncio

from src.schema import Entity
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.KG.MMKG import HierarchicalMMKG
    from src.KG.GraphLayer import GraphLayer

from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("lightrag")

prompts_path = "src/KG/kg_prompts.yaml"
with open(prompts_path, 'r') as file:
    PROMPTS = yaml.load(file, Loader=yaml.FullLoader)


class EntityResolver:

    def __init__(
            self,
            vlm: str = "qwen3-vl-8b",
            embedding_model: SentenceTransformer = None,
            embedding_threshold: float = 0.8,
            editdistance_ratio: float = 0.35,
            top_k_candidates: int = 5,
            use_minhash: bool = False,
            batch_size: int = 10):
        self.llm = LLMRegistry.get(vlm)
        #self.tfidf_threshold = tfidf_threshold
        self.editdistance_ratio = editdistance_ratio
        self.top_k_candidates = top_k_candidates
        self.use_minhash = use_minhash
        self.batch_size = batch_size
        self.max_try = 3
        self.embedding_model = embedding_model
        self.embedding_threshold = embedding_threshold

    async def __call__(self,
                       kg: "HierarchicalMMKG",
                       level: int = None,
                       callback: Callable | None = None):
        """
        Entity Resolution
        Logic:
            1. Build textual corpus from nodes in the graph
            2. Generate candidate pairs using MinHash and editdistance
            3. LLM resolution
            4. Adjust the knowledge graph
        """
        if level is None:
            levels = range(len(kg.layers))
        else:
            levels = [level]
        
        for lvl in levels:
            layer = kg.get_layer(lvl)
            graph = layer.graph
            nodes = list(graph.nodes())

            corpus, node_texts = self._build_corpus(graph, nodes)
            candidate_pairs = self._generate_candidates(
                corpus, nodes, node_texts)
            #confirmed_pairs = self._filter_candidates(candidate_pairs, node_texts)
            confirmed_pairs = candidate_pairs

            logger.info(
                f"[EntityResolution] level {lvl} selected {len(confirmed_pairs)} pairs"
            )

            resolution_result = set()
            resolution_result_lock = asyncio.Lock()
            tasks = []
            for i in range(0, len(confirmed_pairs), self.batch_size):
                candidate_batch = (lvl,
                                    list(confirmed_pairs)[i:i +
                                                            self.batch_size])
                task = asyncio.create_task(
                    self._resolve_candidate_batch(kg, candidate_batch,
                                                    resolution_result,
                                                    resolution_result_lock))
                tasks.append(task)

            await async_task_runner(tasks, max_concurrent=8)

            if resolution_result:
                merge_pairs = [(a, b)
                                for a, b, decision in resolution_result
                                if decision == "yes"]
                await self._merge_connected_components(kg, merge_pairs)

    def _build_corpus(self, graph: nx.MultiDiGraph, nodes):
        corpus = []
        node_texts = {}
        for nid in nodes:
            name = str(graph.nodes[nid].get("name", ""))
            desc = str(graph.nodes[nid].get("attributes",
                                            {}).get("description", ""))
            text = name + " " + desc
            corpus.append(text)
            node_texts[nid] = (name, desc)
        return corpus, node_texts

    def _generate_candidates(self, corpus, nodes, node_texts):
        candidate_pairs = set()
        if self.use_minhash:
            from datasketch import MinHash, MinHashLSH
            lsh = MinHashLSH(threshold=0.8, num_perm=128)
            minhashes = {}
            for idx, text in enumerate(corpus):
                #entities = extract_named_entities(text)
                m = MinHash(num_perm=128)
                # for token in entities:
                #     m.update(token.encode('utf8'))
                for token in text.split():
                    m.update(token.encode('utf8'))
                minhashes[nodes[idx]] = m
                lsh.insert(nodes[idx], m)
            for nid in nodes:
                for similar_nid in lsh.query(minhashes[nid]):
                    if nid != similar_nid:
                        candidate_pairs.add(tuple(sorted((nid, similar_nid))))
                        logger.info(
                            f"MinHash candidate pair: {nid}{node_texts[nid]} - {similar_nid}{node_texts[similar_nid]}"
                        )
        else:
            if len(corpus) == 0:
                return set()
            
            all_embeddings = []
            for i in range(0, len(corpus), self.batch_size):
                batch_texts = corpus[i:i + self.batch_size]
                batch_emb = self.embedding_model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    batch_size=self.batch_size)
                all_embeddings.append(torch.tensor(batch_emb))

            if len(all_embeddings) == 0:
                return set()
            
            embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)

            sim_matrix = cosine_similarity(embeddings.numpy())
            for i in range(len(nodes)):
                sims = sorted(enumerate(sim_matrix[i]),
                              key=lambda x: x[1],
                              reverse=True)[:self.top_k_candidates]
                for j, score in sims:
                    if i != j and score >= self.embedding_threshold:
                        candidate_pairs.add(tuple(sorted(
                            (nodes[i], nodes[j]))))
        return candidate_pairs

    def _filter_candidates(self, candidate_pairs, node_texts):
        confirmed_pairs = set()
        for a, b in candidate_pairs:
            name_a, desc_a = node_texts[a]
            name_b, desc_b = node_texts[b]
            texta = name_a + " " + desc_a
            textb = name_b + " " + desc_b
            if is_similarity(texta, textb):
                confirmed_pairs.add((a, b))
        return confirmed_pairs

    async def _resolve_candidate_batch(self, graphlayer, candidate_batch,
                                       result_set, result_lock):
        layer_id, pairs = candidate_batch
        msgs = self._build_resolution_prompt(graphlayer, pairs)
        try:
            sys_msg, user_msg = msgs
            llm_response = await self.llm.async_chat(
                prompt=user_msg.content, system_prompt=sys_msg.content)
            parsed = safe_parser(llm_response)
            #logger.info(f"llm judgement{parsed}")
        except Exception as e:
            logging.error(f"Failed to parse LLM output: {e},")
            return
        if parsed:
            async with result_lock:  
                for idx, (a, b) in enumerate(pairs, 1):
                    decision = parsed.get(f"Pair {idx}", "").strip().lower()
                    result_set.add((a, b, decision))

    def _build_resolution_prompt(self, graphlayer: "GraphLayer",
                                 pairs) -> List[LLMMessage]:
        msgs = []
        system_prompt = PROMPTS['prompts']["entity_resolution"]
        msgs.append(LLMMessage(role="system", content=system_prompt))
        prompt_parts = []
        for idx, (a, b) in enumerate(pairs, 1):
            ent_a = graphlayer.get_entity(a)
            ent_b = graphlayer.get_entity(b)
            prompt_parts.append(
                f'Pair {idx}: name of entity A is : "{ent_a.name}", '
                f'name of entity B is : "{ent_b.name}", '
                f'description A: "{ent_a.attributes.get("description", "")}", '
                f'description B: "{ent_b.attributes.get("description", "")}"')
        Input_pairs = "\n".join(prompt_parts)
        msgs.append(LLMMessage(role="user", content=Input_pairs))
        return msgs

    async def _resolve_sharename_entity(
        self,
        new_ent: Entity,
        ent_list: List[Entity],
    ) -> Optional[int]:
        """
        Helper function for adding new entity or merge subgraphs when meeting share name entities

        Args:
            new_ent (Entity): _description_
            ent_list (List[Entity]): _description_

        Returns:
            if found the same entity, return the ori entity idx in the ent_list, 
            else return None
        """
        #msgs = []
        system_prompt = PROMPTS['prompts']["entity_resolution"]
        prompt_parts = []
        for idx, ent in enumerate(ent_list, 1):
            prompt_parts.append(
                f'Pair {idx}: name of entity A is : "{new_ent.name}", '
                f'name of entity B is : "{ent.name}", '
                f'description A: "{new_ent.attributes.get("description", "")}", '
                f'description B: "{ent.attributes.get("description", "")}"')
        Input_pairs = "\n".join(prompt_parts)

        for _ in range(self.max_try):
            try:
                llm_response = await self.llm.async_chat(
                    prompt=Input_pairs, system_prompt=system_prompt)
                parsed = safe_parser(llm_response)
                #logger.info(f"llm judgement{parsed}")
                for idx, ent in enumerate(ent_list, 1):
                    decision = parsed.get(f"Pair {idx}", "").strip().lower()
                    if decision == "yes":
                        return idx - 1
                return
            except Exception as e:
                logging.error(f"Failed to parse LLM output: {e}")

    async def _merge_connected_components(self, kg: "HierarchicalMMKG",
                                          merge_pairs):
        merge_graph = nx.Graph()
        merge_graph.add_edges_from(merge_pairs)
        for component in nx.connected_components(merge_graph):
            component = list(component)
            if len(component) <= 1:
                continue
           
            entities_with_levels = [(ent_id, kg.get_entity(ent_id).level)
                                    for ent_id in component]

            entities_with_levels.sort(key=lambda x: x[1])

            base_ent = entities_with_levels[0][0]

            other_ents = [ent_id for ent_id, _ in entities_with_levels[1:]]

            for other in other_ents:
                await kg.merge_entities(base_ent, other)


def has_digit_in_2gram_diff(a, b):
    def to_2gram_set(s):
        return {s[i:i+2] for i in range(len(s) - 1)}

    set_a = to_2gram_set(a)
    set_b = to_2gram_set(b)
    diff = set_a ^ set_b

    return any(any(c.isdigit() for c in pair) for pair in diff)

def is_similarity(a, b):
    if has_digit_in_2gram_diff(a, b):
        return False

    if editdistance.eval(a, b) <= min(len(a), len(b)) // 2:
        return True

    a, b = set(a), set(b)
    max_l = max(len(a), len(b))
    if max_l < 4:
        return len(a & b) > 1

    return len(a & b)*1./max_l >= 0.8
