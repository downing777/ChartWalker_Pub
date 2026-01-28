import random
import math
import json
import time
from typing import Dict, Any, List, Tuple, Set, Optional, Literal
from collections import Counter
import numpy as np
import yaml

from .MMKG import HierarchicalMMKG
from src.VLMs.resources_vlm import get_from_ks_openai
from src.utils import safe_parser

prompts_path = "src/KG/qa_prompts.yaml"

MoveType = Literal["horizontal", "vertical_down", "vertical_up"]
BranchVecMode = Literal["end", "avg"]

class PageRankWalker:
    """
    Source-Weighted PageRank (Monte Carlo Random Walk with Restart) + Constraint-aware path sampler
    + pruning  + reusable path pool + one-pass QA generation.

    Key additions:
    - Pruning: no repeated chunk_id(source) within a branch.
    - Postprocess: truncate a branch at first repeated source; optionally remove down->up same-source bounce.
    - Reuse: path pool with reuse limits; optional JSONL persistence.
    - One-pass QA: includes chart_index (image_index -> chunk_id) so model cites which image it used.
    """

    # ----------------------------- init -----------------------------
    def __init__(
        self,
        mmkg: HierarchicalMMKG,
        max_hops: int = 6,
        chart_map: Optional[dict] = None,
        text_map: Optional[dict] = None,
        # PageRank / RWR hyperparams
        restart_prob: float = 0.18,
        mc_walks: int = 5000,
        mc_len: int = 10,

        # weighting knobs
        rel_w: Optional[Dict[MoveType, float]] = None,
        novelty_lambda: float = 0.6,
        prefer_down_strength: float = 1.6,
        score_attract_beta: float = 0.35,
        # sampling knobs
        min_end_level: int = 2,
        model_name: str = "qwen3-vl-235b-a22b-thinking",
        # pruning knobs
        no_repeat_source_in_branch: bool = True,
        forbid_same_source_backtrack: bool = True,

        entity_emb: Optional[Dict[str, np.ndarray]] = None,      # ent_id -> vec
        relation_emb: Optional[Dict[str, np.ndarray]] = None,    # relation_id -> vec

        sem_edge_gamma: float = 0.0,     

        sem_use_positive_only: bool = True,  # True: max(0,cos)；False: cos 允许为负
    ):
        self.mmkg = mmkg
        self.max_hops = max_hops
        self.chart_map = chart_map or {}
        self.text_map = text_map or {}

        self.restart_prob = restart_prob
        self.mc_walks = mc_walks
        self.mc_len = mc_len

        self.novelty_lambda = novelty_lambda
        self.prefer_down_strength = prefer_down_strength
        self.score_attract_beta = score_attract_beta
        self.min_end_level = min_end_level

        self.model_name = model_name

        self.no_repeat_source_in_branch = no_repeat_source_in_branch
        self.forbid_same_source_backtrack = forbid_same_source_backtrack

        self.rel_w = rel_w or {
            "horizontal": 1.0,
            "vertical_down": 1.5,
            "vertical_up": 0.5,
        }

        self.entity_emb = entity_emb or {}
        self.relation_emb = relation_emb or {}

        self.sem_edge_gamma = sem_edge_gamma

        self.sem_use_positive_only = sem_use_positive_only

        # load prompts
        self.qa_prompt = {}
        try:
            with open(prompts_path, "r") as f:
                PROMPTS = yaml.safe_load(f) or {}
            self.qa_prompt = (PROMPTS.get("prompts") or {})
        except Exception:
            self.qa_prompt = {}

        # path pool for reuse
        self._path_pool: List[Dict[str, Any]] = []
        self._path_pool_reuse: List[int] = []

        # precompute source frequencies and node scores
        self.source_freq = self._compute_source_frequency()
        self.node_score = self._monte_carlo_pagerank()

    def _is_chart_chunk(self, chunk_id: Optional[str]) -> bool:
        if not chunk_id:
            return False
        s = str(chunk_id)
        return s.startswith("chart_") or ("chart" in s)

    def _fetch_source_image(self, chunk_id: str) -> Optional[str]:
        if not chunk_id:
            return None
        if self._is_chart_chunk(chunk_id):
            return self.chart_map.get(chunk_id)
        return None

    def _fetch_source_text(self, chunk_id: str) -> str:
        if not chunk_id:
            return ""
        return self.text_map.get(chunk_id, f"[No text found for {chunk_id}]")

    def _get_ent_emb(self, eid: str) -> Optional[np.ndarray]:
        v = self.entity_emb.get(eid)
        if v is None:
            return None
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def _sem_sim(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        s = self._cosine(a, b)
        if self.sem_use_positive_only:
            s = max(0.0, s)
        return s

    def _sem_multiplier(self, sim: Optional[float], gamma: float) -> float:
        """
        把相似度 sim 映射为权重倍数。
        """
        if sim is None or gamma <= 0:
            return 1.0
        return 1.0 + gamma * sim

    def _branch_vec(self, branch_edges: List[Dict[str, Any]], mode: BranchVecMode = "end") -> Optional[np.ndarray]:
        if not branch_edges:
            return None
        if mode == "end":
            end_id = branch_edges[-1]["to_id"]
            return self._get_ent_emb(end_id)

        # mode == "avg"
        vecs = []
        for e in branch_edges:
            v = self._get_ent_emb(e["to_id"])
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        return np.mean(np.stack(vecs, axis=0), axis=0)

    def _branch_sim(self, a: List[Dict[str, Any]], b: List[Dict[str, Any]], mode: BranchVecMode = "end") -> Optional[float]:
        va = self._branch_vec(a, mode=mode)
        vb = self._branch_vec(b, mode=mode)
        return self._sem_sim(va, vb)

    
    # ----------------------------- graph traversal primitives -----------------------------
    def _iter_edges_from(self, entity_id: str) -> List[Dict[str, Any]]:
        if entity_id not in self.mmkg.id2entity:
            return []

        ent = self.mmkg.get_entity(entity_id)
        level = ent.level
        out = []

        try:
            h_graph = self.mmkg.get_layer(level).graph
        except Exception:
            h_graph = None

        if h_graph is not None and h_graph.has_node(entity_id):
            for u, v, key, data in h_graph.edges(entity_id, keys=True, data=True):
                nbr = v if u == entity_id else u
                cid = (data.get("attributes", {}) or {}).get("chunk_id")
                out.append({
                    "cur": entity_id,
                    "next": nbr,
                    "move_type": "horizontal",
                    "key": key,
                    "source": cid,
                })

        # vertical
        try:
            v_graph = self.mmkg.vertical_layer.graph
        except Exception:
            v_graph = None

        if v_graph is not None and v_graph.has_node(entity_id):
            for u, v, key, data in v_graph.edges(entity_id, keys=True, data=True):
                nbr = v if u == entity_id else u
                if nbr not in self.mmkg.id2entity:
                    continue
                nbr_level = self.mmkg.get_entity(nbr).level
                move_type: MoveType = "vertical_down" if nbr_level > level else "vertical_up"
                cid = (data.get("attributes", {}) or {}).get("chunk_id")
                out.append({
                    "cur": entity_id,
                    "next": nbr,
                    "move_type": move_type,
                    "key": key,
                    "source": cid,
                })

        return out

    # ----------------------------- weighting -----------------------------
    def _source_rarity_weight(self, chunk_id: Optional[str]) -> float:
        if not chunk_id:
            return 1.0
        f = self.source_freq.get(chunk_id, 1)
        return 1.0 / math.sqrt(max(1, f))

    def _level_weight(self, cur: str, nxt: str, move_type: MoveType, prefer_down: bool) -> float:
        cur_lv = self.mmkg.get_entity(cur).level
        nxt_lv = self.mmkg.get_entity(nxt).level

        if prefer_down and move_type == "vertical_down":
            return 1.0 + 0.35 * (nxt_lv - cur_lv) * self.prefer_down_strength
        if prefer_down and move_type == "vertical_up":
            return 0.85
        return 1.0

    def _novelty_weight(self, chunk_id: Optional[str], used_sources: Set[str]) -> float:
        if not chunk_id:
            return 1.0
        return (1.0 + self.novelty_lambda) if chunk_id not in used_sources else 1.0

    def _edge_weight(
            self,
            cur: str,
            edge: Dict[str, Any],
            used_sources: Set[str],
            prefer_down: bool,
            use_node_score: bool = True,
            anchor_emb: Optional[np.ndarray] = None,   # NEW: 主题 anchor（采样时传）
        ) -> float:
            move_type: MoveType = edge["move_type"]
            nxt = edge["next"]
            src = edge.get("source")

            w = 1.0
            w *= self.rel_w.get(move_type, 1.0)
            w *= self._level_weight(cur, nxt, move_type, prefer_down=prefer_down)
            w *= self._source_rarity_weight(src)
            w *= self._novelty_weight(src, used_sources)

            # 语义项 1：cur -> nxt 邻接跳转的连贯性（影响 pagerank 与采样）
            if self.sem_edge_gamma > 0:
                cur_emb = self._get_ent_emb(cur)
                nxt_emb = self._get_ent_emb(nxt)
                sim = self._sem_sim(cur_emb, nxt_emb)
                w *= self._sem_multiplier(sim, self.sem_edge_gamma)


            if use_node_score and self.score_attract_beta > 0:
                s = getattr(self, "node_score", {}).get(nxt, 0.0)
                w *= (1.0 + self.score_attract_beta * s)

            return max(1e-12, w)

    def _edge_weight_sampling(
            self,
            cur: str,
            edge: Dict[str, Any],
            used_sources: Set[str],
            anchor_emb: Optional[np.ndarray] = None,
        ) -> float:
            """
            采样时专用的简化权重计算。
            只保留动态因素和PageRank分数，移除已在PageRank中考虑的静态因素。
            
            保留的因素：
            - node_score: PageRank结果（核心）
            - _level_weight: 采样策略（prefer_down=True）
            - _novelty_weight: 动态新颖性（取决于当前分支的used_sources）
            - anchor_emb: 主题锚点（如果启用）
            
            移除的因素（已在PageRank中考虑）：
            - rel_w: 边类型权重（静态）
            - _source_rarity_weight: 源稀有度（静态）
            - sem_edge_gamma: 语义相似度（如果PageRank时也用了）
            """
            nxt = edge["next"]
            src = edge.get("source")
            move_type: MoveType = edge["move_type"]

            w = 1.0
            
            # 1. PageRank分数（核心）
            if self.score_attract_beta > 0:
                s = getattr(self, "node_score", {}).get(nxt, 0.0)
                w *= (1.0 + self.score_attract_beta * s)
            
            # 2. 层级偏好（采样策略：prefer_down=True）
            w *= self._level_weight(cur, nxt, move_type, prefer_down=True)
            
            # 3. 动态新颖性（取决于当前分支的used_sources）
            w *= self._novelty_weight(src, used_sources)
            
            # 4. 主题锚点（如果启用）
            if anchor_emb is not None and self.sem_edge_gamma > 0:
                nxt_emb = self._get_ent_emb(nxt)
                if nxt_emb is not None:
                    sim = self._sem_sim(anchor_emb, nxt_emb)
                    w *= self._sem_multiplier(sim, self.sem_edge_gamma)

            return max(1e-12, w)

    # ----------------------------- PageRank: Monte Carlo RWR -----------------------------
    def _compute_source_frequency(self) -> Dict[str, int]:
        freq = Counter()
        for eid in self.mmkg.id2entity.keys():
            for e in self._iter_edges_from(eid):
                cid = e.get("source")
                if cid:
                    freq[cid] += 1
        return dict(freq)

    def _seed_candidates(self, k: int = 400) -> List[str]:
        candidates = list(self.mmkg.id2entity.keys())
        if not candidates:
            return []
        scored = []
        for eid in candidates:
            deg = len(self._iter_edges_from(eid))
            scored.append((eid, deg))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in scored[: max(1, k)]]

    def _monte_carlo_pagerank(self) -> Dict[str, float]:
        seeds = self._seed_candidates(k=400)
        if not seeds:
            return {}

        visit = Counter()

        for _ in range(self.mc_walks):
            seed = random.choice(seeds)
            cur = seed
            used_sources: Set[str] = set()

            for _t in range(self.mc_len):
                visit[cur] += 1

                if random.random() < self.restart_prob:
                    cur = seed
                    used_sources.clear()
                    continue

                edges = self._iter_edges_from(cur)
                if not edges:
                    cur = seed
                    used_sources.clear()
                    continue

                weights = [
                    self._edge_weight(cur, e, used_sources, prefer_down=False, use_node_score=False)
                    for e in edges
                ]
                nxt = random.choices([e["next"] for e in edges], weights=weights, k=1)[0]

                for e in edges:
                    if e["next"] == nxt and e.get("source"):
                        used_sources.add(e["source"])
                        break
                cur = nxt

        total = sum(visit.values()) or 1
        return {eid: cnt / total for eid, cnt in visit.items()}

    def _sample_start_node(
        self,
        topk: int = 300,
        seed_level: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        exclude = exclude or set()
        if seed_level is None:
            seed_level = self.seed_level_default

        # candidates from specified layer
        candidates = []
        if seed_level is not None:
            try:
                layer = self.mmkg.get_layer(seed_level)
                candidates = list(layer.entity_index.keys())
            except Exception:
                candidates = []

        if not candidates:
            candidates = list(self.mmkg.id2entity.keys())

        candidates = [eid for eid in candidates if eid not in exclude]
        if not candidates:
            return None

        # topk by node_score
        scored = [(eid, self.node_score.get(eid, 1e-12)) for eid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        pool = [eid for eid, _ in scored[: min(topk, len(scored))]]

        # weights = node_score * semantic(anchor,eid)
        weights = []
        for eid in pool:
            base = max(1e-12, self.node_score.get(eid, 1e-12))
            weights.append(base)

        return random.choices(pool, weights=weights, k=1)[0]

    # ----------------------------- path sampling -----------------------------
    def _render_edge(
        self,
        from_id: str,
        to_id: str,
        move_type: str,
        source: Optional[str],
        rel_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "from_id": from_id,
            "to_id": to_id,
            "from": self.mmkg.get_entity(from_id).name,
            "to": self.mmkg.get_entity(to_id).name,
            "type": move_type,
            "source": source,
            "key": rel_key,
            "from_level": self.mmkg.get_entity(from_id).level,
            "to_level": self.mmkg.get_entity(to_id).level,
        }

    def _branch_sources(self, branch_edges: List[Dict[str, Any]]) -> Set[str]:
        s = set()
        for e in branch_edges:
            cid = e.get("source")
            if cid:
                s.add(cid)
        return s
    
    def _edge_sig(self, e: Dict[str, Any]) -> tuple:
        return (e.get("from_id"), e.get("to_id"), e.get("type"), e.get("source"))

    def _is_prefix_branch(self, a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
        """
        Return True if a is a prefix of b (or equal), based on edge signatures.
        """
        if not a or len(a) > len(b):
            return False
        sa = [self._edge_sig(e) for e in a]
        sb = [self._edge_sig(e) for e in b]
        return sb[:len(sa)] == sa

    def _walk_branch(
        self,
        start_eid: str,
        max_hops: Optional[int] = None,
        prefer_down: bool = True,
        require_chart_end: bool = True,
        avoid_nodes: Optional[Set[str]] = None,
        anchor_emb: Optional[np.ndarray] = None,   # NEW
    ) -> Tuple[List[Dict[str, Any]], Set[str], Optional[str], str]:

        """
        One stochastic branch walk with pruning:
        - No repeated source within a branch (if enabled).
        - Optional forbid immediate backtrack on same source.
        """
        if max_hops is None:
            max_hops = self.max_hops
        avoid_nodes = avoid_nodes or set()

        cur = start_eid
        prev_node: Optional[str] = None  # one-step back node id
        used_sources: Set[str] = set()
        branch_edges: List[Dict[str, Any]] = []
        last_source: Optional[str] = None

        for _ in range(max_hops):
            edges = self._iter_edges_from(cur)
            if not edges:
                break

            cand, wts = [], []
            for e in edges:
                nxt = e["next"]
                src = e.get("source")

                if nxt in avoid_nodes:
                    continue

                if self.no_repeat_source_in_branch and src and src in used_sources:
                    continue

                if self.forbid_same_source_backtrack and prev_node is not None:
                    if nxt == prev_node and src and last_source and src == last_source:
                        continue

                # 使用简化的采样权重：只保留动态因素和PageRank分数
                w = self._edge_weight_sampling(cur, e, used_sources, anchor_emb=anchor_emb)
                cand.append(e)
                wts.append(w)

            if not cand:
                break

            chosen = random.choices(cand, weights=wts, k=1)[0]
            nxt = chosen["next"]
            src = chosen.get("source")

            # record edge
            branch_edges.append(self._render_edge(cur, nxt, chosen["move_type"], src, chosen.get("key")))

            # update state
            if src:
                used_sources.add(src)
            prev_node = cur
            cur = nxt
            last_source = src

            # # stop if sufficiently deep and (optionally) ended at chart evidence
            # cur_lv = self.mmkg.get_entity(cur).level
            # if cur_lv >= self.min_end_level:
            #     if not require_chart_end:
            #         break
            #     if last_source and self._is_chart_chunk(last_source):
            #         break

        if branch_edges:
            end_id = branch_edges[-1]["to_id"]
            last_source = branch_edges[-1].get("source")
            used_sources = self._branch_sources(branch_edges)
        else:
            end_id = start_eid
            last_source = None
            used_sources = set()

        return branch_edges, used_sources, last_source, end_id

    def sample_path(
        self,
        branches: int = 2,
        min_sources: int = 2,
        require_charts: int = 2,
        retries: int = 30,
        branch_max_hops: Optional[int] = None,
        max_branch_attempts: int = 20,

        # keep:
        seed_level: Optional[int] = None,
        branch_sem_tau: Optional[float] = 0.45,
        branch_vec_mode: BranchVecMode = "end",
    ) -> Optional[Dict[str, Any]]:

        for _ in range(retries):
            # sample center entity
            center_id = self._sample_start_node(seed_level=seed_level)
            if not center_id:
                return None
            anchor_emb = self._get_ent_emb(center_id)

            all_paths: List[List[Dict[str, Any]]] = []
            all_sources: List[Set[str]] = []
            endpoints = []
            used_end_ids: Set[str] = set()

            # branch semantic anchor (from first accepted branch)
            first_branch_vec: Optional[np.ndarray] = None

            attempts = 0
            while len(all_paths) < branches and attempts < max_branch_attempts:
                attempts += 1

                # IMPORTANT: always start from center_id
                start_eid = center_id

                branch_edges, used_src, last_src, end_id = self._walk_branch(
                    start_eid=start_eid,
                    max_hops=branch_max_hops or self.max_hops,
                    prefer_down=True,
                    require_chart_end=True,
                    avoid_nodes=used_end_ids,
                    anchor_emb=anchor_emb,
                )

                if len(branch_edges) < 1:
                    continue

                # prefix reject
                covered = False
                for prev in all_paths:
                    if self._is_prefix_branch(branch_edges, prev) or self._is_prefix_branch(prev, branch_edges):
                        covered = True
                        break
                if covered:
                    continue

                # semantic coherence across branches
                if branch_sem_tau is not None and self.entity_emb:
                    if not all_paths:
                        first_branch_vec = self._branch_vec(branch_edges, mode=branch_vec_mode)
                    else:
                        cur_vec = self._branch_vec(branch_edges, mode=branch_vec_mode)
                        if first_branch_vec is not None and cur_vec is not None:
                            sim = self._sem_sim(first_branch_vec, cur_vec)
                            if sim is None or sim < branch_sem_tau:
                                continue

                # accept branch
                used_end_ids.add(end_id)
                all_paths.append(branch_edges)
                all_sources.append(used_src)
                endpoints.append({
                    "id": end_id,
                    "name": self.mmkg.get_entity(end_id).name,
                    "level": self.mmkg.get_entity(end_id).level,
                    "last_source": last_src,
                })

            if len(all_paths) != branches:
                continue

            merged_sources = set().union(*all_sources)
            if len(merged_sources) < min_sources:
                continue

            chart_sources = [s for s in merged_sources if self._is_chart_chunk(s)]
            if len(set(chart_sources)) < require_charts:
                continue

            nodes = self._collect_nodes_from_paths(center_id, all_paths)
            return {
                "center_id": center_id,
                "center": self.mmkg.get_entity(center_id).name,
                "paths": all_paths,
                "sources": list(merged_sources),
                "endpoints": endpoints,
                "nodes": nodes,
            }

        return None

    def _collect_nodes_from_paths(self, center_id: str, paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        node_ids = {center_id}
        for branch in paths:
            for e in branch:
                node_ids.add(e["from_id"])
                node_ids.add(e["to_id"])

        items = []
        for nid in node_ids:
            ent = self.mmkg.get_entity(nid)
            items.append((nid, ent.level, self.node_score.get(nid, 0.0)))

        items.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [{"id": nid, "name": self.mmkg.get_entity(nid).name, "level": self.mmkg.get_entity(nid).level}
                for nid, _, _ in items]

    # ----------------------------- QA answer verification -----------------------------
    def verify_qa_answer(
        self,
        qa_entry: Dict[str, Any],
        max_evidence: int = 4,
        temperature: float = 0.3,
    ) -> bool:
        """
        Verify if the generated answer is correct based on the provided evidence.
        
        Returns:
            bool: True if answer is correct, False otherwise
        """
        # Extract information from QA entry
        question = qa_entry.get("question", "")
        answer = qa_entry.get("answer", "")
        explanation = qa_entry.get("explanation", "")
        evidence = qa_entry.get("evidence", [])
        evidence_charts = qa_entry.get("evidence_charts", [])
        
        if not question or not answer:
            print("[verify_qa_answer] 问题或答案为空，验证失败")
            return False
        
        # Collect evidence items
        evidence_pool = evidence[:max_evidence]
        chart_items, text_items = self._collect_evidence_items(evidence_pool)
        
        chart_images = [it["image_path"] for it in chart_items]
        chart_chunk_ids = [it["chunk_id"] for it in chart_items]
        
        # Build text evidence string
        if text_items:
            text_evidence_str = "\n\n".join([f"[{it['chunk_id']}]\n{it['text']}" for it in text_items])
        else:
            text_evidence_str = "(no text evidence provided)"
        
        # Build chart index string
        if chart_chunk_ids:
            chart_index_str = "\n".join([f"[{i}] chunk_id={cid}" for i, cid in enumerate(chart_chunk_ids)])
        else:
            chart_index_str = "(no chart images provided)"
        
        # Create verification prompt
        verification_prompt = f"""You are a fact-checker verifying the correctness of a question-answer pair based on provided evidence.

Question: {question}

Proposed Answer: {answer}

Explanation provided: {explanation}

Evidence Charts (image_index -> chunk_id):
{chart_index_str}

Text Evidence:
{text_evidence_str}

Your task:
1. Carefully examine the question, proposed answer, and all provided evidence (charts and text).
2. Determine if the proposed answer is CORRECT based on the evidence.
3. Consider:
   - Does the answer accurately reflect the data shown in the charts?
   - Is the answer logically consistent with the text evidence?
   - Are there any factual errors or misinterpretations?
   - Is the answer complete and properly addresses the question?

Output format (STRICT JSON only, no markdown, no extra text):
```json
{{
  "is_correct": true or false,
  "reasoning": "Brief explanation of why the answer is correct or incorrect",
  "confidence": "high" or "medium" or "low"
}}
```

Note: Only mark as correct if the answer is factually accurate and well-supported by the evidence. If there are any doubts, mark as incorrect."""

        try:
            print(f"[verify_qa_answer] 开始验证答案...")
            print(f"[verify_qa_answer] 问题: {question[:100]}...")
            print(f"[verify_qa_answer] API调用开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            response = get_from_ks_openai(
                prompt=verification_prompt,
                model=self.model_name,
                image_paths=chart_images,
                thinking=True,
                temperature=temperature,
            )
            
            print(f"[verify_qa_answer] API调用完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            result = safe_parser(response)
            
            if isinstance(result, dict):
                is_correct = result.get("is_correct", False)
                reasoning = result.get("reasoning", "")
                confidence = result.get("confidence", "low")
                
                print(f"[verify_qa_answer] 验证结果: {'✓ 正确' if is_correct else '✗ 错误'}")
                print(f"[verify_qa_answer] 置信度: {confidence}")
                print(f"[verify_qa_answer] 推理: {reasoning[:200]}...")
                
                return is_correct
            else:
                print(f"[verify_qa_answer] 解析失败，默认认为答案错误")
                print(f"[verify_qa_answer] 原始响应: {response[:500]}...")
                return False
                
        except Exception as e:
            print(f"[verify_qa_answer] 验证过程出错: {repr(e)}")
            # 如果验证过程出错，默认认为答案错误，要求重新生成
            return False

    # ----------------------------- evidence packing (keeps chart mapping) -----------------------------
    def _collect_evidence_items(self, chunk_ids: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Returns:
          chart_items: [{"chunk_id":..., "image_path":...}, ...]  (order preserved)
          text_items:  [{"chunk_id":..., "text":...}, ...]
        """
        chart_items: List[Dict[str, str]] = []
        text_items: List[Dict[str, str]] = []
        for cid in chunk_ids:
            img = self._fetch_source_image(cid)
            if img:
                chart_items.append({"chunk_id": cid, "image_path": img})
            else:
                t = self._fetch_source_text(cid)
                if t:
                    text_items.append({"chunk_id": cid, "text": t})
        return chart_items, text_items

    def _format_paths_for_prompt(self, paths: List[List[Dict[str, Any]]], center_id: Optional[str] = None, endpoints: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format paths with enhanced structure highlighting:
        - Center entity (starting point)
        - Branch depth progression
        - Endpoints (final entities reached)
        - Level transitions (vertical_up/down vs horizontal)
        """
        lines = []
        
        # Add center entity info if provided
        if center_id:
            center_ent = self.mmkg.get_entity(center_id)
            lines.append(f"Center Entity: {center_ent.name} (Level {center_ent.level})")
            lines.append("")
        
        for i, branch in enumerate(paths):
            if not branch:
                lines.append(f"[Branch {i}] (empty)")
                continue
            
            # Calculate branch depth info
            start_level = branch[0].get('from_level', 0)
            end_level = branch[-1].get('to_level', 0)
            depth_change = end_level - start_level
            num_hops = len(branch)
            
            # Branch header with depth info
            depth_indicator = "↓" if depth_change > 0 else "↑" if depth_change < 0 else "→"
            lines.append(f"[Branch {i}] {depth_indicator} {num_hops} hops, Level {start_level} → Level {end_level}")
            
            # Add endpoint info if available
            if endpoints and i < len(endpoints):
                ep = endpoints[i]
                lines.append(f"  Endpoint: {ep.get('name', 'N/A')} (Level {ep.get('level', 'N/A')}, source: {ep.get('last_source', 'N/A')})")
            
            # Format edges with emphasis on level transitions
            for j, e in enumerate(branch):
                from_level = e.get('from_level', 0)
                to_level = e.get('to_level', 0)
                move_type = e.get('type', '')
                
                # Indent based on depth
                indent = "  " + "  " * min(j, 3)  # Limit indentation depth
                
                # Highlight level transitions
                if move_type == "vertical_down":
                    level_marker = f"↓L{from_level}→L{to_level}"
                elif move_type == "vertical_up":
                    level_marker = f"↑L{from_level}→L{to_level}"
                else:
                    level_marker = f"→L{from_level}"
                
                src_info = f"[src={e.get('source', 'N/A')}]" if e.get('source') else ""
                lines.append(
                    f"{indent}{e['from']} --{move_type} {level_marker} {src_info}--> {e['to']}"
                )
            
            lines.append("")  # Blank line between branches
        
        return "\n".join(lines)

    # ----------------------------- QA generation (one-pass) -----------------------------
    def gen_QA(
        self,
        path_info: Dict[str, Any],
        max_evidence: int = 4,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        One-pass QA generation, with chart_index mapping so model can cite image_index + chunk_id.
        """
        # --- 1) Preselect evidence (charts first) ---
        sources = path_info.get("sources", []) or []
        chart_sources = [s for s in sources if self._is_chart_chunk(s)]
        other_sources = [s for s in sources if s not in chart_sources]

        chart_sources = list(dict.fromkeys(chart_sources))
        other_sources = list(dict.fromkeys(other_sources))

        evidence_pool = (chart_sources + other_sources)[:max_evidence]

        # --- 2) Collect evidence payloads (preserve mapping) ---
        chart_items, text_items = self._collect_evidence_items(evidence_pool)

        chart_images = [it["image_path"] for it in chart_items]
        chart_chunk_ids = [it["chunk_id"] for it in chart_items]
        chunk2img_index = {cid: idx for idx, cid in enumerate(chart_chunk_ids)}

        if chart_chunk_ids:
            chart_index_str = "\n".join([f"[{i}] chunk_id={cid}" for i, cid in enumerate(chart_chunk_ids)])
        else:
            chart_index_str = "(no chart images provided)"

        if text_items:
            text_evidence_str = "\n\n".join([f"[{it['chunk_id']}]\n{it['text']}" for it in text_items])
        else:
            text_evidence_str = "(no text evidence provided)"

        paths_str = self._format_paths_for_prompt(
            path_info.get("paths", []),
            center_id=path_info.get("center_id"),
            endpoints=path_info.get("endpoints")
        )

        tmpl = self.qa_prompt.get("QA_OnePass_easy")
        prompt = tmpl.format(
            reasoning_paths=paths_str,
            available_sources=json.dumps(evidence_pool, ensure_ascii=False),
            chart_index=chart_index_str,
            text_evidence=text_evidence_str,
        )
        out = None
        last_resp = None
        last_err = None

        for _try in range(3):
            try:
                print("PROMPT TEST")
                print("prompt=", prompt)
                print("chart_images=", chart_images)
                print(f"[gen_QA] 准备调用API (尝试 {_try+1}/3), 模型: {self.model_name}")
                print(f"[gen_QA] API调用开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                last_resp = get_from_ks_openai(
                    prompt=prompt,
                    model=self.model_name,
                    image_paths=chart_images,
                    thinking=True,
                    temperature=temperature,
                )
                print(f"[gen_QA] API调用完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                out = safe_parser(last_resp)
                print("[gen_QA] model response=", last_resp)
                if isinstance(out, dict) and out:
                    break
                else:
                    out = None
            except Exception as e:
                last_err = repr(e)
                print(f"[gen_QA] parse failed try={_try+1}/3 err={last_err}")
                continue

        if not isinstance(out, dict):
            out = {
                "_raw_response": last_resp,
                "_parse_error": last_err,
            }
            return {
                 "path_info": path_info,
                "evidence_used": evidence_pool,
                "chart_index": chunk2img_index,
                **out,
            }

        # --- 3) Validate / defaults ---
        valid_types = {"FactCheck", "Manipulation", "Comparison", "Trend", "Analysis"}
        if out.get("query_type") not in valid_types:
            out["query_type"] = "Comparison"

        out.setdefault("question", "")
        out.setdefault("answer", "")
        out.setdefault("explanation", "")

        # evidence must be subset of evidence_pool
        ev = out.get("evidence")
        if not isinstance(ev, list):
            ev = evidence_pool.copy()
        else:
            ev = [cid for cid in ev if cid in set(evidence_pool)]
            if not ev:
                ev = evidence_pool.copy()
        out["evidence"] = ev

        # evidence_charts must match chart_index mapping
        evc = out.get("evidence_charts")
        if not isinstance(evc, list):
            evc = []
        cleaned_evc = []
        used_chart_cids = set()
        for item in evc:
            if not isinstance(item, dict):
                continue
            cid = item.get("chunk_id")
            if cid not in chunk2img_index:
                continue
            cleaned_evc.append({"chunk_id": cid, "image_index": chunk2img_index[cid]})
            used_chart_cids.add(cid)

        # enforce: if >=2 charts available in evidence_pool, ensure at least 2 distinct cited charts
        # if len(chart_chunk_ids) >= 2 and len(used_chart_cids) < 2:
        #     fallback = chart_chunk_ids[:2]
        #     cleaned_evc = [{"chunk_id": cid, "image_index": chunk2img_index[cid]} for cid in fallback]
        #     for cid in fallback:
        #         if cid not in out["evidence"]:
        #             out["evidence"].append(cid)

        out["evidence_charts"] = cleaned_evc

        # --- 4) Calculate difficulty scores ---
        # 1. 图片数量：基于 cleaned_evc（模型实际使用的图表）
        num_charts = len(cleaned_evc)
        
        # 2. 计算hop数（从center到实际使用的证据的最长路径）
        def calculate_max_hops(path_info, used_evidence):
            """
            计算从center节点到实际使用的证据的最长hop数
            返回：从center到任意使用证据的最大边数
            """
            center_id = path_info["center_id"]
            paths = path_info["paths"]
            max_hops = 0
            
            # 对于每个使用的证据，找到它在路径中的位置
            for evidence_id in used_evidence:
                evidence_hops = 0
                
                # 遍历所有分支，找到包含该证据的分支
                for branch in paths:
                    branch_hops = 0
                    found_evidence = False
                    
                    for edge in branch:
                        branch_hops += 1
                        if edge.get("source") == evidence_id:
                            found_evidence = True
                            break  # 找到就停止，取这个分支的hop数
                    
                    if found_evidence:
                        evidence_hops = max(evidence_hops, branch_hops)
                
                # 取所有证据中的最大hop数
                if evidence_hops > 0:
                    max_hops = max(max_hops, evidence_hops)
            
            return max_hops
        
        used_evidence_set = set(out.get("evidence", []))
        max_hops = calculate_max_hops(path_info, used_evidence_set)
        
        # 3. 主观评分：从模型返回中提取
        difficulty_subjective = out.get("difficulty_subjective")
        if difficulty_subjective is None:
            difficulty_subjective = 2  # 默认中等难度
        else:
            # 确保在1-3范围内
            try:
                difficulty_subjective = max(1, min(3, int(difficulty_subjective)))
            except (ValueError, TypeError):
                difficulty_subjective = 2
        
        # 计算总分
        difficulty_total = num_charts + max_hops + difficulty_subjective
        
        # 添加到输出
        out["difficulty"] = {
            "num_charts": num_charts,
            "num_hops": max_hops,
            "subjective": difficulty_subjective,
            "total": difficulty_total
        }

        return {
            "path_info": path_info,
            "evidence_used": evidence_pool,
            "chart_index": chunk2img_index,
            **out,
        }

    # ----------------------------- path reuse: pool + batch generation -----------------------------
    def ensure_path_pool(
        self,
        pool_size: int,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        dedup: bool = True,
        max_attempts: Optional[int] = None,
    ) -> None:
        """
        Fill/extend the internal path pool to at least pool_size.
        """
        sampling_kwargs = sampling_kwargs or {}
        if len(self._path_pool) >= pool_size:
            return

        need = pool_size - len(self._path_pool)
        seen = set()

        if dedup:
            for p in self._path_pool:
                seen.add(self.path_signature(p))

        attempts = 0
        max_attempts = max_attempts or (need * 80)

        while need > 0 and attempts < max_attempts:
            attempts += 1
            p = self.sample_path(**sampling_kwargs) if sampling_kwargs else self.sample_path()
            if not p:
                continue
            if dedup:
                sig = self.path_signature(p)
                if sig in seen:
                    continue
                seen.add(sig)
            self._path_pool.append(p)
            self._path_pool_reuse.append(0)
            need -= 1

    def path_signature(self, path_info: Dict[str, Any]) -> str:
        """
        Signature for dedup: sort edges by (from_id,to_id,type,source) across all branches.
        """
        parts = []
        for b in path_info.get("paths", []):
            for e in b:
                parts.append(f"{e.get('from_id')}|{e.get('to_id')}|{e.get('type')}|{e.get('source')}")
        parts.sort()
        return "||".join(parts)

    def next_path_from_pool(
        self,
        max_reuse_per_path: int = 3,
        resample_prob: float = 0.25,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get a path from pool with reuse limit. With probability resample_prob, add a new path to pool.
        """
        sampling_kwargs = sampling_kwargs or {}

        # occasionally add fresh path to pool
        if random.random() < resample_prob:
            pnew = self.sample_path(**sampling_kwargs) if sampling_kwargs else self.sample_path()
            if pnew:
                self._path_pool.append(pnew)
                self._path_pool_reuse.append(0)

        # pick candidate with remaining reuse budget
        candidates = [i for i, c in enumerate(self._path_pool_reuse) if c < max_reuse_per_path]
        if not candidates:
            # reset reuse counts if exhausted
            self._path_pool_reuse = [0 for _ in self._path_pool]
            candidates = list(range(len(self._path_pool)))

        idx = random.choice(candidates)
        self._path_pool_reuse[idx] += 1
        return self._path_pool[idx]

    def gen_QA_dataset(
        self,
        n_qas: int,
        pool_size: int = 200,
        max_reuse_per_path: int = 3,
        resample_prob: float = 0.25,
        max_evidence: int = 4,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        dedup_pool: bool = True,
        verify_answer: bool = True,
        max_retry_per_qa: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Batch generate QA with reusable sampled paths.
        
        Args:
            verify_answer: If True, verify each generated answer and retry if incorrect
            max_retry_per_qa: Maximum number of retries per QA if verification fails
        """
        sampling_kwargs = sampling_kwargs or {}
        self.ensure_path_pool(pool_size=pool_size, sampling_kwargs=sampling_kwargs, dedup=dedup_pool)

        dataset = []
        for i in range(n_qas):
            print(f"\n[gen_QA_dataset] 进度: {i+1}/{n_qas}")
            
            qa = None
            verified = False
            retry_count = 0
            
            while not verified and retry_count < max_retry_per_qa:
                # Get a path from pool
                p = self.next_path_from_pool(
                    max_reuse_per_path=max_reuse_per_path,
                    resample_prob=resample_prob,
                    sampling_kwargs=sampling_kwargs,
                )
                
                if retry_count == 0:
                    print(f"[gen_QA_dataset] 开始生成第 {i+1} 个QA对...")
                else:
                    print(f"[gen_QA_dataset] 重新生成第 {i+1} 个QA对 (尝试 {retry_count+1}/{max_retry_per_qa})...")
                
                # Generate QA
                qa = self.gen_QA(p, max_evidence=max_evidence, temperature=temperature)
                
                # Verify answer if enabled
                if verify_answer:
                    is_correct = self.verify_qa_answer(qa, max_evidence=max_evidence, temperature=0.3)
                    if is_correct:
                        verified = True
                        print(f"[gen_QA_dataset] ✓ 第 {i+1} 个QA对验证通过")
                    else:
                        retry_count += 1
                        if retry_count < max_retry_per_qa:
                            print(f"[gen_QA_dataset] ✗ 第 {i+1} 个QA对验证失败，将重新生成...")
                        else:
                            print(f"[gen_QA_dataset] ✗ 第 {i+1} 个QA对验证失败，已达到最大重试次数，保留该QA对")
                            verified = True  # 达到最大重试次数，仍然保留
                else:
                    verified = True  # 如果未启用验证，直接通过
            
            if qa:
                dataset.append(qa)
                print(f"[gen_QA_dataset] 第 {i+1} 个QA对生成完成")
            else:
                print(f"[gen_QA_dataset] ⚠ 警告: 第 {i+1} 个QA对生成失败，跳过")
        
        return dataset

    @staticmethod
    def dump_paths_jsonl(paths: List[Dict[str, Any]], out_path: str) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    @staticmethod
    def load_paths_jsonl(in_path: str) -> List[Dict[str, Any]]:
        out = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
