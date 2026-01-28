import gym
from gym import spaces
from typing import List, Dict, Tuple, Optional, Any, Union, Set
import numpy as np
from enum import Enum
from src.KG import HierarchicalMMKG
from src.VLMs import LLMRegistry
import json
from PIL import Image
import os
from openai import OpenAI
import random


CHARTMRAG_IMAGE_DIR = "/share/project/tangning/Dataset/chart-mrag/images"
CHARTQA_IMAGE_DIR = "/share/project/tangning/Dataset/chartqa_pro/chartqa_images"
TEXT_CORPUS = "/share/project/tangning/Dataset/chart-mrag/merge_text_corpus.jsonl"

# BASE_URL = "http://localhost:10086/v1"
# API_KEY = "empty"

class ActionType(Enum):
    START = 0
    EDGE_SEARCH = 1
    #EXTRACT_INFO = 2
    MOVE = 2
    BACKWARD = 3
    STOP = 4


class GymMMKGEnv(gym.Env):
    """
    Knowledge Graph Navigation Env with name-based interaction.
    Agent actions and observations use human-readable entity names instead of IDs.
    Knowledge Graph Navigation Env with hierarchical layers support. 
    Actions: 
    - 0 START: choose a start entity from provided candidates (or random) 
    - 1 EDGE_SEARCH: inspect a chosen relation's source (e.g., original image / chunk) 
    - 2 EXTRACT_INFO
    - 3 MOVE: move to neighbor entity via chosen relation 
    - 4 BACKWARD: go back to previously visited node (one-step or to a visited entity) 
    - 5 STOP: finish and judge (LLM or graph-based) 
    --- Action input accepted forms in step(action): 
    - int (action_type only, no arg) 
    - tuple/list: (action_type_int, arg) 
    - dict: {"type": action_type_int, "arg": arg} 
    
    arg semantics: 
    - For START: arg can be entity_id (str) or index (int) within start_candidate
    - For EDGE_SEARCH or MOVE: arg can be relation target entity id (str), relation id (str), or index (int) into available_relations 
    - For BACKWARD: arg can be None (pop one), entity_id (str) or visited index (int) 
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 kg_path: str,
                 query: str,
                 answer: str,
                 start_candidates: List[str],
                 gt_sources: List[str] = None, 
                 gt_entities: List[str] = None,
                 max_steps: int = 20,
                 reward_config: Optional[Dict] = None,
                 model_name: str = None,
                 **kwargs):
        super().__init__()
        self.kg: HierarchicalMMKG = HierarchicalMMKG.load(kg_path)
        self.max_steps = max_steps
        self.vert_layer = self.kg.get_layer(level="vert")
        self.text_map = {}
        with open(TEXT_CORPUS, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.text_map[data["id"]] = data["text"]

        self.reward_config = {
            "illegal_penalty": -1.0,
            "step_penalty": 0.0,
            "retrieve_reward": 2.0,
            "move_reward": 2.0,
            "success_reward": 5.0,
            "answer_reward": 5.0
        }
        if reward_config:
            self.reward_config.update(reward_config)

        self.action_space = spaces.Discrete(6)  # START, EDGE_SEARCH,EXTRACT_INFO, MOVE, BACKWARD, STOP

        # internal state
        self.current_entity_id: Optional[str] = None
        self.visited_entity_ids: Set[str] = set()  # 改为set，不保持顺序
        self.revealed_sources: Set[str] = set()  # 存储source_id
        self.steps_taken: int = 0
        self.agent_answer: str = None

        # external
        self.start_candidates = start_candidates
        self.gt_sources = gt_sources
        self.gt_entities = gt_entities
        self.query = query
        self.answer = answer
        self.model_name = model_name
        
        # Memory
        #self.last_retrieved_source: Optional[Dict] = None
        #self.memory: List[str] = []
        self.history: List[str] = []

    # ---------- Utility ----------
    def _get_id_by_name(self, name: str) -> Optional[str]:
        """Find entity id by its name."""
        for eid, ent in self.kg.id2entity.items():
            if ent.name == name:
                return eid
        return None

    def _get_name_by_id(self, eid: str) -> str:
        ent = self.kg.get_entity(eid)
        return ent.name if ent else eid

    # ---------- Gym API ----------

    def reset(self) -> Dict:
        self.steps_taken = 0
        self.current_entity_id = None
        self.visited_entity_ids = set()  
        self.revealed_sources = set()
        self.current_entity_id = None
       #self.last_retrieved_source = None
        self.history = []

        if not self.start_candidates:
            all_entities = list(self.kg.get_layer(0).graph.nodes())
            if not all_entities:
                raise ValueError("KG layer 0 has no entities to start from.")
            self.current_entity_id = np.random.choice(all_entities)
            self.visited_entity_ids.add(self.current_entity_id)

        return self._get_observation()

    def step(
        self, action: Union[int, Tuple, List,
                            Dict]) -> Tuple[Dict, float, bool, Dict]:
        self.steps_taken += 1
        info = {}
        #done = False
        excuted = False
        reward = self.reward_config["step_penalty"]

        if isinstance(action, dict):
            action_type = action.get("type")
            arg = action.get("arg", None)
        elif isinstance(action, (list, tuple)) and len(action) >= 1:
            action_type = action[0]
            arg = action[1] if len(action) > 1 else None
        else:
            action_type = action
            arg = None

        try:
            act_enum = ActionType(action_type)
        except Exception:
            return self._get_observation(), self.reward_config["illegal_penalty"], True, {"error": "illegal_action_type"}

        if act_enum == ActionType.START:
            excuted, obs = self._handle_start(arg)
            reward = 0.0 if excuted else self.reward_config["illegal_penalty"]

        elif act_enum == ActionType.EDGE_SEARCH:
            excuted, success_retrieve, obs = self._handle_edge_search(arg)
            reward = self.reward_config["step_penalty"] if excuted else self.reward_config["illegal_penalty"]
            if success_retrieve:
                reward += self.reward_config["retrieve_reward"]
            else:
                reward += self.reward_config["illegal_penalty"]

        # elif act_enum == ActionType.EXTRACT_INFO:
        #     excuted, obs = self._handle_extract_info(arg)
        #     reward = self.reward_config["step_penalty"] if excuted else self.reward_config["illegal_penalty"]

        elif act_enum == ActionType.MOVE:
            excuted,success_move, obs = self._handle_move(arg)
            reward = self.reward_config["step_penalty"] if excuted else self.reward_config["illegal_penalty"]
            if success_move:
                reward += self.reward_config["move_reward"]

        elif act_enum == ActionType.BACKWARD:
            excuted, obs = self._handle_backward(arg)
            reward = self.reward_config["step_penalty"] if excuted else self.reward_config["illegal_penalty"]

        elif act_enum == ActionType.STOP:
            excuted, obs = self._handle_stop(arg)
            reward = self.reward_config["step_penalty"] if excuted else self.reward_config["illegal_penalty"]


        if self._check_sources():
            reward += self.reward_config["success_reward"]

        return obs, reward, excuted, info 

    # ---------- Observation ----------
    def _get_observation(self) -> Dict:
        if self.current_entity_id is None:
            return {
                "entity_desc": "",
                "available_relations": [],
                "visited_entities": [],
                "current_entity_name": None,
                "revealed_sources": list(self.revealed_sources),
                "query": self.query,
                "start_candidates": self.start_candidates,
                "action_history": self.history,
                "image": [],
                "text": []
            }

        current_entity = self.kg.get_entity(self.current_entity_id)
        same_layer = self.kg.get_layer(current_entity.level)
        available_relations = []

        for u, v, key, data in same_layer.graph.edges(
                self.current_entity_id,
                keys=True, data=True
        ):
            nbr = v if u == self.current_entity_id else u
            nbr_ent = self.kg.get_entity(nbr)
            attributes = data.get("attributes", {})
            source = attributes.get("chunk_id", "")
            # if source in self.revealed_sources:
            #     continue
            available_relations.append({
                "neighbor": nbr_ent.name,
                "type":data.get("type", "related"),
                "move_type": "horizontal",
                "description":attributes.get("description", ""),
                "source":source
            })

        if self.vert_layer and self.vert_layer.graph.has_node(
                self.current_entity_id):
            for u, v, key, data in self.vert_layer.graph.edges(
                    self.current_entity_id,
                    keys=True, data=True
            ):
                nbr = v if u == self.current_entity_id else u
                nbr_ent =self.kg.get_entity(nbr)
                nbr_level = nbr_ent.level
                move_type= "vertical_down" if nbr_level > current_entity.level else "vertical_up"
                attributes = data.get("attributes", {})
                source = attributes.get("chunk_id", "")
                # if source in self.revealed_sources:
                #     continue
                available_relations.append({
                    "neighbor": nbr_ent.name,
                    "move_type": move_type,
                    "type":data.get("type", "vertical"),
                    "description":attributes.get("description", ""),
                    "source":source
                })
        
        visited_names = [
            self._get_name_by_id(eid) for eid in self.visited_entity_ids
        ]

        # process multimodal data
        images = []
        texts = []

        if self.revealed_sources:
            for source_id in self.revealed_sources:
                if source_id not in self.gt_sources:
                    continue

                if "paragraph" in source_id:
                    text = self.text_map.get(source_id, "")
                    texts.append({"source_id": source_id, "text": text})
                    continue

                image_dir = None
                if "chart" in source_id:
                    image_dir = CHARTMRAG_IMAGE_DIR
                elif "test" in source_id:
                    image_dir = CHARTQA_IMAGE_DIR

                if image_dir is not None:
                    image_path = os.path.join(image_dir, f"{source_id}.png")

                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path).convert("RGB")
                            img_array = np.array(img).astype(np.uint8)
                            images.append({"source_id": source_id, "image": img_array})
                        except Exception as e:
                            print(f"[Warning] Failed to load image for {source_id}: {e}")
                    else:
                        print(f"[Info] Image file not found for source {source_id}: {image_path}")
                    continue

                # 3) 未知类型
                print(f"[Warning] Unknown source type for {source_id}")

        obs = {
            "start_candidates": self.start_candidates,
            "current_entity_name": current_entity.name,
            "entity_desc": current_entity.attributes.get("description", ""),
            "available_relations": available_relations,
            "visited_entities": visited_names,
            "revealed_sources": list(self.revealed_sources),
            "query": self.query,
            "action_history": self.history,
            "text": texts,
            "image": images,
        }

        if self.agent_answer:
            obs["agent_answer"] = self.agent_answer
        return obs


    def _handle_start(self, arg) -> Tuple[bool, Dict]:
        """
        Start action is guaranteed to succeed as long as start_candidates is not empty.

        Priority:
        1) Use arg if it matches a start candidate AND exists in KG
        2) Otherwise fallback to start_candidates[0]
        """

        if not self.start_candidates:
            return False, self._get_observation()

        chosen_eid = None
        chosen_name = None

        # ---------- 1. 尝试使用 agent 提供的 arg ----------
        if isinstance(arg, str):
            arg = arg.strip()
            if arg in self.start_candidates:
                eid = self._get_id_by_name(arg)
                if eid is not None:
                    chosen_eid = eid
                    chosen_name = arg

        # ---------- 2. fallback：使用第一个 start_candidate ----------
        if chosen_eid is None:
            fallback_name = self.start_candidates[0]
            fallback_eid = self._get_id_by_name(fallback_name)

            chosen_eid = fallback_eid
            chosen_name = fallback_name

            if arg is not None:
                print(
                    f"[START FALLBACK] Invalid start arg '{arg}', fallback to '{fallback_name}'"
                )


        self.current_entity_id = chosen_eid
        self.visited_entity_ids = {chosen_eid}

        self.history.append(
            f"STEP={self.steps_taken} | ACTION=start | {chosen_name}"
        )

        return True, self._get_observation()


    #TODO 
    def _handle_edge_search(self, arg) -> Tuple[bool, bool, Dict]:
        obs = self._get_observation()
        avail = obs["available_relations"]
        success_retrieve = False
        if not avail:
            return False, success_retrieve, obs

        chosen_rel = None
        if arg is None:
            chosen_rel = np.random.choice(avail)
        elif isinstance(arg, int):
            chosen_rel = avail[arg % len(avail)]
        elif isinstance(arg, str):
            for rel in avail:
                if rel["neighbor"] == arg:
                    chosen_rel = rel
                    break

        if not chosen_rel:
            return False, success_retrieve, obs

        chunk = chosen_rel.get("source")
        if chunk:
            if not chunk in self.revealed_sources:
                self.revealed_sources.add(chunk)
                # 不能重复添加reward
                if self.gt_sources and (chunk in self.gt_sources):
                    success_retrieve = True
                self.history.append(
                    f"STEP={self.steps_taken} | ACTION=edge_search | "
                    f"{self._get_name_by_id(self.current_entity_id)} -->{chosen_rel['neighbor']}"
                    f"SOURCE={chunk} |"
                )
            else:
                self.history.append(
                    f"STEP={self.steps_taken} | ACTION=edge_search | "
                    f"WARNING: REPEAT SEARCH{self._get_name_by_id(self.current_entity_id)} -->{chosen_rel['neighbor']}"
                    f"SOURCE={chunk} |"
                )
            return True, success_retrieve, self._get_observation()

       
        
    
    # def _handle_extract_info(self, arg)-> Tuple[bool, Dict]:
    #     if not self.last_retrieved_source or not isinstance(arg, str):
    #         return False, self._get_observation()

    #     self.memory.append({self.last_retrieved_source:arg.strip()})
    #     self.last_retrieved_source = None
    #     return True, self._get_observation()

    def _handle_move(self, arg) -> Tuple[bool, Dict]:
        obs = self._get_observation()
        avail = obs["available_relations"]
        success_move = False
        if not avail:
            return False, success_move, obs

        chosen_rel = None
        if arg is None:
            chosen_rel = np.random.choice(avail)
        elif isinstance(arg, int):
            chosen_rel = avail[arg % len(avail)]
        elif isinstance(arg, str):
            # move by target entity name
            for rel in avail:
                if rel["neighbor"] == arg:
                    chosen_rel = rel
                    break

        if not chosen_rel:
            return False,success_move, obs

        tgt_name = chosen_rel["neighbor"]
        tgt_id = self._get_id_by_name(tgt_name)

        if not tgt_id or tgt_id not in self.kg.id2entity:
            return False, success_move, obs

        chunk = chosen_rel.get("source")
        if chunk:
            if not chunk in self.revealed_sources:
                self.revealed_sources.add(chunk)
        # if tgt_name in self.gt_entities:
        #     success_move = True
        self.history.append(
            f"STEP={self.steps_taken} | ACTION=move |FROM {self._get_name_by_id(self.current_entity_id)} TO={tgt_name}"
        )
        success_move = True
        self.visited_entity_ids.add(tgt_id)
        self.current_entity_id = tgt_id
        
        return True, success_move, self._get_observation()

    def _handle_backward(self, arg) -> Tuple[bool, Dict]:
        """回退到之前访问过的节点"""
        if len(self.visited_entity_ids) <= 1:
            return False, self._get_observation()

        visited_without_current = self.visited_entity_ids - {
            self.current_entity_id
        }

        if not visited_without_current:
            return False, self._get_observation()

        target_id = None

        if arg is None:
            return False, self._get_observation()

        elif isinstance(arg, str):
            eid = self._get_id_by_name(arg)
            if eid and eid in visited_without_current:
                target_id = eid

        if target_id and target_id in self.visited_entity_ids:
            self.current_entity_id = target_id
            return True, self._get_observation()

        return False, self._get_observation()

    def _handle_stop(self, arg):
        if isinstance(arg, str):
            self.agent_answer = arg
            obs =self._get_observation()
            return True, obs
        else:
            obs = self._get_observation()
            return False, obs
    
    def _check_max_steps(self):
        return self.steps_taken >= self.max_steps
    
    def _check_sources(self):
        return set(self.gt_sources).issubset(self.revealed_sources)
