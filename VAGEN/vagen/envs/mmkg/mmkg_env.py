from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

import asyncio
import numpy as np
from PIL import Image
import json
from ..gym_image_env import GymImageEnv
from .utils.utils import parse_response
from .utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
    stop_template
)


from src.VLMs import LLMRegistry, LLMMessage
from src.utils.utils import safe_parser

from .mmkg import GymMMKGEnv
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class MMKGConfig:
    render_mode: str = "vision"      # text / vision
    max_steps: int = 10
    max_actions_per_step: int = 1
    action_sep: str = None
    format_reward: float = 0.0
    image_placeholder: str = "<image>"
    prompt_format: str = "free_think"

    kg_path: str = None
    qa_path: str = None
    mode: str = "train"
    eval_cursor :int = 0


# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------

class MMKG(GymImageEnv):
    """
    Async Sokoban-style wrapper for GymMMKGEnv.
    GymMMKGEnv is treated as a pure world-model (state machine).
    """

    ACTION_LOOKUP = {
        "start": 0,
        "edge_search": 1,
        #"extract_info": 2,
        "move": 2,
        "backward": 3,
        "stop": 4,
    }

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        self.config = MMKGConfig(**env_config)
        df = pd.read_csv(self.config.qa_path)
        self.task_pool = []
        for _, row in df.iterrows():
            self.task_pool.append({
                "query": row["query"],
                "answer": row["answer"],
                "gt_sources": json.loads(row["gt_sources"]),
                "gt_entities": json.loads(row["gt_entities"]),
                "start_candidates": json.loads(row["start_candidates"]),
            })

        # underlying gym env (DO NOT MODIFY)
        
        self.current_task = None
        self.env = None
        self.eval_cursor = getattr(self.config, "eval_cursor", 0)

        self.total_reward: float = 0.0
        self.stop_phase: bool = False
        self.judge_model = LLMRegistry.get("Qwen/Qwen3-14B")
    
    async def reset(self, seed: int):
        rng = np.random.default_rng(seed)

        # 1. sample task
        if self.config.mode == "eval":
            task = self.task_pool[self.eval_cursor]
            self.eval_cursor = (self.eval_cursor + 1) % len(self.task_pool)
        else:
            task = rng.choice(self.task_pool)

        self.current_task = task
        self.stop_phase = False

        # 2. instantiate world env
        self.env = GymMMKGEnv(
            kg_path=self.config.kg_path,
            query=task["query"],
            answer=task["answer"],
            start_candidates=task["start_candidates"],
            gt_sources=task["gt_sources"],
            gt_entities=task["gt_entities"],
            max_steps=self.config.max_steps,
        )

        self.reward_config = self.env.reward_config
        self.total_reward = 0.0

        # 3. render initial obs
        obs = await self._render_async(init_obs=True)
        return obs, {}
       

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await asyncio.to_thread(self.env.close)

    async def system_prompt(self) -> Dict[str, Any]:
        return {
            "obs_str": system_prompt() + "\n" 
        }


    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:

        parsed = parse_response(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
            prompt_format=self.config.prompt_format,
        )

        action_list = parsed.get("actions", [])

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)


        metrics = {
            "turn_metrics": {
                "action_is_valid": bool(action_list),
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if action_list:
            raw_action = action_list[0]
            parts = raw_action.split(None, 1)
            action = parts[0]
            arg = parts[1].strip() if len(parts) > 1 else None

            if action in self.ACTION_LOOKUP:
                action_type = self.ACTION_LOOKUP[action]

                if arg is None:
                    payload = action_type
                else:
                    try:
                        payload = (action_type, int(arg))
                    except Exception:
                        payload = {"type": action_type, "arg": arg}

                gym_obs, step_reward, executed, gym_info,  = \
                    await asyncio.to_thread(self.env.step, payload)

                reward = float(step_reward)
                metrics["turn_metrics"]["action_is_effective"] = bool(executed)

                if self.stop_phase:
                    done = True
                    answer_success = await self._judge_success(gym_obs)
                    if answer_success:
                        reward += self.reward_config.get("answer_reward", 5.0)
                    info["agent_answer"] = gym_obs.get("agent_answer", "")
                    info["answer_success"] = answer_success

                if self.env._check_sources():
                    metrics["traj_metrics"]["success"] = True
                    self.stop_phase = True
              
                if self.env._check_max_steps():
                    self.stop_phase = True

                if isinstance(gym_info, dict):
                    info.update(gym_info)
            else:
                metrics["turn_metrics"]["action_is_valid"] = False
                reward = self.reward_config.get("illegal_penalty", -1.0)
        else:
            metrics["turn_metrics"]["action_is_valid"] = False
            reward = self.reward_config.get("illegal_penalty", -1.0)

        
        if metrics["turn_metrics"]["action_is_valid"] and parsed.get("format_correct", False):
            reward += getattr(self.config, "format_reward", 0.0)
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += reward

        obs = await self._render_async(init_obs=False)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Rendering (Sokoban-aligned)
    # ------------------------------------------------------------------

    async def _render_async(self, init_obs: bool) -> Dict[str, Any]:
        def format_candidates(cands):
            if not cands:
                return "[]"
            if len(cands) == 1:
                return f"[{cands[0]},]"
            return "[" + ", ".join(cands) + "]"
        try:
            gym_obs = self.env._get_observation()
        except Exception:
            gym_obs = {}

        img_placeholder = self.config.image_placeholder
        #multi_modal_input: Optional[Dict[str, List[Image.Image]]] = None


        # ---------- build text ----------
        skip_keys = {"image", "text", "available_relations", "query","start_candidates", "action_history"}

        obs_text = f"[Task / Query]: {self.env.query}\n\n"
        obs_text += "\n".join(
            f"{k}: {v}"
            for k, v in gym_obs.items()
            if k not in skip_keys
        )


        # ---------- text ----------
        texts = gym_obs.get("text", [])
        text_block = None
        if texts:
            text_block= "\n\n[Useful Observed Text]:\n"
            text_block += "\n".join(
                f"{t['source_id']}: {t['text']}" for t in texts
            )
            obs_text += text_block
       

        if self.config.render_mode == 'vision':
            img_placeholder = str(getattr(self.config, "image_placeholder", "<image>"))
            pil_images = []

            try:
                obs_images = gym_obs.get("image", []) or []
                image_block = "Useful Observed Source Image: \n"
                
                if len(obs_images) > 0:
                    for img_dict in obs_images:
                    
                        img_data = img_dict.get("image")
                        source_id = img_dict.get("source_id", None)

                        if isinstance(img_data, np.ndarray):
                            pil_images.append(Image.fromarray(img_data))
                        elif isinstance(img_data, Image.Image):
                            pil_images.append(img_data)

                        image_block += f"id={source_id}: {img_placeholder}"
                    
                else:
                    empty_image = Image.new("RGB", (10, 10), color=(0, 0, 0))
                    pil_images = [empty_image]
                    image_block = f"No Useful Source Images:\n {img_placeholder}"

            except Exception as e:
                print(f"[Render Warning] Failed to load multimodal images: {e}", flush=True)
                empty_image = Image.new("RGB", (10, 10), color=(0, 0, 0))
                pil_images = [empty_image]
                image_block = f"No Useful Source Images:\n {img_placeholder}"
            
            obs_text = obs_text.strip() + "\n" + image_block
        
        # ---------- relations ----------
        rels = gym_obs.get("available_relations", [])
        revealed_sources = set(self.env.revealed_sources)

        if rels:
            searchable_lines = []
            forbidden_lines = []

            for i, r in enumerate(rels):
                source = r.get("source")
                line = (
                    f"[{i}] {self.env._get_name_by_id(self.env.current_entity_id)}-({r.get('type')})-> {r.get('neighbor')}; "
                    f"src={source}; "
                    f"desc={r.get('description')}"
                )

                if source in revealed_sources:
                    forbidden_lines.append(line + "  [FORBIDDEN: source already revealed]")
                else:
                    searchable_lines.append(line)

            if searchable_lines:
                obs_text += "\n[Searchable Relations]:\n"
                obs_text += "\n".join(searchable_lines)

            if forbidden_lines:
                obs_text += "\n\n[Forbidden Relations — DO NOT SEARCH]:\n"
                obs_text += "\n".join(forbidden_lines)

        # ---------- actions ----------
        obs_text += "\n[Action History]:\n"
        obs_text += "\n".join(gym_obs.get("action_history")) + "\n"

        multi_modal_data = {img_placeholder: pil_images}
        format_prompt_text = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
        )
        
        if init_obs:
            init_obs_text = f"Task / Query: {self.env.query}\n\n"
            init_obs_text += f"\nNO Image Observed:\n {img_placeholder}"
            obs_str = init_observation_template(observation=init_obs_text,start_candidates=format_candidates(self.env.start_candidates)) + "\n" + format_prompt_text
        
        elif self.stop_phase:
            # 查全
            stop_obs = f"Task / Query: {self.env.query}\n\n"
            stop_obs += image_block
            if text_block:
                stop_obs += "\n" + text_block
            obs_str = stop_template(observation=stop_obs) + "\n" + format_prompt_text

        else:
            obs_str = action_template(
                observation=obs_text,
            ) + "\n" + format_prompt_text

        n_imgs = len(multi_modal_data.get(img_placeholder, []))
        n_tokens = obs_str.count(img_placeholder)
        if n_tokens != n_imgs:
            raise ValueError(
                f"Placeholder/image mismatch: {n_tokens} '{img_placeholder}' tokens in obs_str, "
                f"but {n_imgs} images provided."
            )
        return {"obs_str": obs_str, "multi_modal_input": multi_modal_data}


    # ------------------------------------------------------------------
    # Success Judge (unchanged, wrapper-level)
    # ------------------------------------------------------------------

    async def _judge_success(self, obs: Dict[str, Any]) -> bool:
        agent_answer = obs.get("agent_answer", "")
        gt_answer = self.env.answer
        query = self.env.query

        prompt = f"""
        You are an answer judge.

        QUESTION:
        {query}

        AGENT_ANSWER:
        {agent_answer}

        REFERENCE_ANSWER:
        {gt_answer}

        Return ONLY valid JSON: {{"success": true/false}}
        """

        for _ in range(3):
            try:
                msg = [LLMMessage(role="user", content=prompt)]
                resp = await asyncio.to_thread(
                    self.judge_model.generate_response,
                    messages=msg
                )
                parsed = safe_parser(resp)
                return bool(parsed.get("success", False))
            except Exception:
                continue
        return False
