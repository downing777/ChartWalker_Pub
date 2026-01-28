from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

import asyncio
import numpy as np
from PIL import Image

from ..gym_image_env import GymImageEnv
from .utils.utils import parse_response
from .utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
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
    action_sep: str = ","
    format_reward: float = 0.5
    image_placeholder: str = "<image>"
    prompt_format: str = "free_think"

    kg_path: str = None
    qa_path: str = None


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
        "extract_info": 2,
        "move": 3,
        "backward": 4,
        "stop": 5,
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
                "gt_sources": row.get("gt_sources", []),
                "start_candidates": row.get("start_candidates", []),
            })

        # underlying gym env (DO NOT MODIFY)
        self.current_task = None
        self.env = None
        self.reward_config = None

        self.total_reward: float = 0.0
        self.valid_actions: List[str] = []

        self.judge_model = LLMRegistry.get("Qwen/Qwen3-14B")
    
    async def reset(self, seed: int):
        rng = np.random.default_rng(seed)

        # 1. sample task
        task = rng.choice(self.task_pool)
        self.current_task = task

        # 2. instantiate world env
        self.env = GymMMKGEnv(
            kg_path=self.config.kg_path,
            query=task["query"],
            answer=task["answer"],
            start_candidates=task["start_candidates"],
            gt_sources=task["gt_sources"],
            max_steps=self.config.max_steps,
        )

        self.reward_config = self.env.reward_config
        self.total_reward = 0.0
        self.valid_actions = []

        # 3. render initial obs
        obs = await self._render_async(init_obs=True)
        return obs, {}
       

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await asyncio.to_thread(self.env.close)

    async def system_prompt(self) -> Dict[str, Any]:
        format_prompt_text = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
        )
        return {
            "obs_str": system_prompt() + "\n" + format_prompt_text
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

        self.valid_actions = []

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

                gym_obs, step_reward, done_flag, gym_info, executed = \
                    await asyncio.to_thread(self.env.step, payload)

                reward = float(step_reward)
                done = bool(done_flag)
                self.valid_actions.append(raw_action)

                metrics["turn_metrics"]["action_is_effective"] = bool(executed)

                if done and await self._judge_success(gym_obs):
                    metrics["traj_metrics"]["success"] = True
                    self.reward += 10.0

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
        try:
            gym_obs = self.env._get_observation()
        except Exception:
            gym_obs = {}

        img_placeholder = self.config.image_placeholder
        multi_modal_input: Optional[Dict[str, List[Image.Image]]] = None

        # ---------- build text ----------
        skip_keys = {"image", "text", "available_relations", "query"}

        obs_text = f"Task / Query: {self.env.query}\n\n"
        obs_text += "\n".join(
            f"{k}: {v}"
            for k, v in gym_obs.items()
            if k not in skip_keys
        )

        # ---------- relations ----------
        rels = gym_obs.get("available_relations", [])
        if rels:
            rel_lines = []
            for i, r in enumerate(rels):
                rel_lines.append(
                    f"[{i}] -({r.get('type')})-> {r.get('neighbor')} "
                    f"(move={r.get('move_type')}, source={r.get('source')})"
                )
            obs_text += "\n\nAvailable Relations:\n" + "\n".join(rel_lines)

        # ---------- text ----------
        texts = gym_obs.get("text", [])
        if texts:
            obs_text += "\n\nLast Observed Text:\n"
            obs_text += "\n".join(
                f"{t['source_id']}: {t['text']}" for t in texts
            )


        multi_modal_input = None
        image_block = ""

        if self.config.render_mode == "vision":
            obs_images = gym_obs.get("image", []) or []

            if len(obs_images) > 0:
                assert len(obs_images) == 1

                img_dict = obs_images[0]
                img_data = img_dict.get("image")
                source_id = img_dict.get("source_id", None)

                pil_images = []
                if isinstance(img_data, np.ndarray):
                    pil_images.append(Image.fromarray(img_data))
                elif isinstance(img_data, Image.Image):
                    pil_images.append(img_data)

                if pil_images:
                    image_block = (
                        "Last Observed Source Image:\n"
                        f"id={source_id}: {img_placeholder}"
                    )
                    multi_modal_input = {img_placeholder: pil_images}

        # ---------- wrap text ----------
        obs_text = obs_text.strip()
        if image_block:
            obs_text = obs_text + "\n\n" + image_block

        if init_obs:
            obs_str = init_observation_template(obs_text)
        else:
            obs_str = action_template(self.valid_actions, obs_text)

        format_prompt_text = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
        )
        obs_str = obs_str + "\n" + format_prompt_text

        # ---------- safety check ----------
        if multi_modal_input is not None:
            n_imgs = len(multi_modal_input[img_placeholder])
            n_tokens = obs_str.count(img_placeholder)
            if n_imgs != n_tokens:
                raise ValueError(
                    f"Placeholder/image mismatch: {n_tokens} tokens vs {n_imgs} images"
                )

        # ---------- return ----------
        obs = {"obs_str": obs_str}
        if multi_modal_input is not None:
            obs["multi_modal_input"] = multi_modal_input

        return obs


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
