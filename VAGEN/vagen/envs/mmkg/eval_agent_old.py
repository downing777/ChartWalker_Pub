# eval_mmkg_agent.py
import os
import json
import time
import argparse
import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# 1) 这里替换成你工程里的真实 import 路径
#    假设 MMKG 类在 vagen/env/mmkg_env.py 或类似位置
from VAGEN.vagen.envs.mmkg.mmkg_env import MMKG  # <-- 改这里

# 2) 这里替换成你给的 API 封装 import
from src.VLMs.resources_vlm import get_from_ks_openai  # <-- 改这里


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_pil_images(
    pil_images: List[Image.Image],
    out_dir: str,
    prefix: str,
) -> List[str]:
    """
    将 PIL 图片保存为 png，返回 image_paths 列表供 get_from_ks_openai 使用
    """
    ensure_dir(out_dir)
    paths = []
    for i, im in enumerate(pil_images):
        if im is None:
            continue
        fp = os.path.join(out_dir, f"{prefix}_{i}.png")
        # 避免 mode/透明度导致保存异常
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(fp, format="PNG")
        paths.append(fp)
    return paths


def extract_images_from_obs(
        obs: Dict[str, Any]) -> Tuple[List[Image.Image], str]:
    """
    从 env 返回的 obs 结构中取出 PIL images。
    你的 obs: {"obs_str": ..., "multi_modal_input": {placeholder: [PIL,...]}}
    返回 (images, placeholder_key)
    """
    mm = obs.get("multi_modal_input") or {}
    if not mm:
        return [], "<image>"
    # 一般只有一个 key：placeholder
    placeholder = next(iter(mm.keys()))
    images = mm.get(placeholder) or []
    return images, placeholder

def compute_recall(revealed_sources, gt_sources) -> Tuple[float, int, int]:
    """
    recall = |revealed ∩ gt| / |gt|
    返回: (recall, hit_count, gt_count)
    """
    gt = set(gt_sources or [])
    rev = set(revealed_sources or [])
    gt_count = len(gt)
    hit = len(gt & rev)
    recall = (hit / gt_count) if gt_count > 0 else 0.0
    return recall, hit, gt_count


@dataclass
class EpisodeStats:
    episode: int
    seed: int
    success: bool
    total_reward: float
    steps: int
    action_valid_rate: float
    action_effective_rate: float
    avg_step_reward: float
    wall_time_sec: float
    model: str

    query: str
    answer: str
    gt_sources_count: int
    revealed_sources_count: int
    hit_sources_count: int
    final_recall: float


# -------------------------
# Core rollout
# -------------------------
async def run_one_episode(
    env: MMKG,
    episode_idx: int,
    seed: int,
    model: str,
    api_key: str,
    base_url: str,
    thinking: Optional[bool],
    temperature: float,
    timeout: int,
    stream: bool,
    max_steps: int,
    image_dump_dir: str,
    save_traj_path: Optional[str] = None,
) -> EpisodeStats:
    t0 = time.time()

    # system prompt（一次即可）
    sys = await env.system_prompt()
    sys_prompt_text = sys["obs_str"]

    obs, _ = await env.reset(seed=seed)

    # ---- task info from underlying env ----
    task_query = getattr(env.env, "query", "")
    task_answer = getattr(env.env, "answer", "")
    gt_sources = list(getattr(env.env, "gt_sources", []) or [])

    total_reward = 0.0
    steps = 0

    n_valid = 0
    n_effective = 0

    traj_records = []

    done = False
    while (not done) and steps < max_steps:
        obs_str = obs["obs_str"]
        pil_images, placeholder = extract_images_from_obs(obs)

        # dump images -> paths
        step_prefix = f"ep{episode_idx:04d}_s{seed}_t{steps:03d}"
        step_img_dir = os.path.join(image_dump_dir, f"ep{episode_idx:04d}", f"t{steps:03d}")
        img_paths = save_pil_images(pil_images, step_img_dir, prefix=step_prefix) if pil_images else None

        # call model (sync) in thread to avoid blocking the event loop
        def _call():
            return get_from_ks_openai(
                prompt=obs_str,
                model=model,
                system_prompt=sys_prompt_text,
                api_key=api_key,
                base_url=base_url,
                image_paths=img_paths,
                thinking=thinking,
                temperature=temperature,
                max_retries=3,
                timeout=timeout,
                stream=stream,
            )

        raw_resp = await asyncio.to_thread(_call)

        # env.step 期望传 action_str（包含 <answer>...）
        obs, reward, done, info = await env.step(raw_resp)

        total_reward += float(reward)
        steps += 1

        # metrics
        m = (info or {}).get("metrics", {})
        turn_m = (m.get("turn_metrics") or {})
        is_valid = bool(turn_m.get("action_is_valid", False))
        is_effective = bool(turn_m.get("action_is_effective", False))
        n_valid += int(is_valid)
        n_effective += int(is_effective)

        # 记录轨迹，便于复盘
        traj_records.append(
            {
                "episode": episode_idx,
                "seed": seed,
                "step": steps,
                "obs_str": obs_str,
                "img_paths": img_paths,
                "model_response": raw_resp,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        if done:
            break

    # episode success
    # 你在 step 里写了 info["success"] = metrics["traj_metrics"]["success"]
    success = bool((info or {}).get("success", False))
    final_revealed = list(getattr(env.env, "revealed_sources", set()) or set())
    final_recall, final_hit, final_gt_cnt = compute_recall(final_revealed, gt_sources)

    wall = time.time() - t0
    action_valid_rate = (n_valid / steps) if steps else 0.0
    action_effective_rate = (n_effective / steps) if steps else 0.0
    avg_step_reward = (total_reward / steps) if steps else 0.0

    stats = EpisodeStats(
        episode=episode_idx,
        seed=seed,
        success=success,
        total_reward=float(total_reward),
        steps=int(steps),
        action_valid_rate=float(action_valid_rate),
        action_effective_rate=float(action_effective_rate),
        avg_step_reward=float(avg_step_reward),
        wall_time_sec=float(wall),
        model=model,

        query=task_query,
        answer=task_answer,
        gt_sources_count=int(final_gt_cnt),
        revealed_sources_count=int(len(set(final_revealed))),
        hit_sources_count=int(final_hit),
        final_recall=float(final_recall),
    )

    # 保存轨迹（jsonl）
    if save_traj_path:
        ensure_dir(os.path.dirname(save_traj_path))
        with open(save_traj_path, "a", encoding="utf-8") as f:
            for r in traj_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


async def main_async(args):
    # env config：你 MMKGConfig 需要这些
    env_config = dict(            # vision / text
        max_steps=args.env_max_steps,              # underlying GymMMKGEnv max_steps
        max_actions_per_step=1,
        action_sep=None,
        format_reward=0.0,
        image_placeholder="<image>",
        prompt_format="free_think",
        kg_path=args.kg_path,
        qa_path=args.qa_path,
    )

    env = MMKG(env_config)

    ensure_dir(args.out_dir)
    image_dump_dir = os.path.join(args.out_dir, "mm_images")
    ensure_dir(image_dump_dir)

    traj_path = os.path.join(args.out_dir, "traj.jsonl") if args.save_traj else None

    all_stats: List[EpisodeStats] = []

    # 固定 episode seeds，保证可复现
    base_seed = args.seed
    for ep in range(args.episodes):
        ep_seed = base_seed + ep
        st = await run_one_episode(
            env=env,
            episode_idx=ep,
            seed=ep_seed,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            thinking=args.thinking,
            temperature=args.temperature,
            timeout=args.timeout,
            stream=args.stream,
            max_steps=args.rollout_max_steps,   # wrapper 测试的最大步数
            image_dump_dir=image_dump_dir,
            save_traj_path=traj_path,
        )
        all_stats.append(st)
        print(json.dumps(asdict(st), ensure_ascii=False))

    await env.close()

    # 汇总
    n = len(all_stats)
    succ = sum(int(s.success) for s in all_stats)
    acc = succ / n if n else 0.0
    avg_reward = sum(s.total_reward for s in all_stats) / n if n else 0.0
    avg_steps = sum(s.steps for s in all_stats) / n if n else 0.0
    avg_valid = sum(s.action_valid_rate for s in all_stats) / n if n else 0.0
    avg_eff = sum(s.action_effective_rate for s in all_stats) / n if n else 0.0
    avg_wall = sum(s.wall_time_sec for s in all_stats) / n if n else 0.0

    avg_final_recall = sum(s.final_recall for s in all_stats) / n if n else 0.0
    avg_hit = sum(s.hit_sources_count for s in all_stats) / n if n else 0.0
    avg_gt = sum(s.gt_sources_count for s in all_stats) / n if n else 0.0

    summary = {
        "episodes": n,
        "success": succ,
        "accuracy": acc,
        "avg_total_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_action_valid_rate": avg_valid,
        "avg_action_effective_rate": avg_eff,
        "avg_wall_time_sec": avg_wall,
        "avg_final_recall": avg_final_recall,
        "avg_hit_sources_count": avg_hit,
        "avg_gt_sources_count": avg_gt,
        "model": args.model,
        "env_max_steps": args.env_max_steps,
        "rollout_max_steps": args.rollout_max_steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kg_path", type=str, default = "/share/project/tangning/MMGraph/working_dirs/chartrag/graph_chartmrag_qwen3-vl-8b_conti_charttext_rmduprel_1215.pkl")
    p.add_argument("--qa_path", type=str, default="/share/project/tangning/MMGraph_Lite/VAGEN/examples/mmkg/datasets/chenhan_1_11/easy_cleaned.csv")
    p.add_argument("--out_dir", type=str, default="./mmkg_eval_out")

    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)

    # env
    p.add_argument("--env_max_steps", type=int, default=15, help="传给 GymMMKGEnv 的 max_steps")
    p.add_argument("--rollout_max_steps", type=int, default=15, help="评测时最多 rollout 步数（wrapper 层）")

    # model call
    p.add_argument("--model", type=str, default="qwen3-vl-235b-a22b-thinking")
    p.add_argument("--api_key", type=str, default = "9f1fd846-feb3-4216-996e-9e7a3dab7820")
    p.add_argument("--base_url", type=str, default="https://kspmas.ksyun.com/v1/")
    p.add_argument("--thinking", type=lambda x: None if x == "None" else (x.lower() == "true"), default="True")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--timeout", type=int, default=200)
    p.add_argument("--stream",default=False)

    p.add_argument("--save_traj", action="store_true", help="保存每步 obs/resp/info 到 jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
