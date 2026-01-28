# eval_mmkg_agent.py
import os
import json
import time
import argparse
import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# === Project imports (按你工程实际路径) ===
from VAGEN.vagen.envs.mmkg.mmkg_env import MMKG
from src.VLMs.resources_vlm import get_from_ks_openai


import random
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
    ensure_dir(out_dir)
    paths = []
    for i, im in enumerate(pil_images):
        if im is None:
            continue
        fp = os.path.join(out_dir, f"{prefix}_{i}.png")
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(fp, format="PNG")
        paths.append(fp)
    return paths


def extract_images_from_obs(
    obs: Dict[str, Any]
) -> Tuple[List[Image.Image], str]:
    mm = obs.get("multi_modal_input") or {}
    if not mm:
        return [], "<image>"
    placeholder = next(iter(mm.keys()))
    images = mm.get(placeholder) or []
    return images, placeholder


def compute_recall(revealed_sources, gt_sources) -> Tuple[float, int, int]:
    gt = set(gt_sources or [])
    rev = set(revealed_sources or [])
    hit = len(gt & rev)
    return (hit / len(gt)) if gt else 0.0, hit, len(gt)

def _is_rate_limit_err(e: Exception) -> bool:
    s = repr(e)
    # 兼容 openai.RateLimitError / 429 / token rate limit exceeded 等
    return ("RateLimitError" in s) or ("429" in s) or ("Too Many Requests" in s) or ("token rate limit" in s)

async def call_llm_with_backoff(
    call_fn,
    max_attempts: int = 8,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
):
    """
    call_fn: 一个无参函数，内部执行真实请求（同步函数也行）
    """
    for attempt in range(1, max_attempts + 1):
        try:
            # 兼容同步 call
            return await asyncio.to_thread(call_fn)
        except Exception as e:
            if not _is_rate_limit_err(e):
                raise
            # 指数退避 + jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (0.75 + 0.5 * random.random())
            print(f"[WARN] Rate limited (attempt {attempt}/{max_attempts}). Sleep {delay:.2f}s then retry.")
            await asyncio.sleep(delay)
    raise RuntimeError("All retries failed due to rate limit.")
# -------------------------
# Episode Stats
# -------------------------
@dataclass
class EpisodeStats:
    episode: int
    seed: int
    retrieve_success: bool
    answer_success: bool
    total_reward: float
    steps: int
    action_valid_rate: float
    action_effective_rate: float
    avg_step_reward: float
    wall_time_sec: float
    model: str

    query: str
    answer: str
    agent_answer:str
    gt_sources_count: int
    revealed_sources_count: int
    gt_sources: List[str] 
    revealed_sources: List[str]
    hit_sources_count: int
    final_recall: float


# -------------------------
# Rollout
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
    sys_prompt = (await env.system_prompt())["obs_str"]
    obs, _ = await env.reset(seed=seed)

    task_query = getattr(env.env, "query", "")
    task_answer = getattr(env.env, "answer", "")
    gt_sources = getattr(env.env, "gt_sources", []) or []

    total_reward = 0.0
    steps = 0
    n_valid = 0
    n_effective = 0
    done = False

    traj_records = []
    last_info ={}
    while not done and (steps < max_steps):
        obs_str = obs["obs_str"]
        pil_images, _ = extract_images_from_obs(obs)

        step_dir = os.path.join(
            image_dump_dir, f"ep{episode_idx:04d}", f"t{steps:03d}"
        )
        img_paths = (
            save_pil_images(pil_images, step_dir, f"s{seed}_t{steps}")
            if pil_images else None
        )

        def _call():
            return get_from_ks_openai(
                prompt=obs_str,
                system_prompt=sys_prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                image_paths=img_paths,
                thinking=thinking,
                temperature=temperature,
                timeout=timeout,
                max_retries=3,
                stream=stream,
            )

        try:
            # 用外层 backoff 吸收 429
            raw_resp = await call_llm_with_backoff(_call, max_attempts=8, base_delay=2.0)
        except Exception as e:
            print(f"[ERROR] API call failed (final): {e}")
            # 不中断整个进程：结束当前 episode，并产出 stats
            done = True
            break

        obs, reward, done, info = await env.step(raw_resp)
        last_info = info 

        total_reward += float(reward)
        steps += 1

        m = (info or {}).get("metrics", {})
        tm = m.get("turn_metrics", {})
        n_valid += int(tm.get("action_is_valid", False))
        n_effective += int(tm.get("action_is_effective", False))

        traj_records.append({
            "episode": episode_idx,
            "step": steps,
            "obs": obs_str,
            "response": raw_resp,
            "reward": reward,
            "done": done,
        })

    retrieve_success = bool((last_info or {}).get("success", False))
    answer_success = bool((last_info or {}).get("answer_success", False))
    agent_answer = last_info.get("agent_answer", "")
    revealed = list(getattr(env.env, "revealed_sources", []) or [])
    recall, hit, gt_cnt = compute_recall(revealed, gt_sources)

    wall = time.time() - t0

    stats = EpisodeStats(
        episode=episode_idx,
        seed=seed,
        retrieve_success=retrieve_success,
        answer_success=answer_success,
        total_reward=total_reward,
        steps=steps,
        action_valid_rate=n_valid / steps if steps else 0.0,
        action_effective_rate=n_effective / steps if steps else 0.0,
        avg_step_reward=total_reward / steps if steps else 0.0,
        wall_time_sec=wall,
        model=model,
        query=task_query,
        answer=task_answer,
        agent_answer = agent_answer,
        gt_sources_count=gt_cnt,
        revealed_sources_count=len(set(revealed)),
        gt_sources = gt_sources,
        revealed_sources = revealed,
        hit_sources_count=hit,
        final_recall=recall,
    )

    if save_traj_path:
        ensure_dir(os.path.dirname(save_traj_path))
        with open(save_traj_path, "a", encoding="utf-8") as f:
            for r in traj_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


# -------------------------
# Main
# -------------------------
async def main_async(args):

    env = MMKG(dict(
        kg_path=args.kg_path,
        qa_path=args.qa_path,
        max_steps=args.env_max_steps,
        image_placeholder="<image>",
        prompt_format="free_think",
        max_actions_per_step=1,
        mode = "eval",
        eval_cursor = 30
    ))

    ensure_dir(args.out_dir)
    image_dump_dir = os.path.join(args.out_dir, "mm_images")
    ensure_dir(image_dump_dir)

    episode_jsonl = os.path.join(args.out_dir, "episodes.jsonl")
    traj_path = os.path.join(args.out_dir, "traj.jsonl") if args.save_traj else None

    all_stats = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\n===== Episode {ep} (seed={seed}) =====")

        try:
            st = await run_one_episode(
                env, ep, seed,
                args.model, args.api_key, args.base_url,
                args.thinking, args.temperature,
                args.timeout, args.stream,
                args.rollout_max_steps,
                image_dump_dir,
                traj_path,
            )
        except Exception as e:
            print(f"[FATAL] Episode {ep} crashed: {e}")
            st = EpisodeStats(
                episode=ep,
                seed=seed,
                retrieve_success=False,
                answer_success=False,
                total_reward=0.0,
                steps=0,
                action_valid_rate=0.0,
                action_effective_rate=0.0,
                avg_step_reward=0.0,
                wall_time_sec=0.0,
                model=args.model,
                query="",
                answer="",
                agent_answer="",
                gt_sources_count=0,
                revealed_sources_count=0,
                gt_sources=[],
                revealed_sources=[],
                hit_sources_count=0,
                final_recall=0.0,
            )

        with open(episode_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(st), ensure_ascii=False) + "\n")

        all_stats.append(st)
        print(json.dumps(asdict(st), ensure_ascii=False))

        if (ep + 1) % args.sleep_every == 0 and ep + 1 < args.episodes:
            print(f"[INFO] Sleeping {args.sleep_sec}s...")
            await asyncio.sleep(args.sleep_sec)

    await env.close()

    n = len(all_stats)

    retrieve_succ = sum(int(s.retrieve_success) for s in all_stats)
    retrieve_acc = retrieve_succ / n if n else 0.0

    answer_succ = sum(int(s.answer_success) for s in all_stats)
    answer_acc = answer_succ / n if n else 0.0
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
        "retrieve_success": retrieve_succ,
        "retrieve_accuracy": retrieve_acc,
        "answer_success": answer_succ,
        "answer_accuracy": answer_acc,
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



# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kg_path", type=str, default = "/share/project/tangning/MMGraph/working_dirs/chartrag/graph_chartmrag_qwen3-vl-8b_conti_charttext_rmduprel_1215.pkl")
    p.add_argument("--qa_path", type=str, default="/share/project/tangning/MMGraph_Lite/VAGEN/examples/mmkg/datasets/chartrag_mix_1_15/mixed_clean_test.csv")
    p.add_argument("--out_dir", type=str, default="./mmkg_eval_out_conti")

    p.add_argument("--episodes", type=int, default=28)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--env_max_steps", type=int, default=15)
    p.add_argument("--rollout_max_steps", type=int, default=20)

    p.add_argument("--model", type=str, default="qwen3-vl-235b-a22b-thinking")
    p.add_argument("--api_key", type=str, default = "9f1fd846-feb3-4216-996e-9e7a3dab7820")
    p.add_argument("--base_url", type=str, default="https://kspmas.ksyun.com/v1/")
    p.add_argument("--thinking", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--timeout", type=int, default=200)
    p.add_argument("--stream", action="store_true")

    p.add_argument("--sleep_every", type=int, default=1)
    p.add_argument("--sleep_sec", type=float, default=10.0)
    p.add_argument("--save_traj", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
