#!/usr/bin/env python3

from src.Dataset.ChartMRag import ChartMRag
from src.MMRAG import MMRag
from src.KG import HierarchicalMMKG, PageRankWalker
import logging
import os
import argparse
import random
import numpy as np
import torch
import json
from collections import Counter

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pkl_path", type=str, required=True,)
    parser.add_argument("--output_dir", type=str, default="./qa_generated",)
    parser.add_argument("--n_qas", type=int, default=10,)
    parser.add_argument("--seed_level", type=int, default=2,)
    parser.add_argument("--branches", type=int, default=4,)
    parser.add_argument("--branch_max_hops", type=int, default=4,)
    parser.add_argument("--max_evidence", type=int, default=5,)
    parser.add_argument("--pool_size", type=int, default=120,)
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--emb_model", type=str, 
                       default="Qwen/Qwen3-Embedding-8B",)
    parser.add_argument("--device", type=str, default="cuda:1",)
    
    args = parser.parse_args()

    seed_everything(args.seed)
    

    config = {"qa_dataset": "chart-mrag"}
    dataset = ChartMRag(config)


    mmrag_args = argparse.Namespace(
        emb_model=args.emb_model,
        device=args.device,
        kg_vlm="qwen3-vl-235b-a22b-thinking",  
        save_path=os.path.splitext(args.pkl_path)[0],  
    )
    mmrag = MMRag(args=mmrag_args)
    

    mmrag.kg = HierarchicalMMKG.load(args.pkl_path, args=mmrag_args)


    try:
        emb_path = args.pkl_path.replace(".pkl", "_emb.pkl")
        if os.path.exists(emb_path):
            import pickle
            with open(emb_path, "rb") as f:
                emb_data = pickle.load(f)
                mmrag.entity2emb = emb_data.get("entity2emb", {})
                mmrag.relation2emb = emb_data.get("relation2emb", {})
            print(f"Embedding loaded: {len(mmrag.entity2emb)} ")
        else:
            mmrag.entity2emb = {}
            mmrag.relation2emb = {}
    except Exception as e:
        print(f"Warning: {e}")
        mmrag.entity2emb = {}
        mmrag.relation2emb = {}
    

    walker = PageRankWalker(
        mmkg=mmrag.kg,
        chart_map=dataset.chart_map,
        text_map=dataset.merged_text_map,
        entity_emb=mmrag.entity2emb,
        relation_emb=mmrag.relation2emb,
        sem_edge_gamma=0.6,  
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    sampling_kwargs = dict(
        seed_level=args.seed_level,
        branches=args.branches,
        branch_max_hops=args.branch_max_hops,
        retries=40,
        max_branch_attempts=40,
        min_sources=2,
        require_charts=2,
        branch_vec_mode="end",
    )
    
    gen_kwargs = dict(
        pool_size=args.pool_size,
        max_reuse_per_path=2,
        resample_prob=0.30,
        dedup_pool=True,
        max_evidence=args.max_evidence,
        temperature=0.7,
    )
    
    try:
        dataset_out = walker.gen_QA_dataset(
            n_qas=args.n_qas,
            sampling_kwargs=sampling_kwargs,
            **gen_kwargs
        )
        
        output_file = os.path.join(args.output_dir, "qa_generated.jsonl")
        save_jsonl(dataset_out, output_file)

        if dataset_out:
            query_types = Counter([qa.get("query_type", "Unknown") for qa in dataset_out])
            print(f"\nQuery Type: {dict(query_types)}")
            
            avg_difficulty = sum([qa.get("difficulty", {}).get("total", 0) for qa in dataset_out]) / len(dataset_out)
            print(f"Difficulty: {avg_difficulty:.2f}")
        
    except Exception as e:
        print(f"Error during QA generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

