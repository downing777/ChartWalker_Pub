from src.Dataset.ChartMRag import ChartMRag

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sentence_transformers.util import cos_sim

from src.MMRAG import MMRag

import logging
import asyncio
import os
import re

import argparse
from src.KG.MMKG import HierarchicalMMKG
from sentence_transformers import SentenceTransformer

from src.utils import setup_logger
import time
from typing import Dict, List
import pickle
import json

setup_logger("lightrag", level="DEBUG")
logger = logging.getLogger("lightrag")

logger.info("Logging system initialized!")

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="argsuration for the hierarchical KG")

parser.add_argument("--hierarchy",
                    action="store_false",  
                    default=True, 
                    )

parser.add_argument("--dataset",default="chartmrag", type=str)


parser.add_argument("--emb_model",default="Qwen/Qwen3-Embedding-8B", type=str)

parser.add_argument("--vlm",default="qwen3-vl-8b",type=str)

parser.add_argument("--device",default="cuda:0", type=str)

parser.add_argument("--top_k",default=5, type=int)

parser.add_argument("--sample_size", default=None, help = "sample size for the entry to process" , type=int)




args = parser.parse_args()
args.save_path = f"graph_{args.dataset}_{args.vlm}"
args.save_path = os.path.join(script_dir, args.save_path)
args.kg_vlm = args.vlm  # Set kg_vlm for KG extraction

def load_corpus_dict(index_path):
    corpus = {}
    with open(index_path, "rb") as f:
        cache = pickle.load(f)
    for id, emb in zip(cache["index_ids"], cache["index_embs"]):
        corpus[id] = emb

    return corpus


if __name__ == "__main__":
    config = {"qa_dataset": "Dataset/chart-mrag"}
    dataset = ChartMRag(config)
    #loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    mmrag = MMRag(args = args)
    asyncio.run(mmrag.insert(dataset = dataset, stage="raw"))
    # mmrag.kg = HierarchicalMMKG.load(f"{mmrag.args.save_path}.pkl")



   