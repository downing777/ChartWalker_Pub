## Implementation of the MMRAG,
# TODO complete the knowledge graph construction
import asyncio
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Optional,
    List,
    Dict,
    Literal
)

from src.utils.utils import async_task_runner, Tokenizer

from src.KG.MMKG import HierarchicalMMKG
from src.KG.entity_resolver import EntityResolver


from .operates import multimodal_chunking_func, extract_entities, extract_chart_entities
from src.schema import MMChunk
from torch.utils.data import Dataset, DataLoader


from argparse import Namespace
from sentence_transformers import SentenceTransformer
import numpy as np

from src.Dataset.ChartMRag import ChartMRag
import time
from src.VLMs import QwenChat, LLMRegistry, LLM
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
import logging

logger = logging.getLogger("lightrag")
@dataclass
class MMRag:
    chunking_func: Callable[
        [
            Tokenizer,
            str,
        ],
        List[MMChunk],
    ] = field(default_factory=lambda: multimodal_chunking_func)
    #= field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:
        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `file path`: The file to be split into chunks.
    Defaults to `chunking_by_token_size` if not specified.
    """

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.
    """

    max_concurrency: int = field(default=2)
    args: Namespace = field(default=None)
    emb_model: SentenceTransformer = field(default=None)
    
    def __post_init__(self,):
        self.base_graph_lock = asyncio.Lock()
        
        self.entity2emb: Dict[str, np.ndarray] = {}
        self.relation2emb: Dict[str, np.ndarray] = {}
        self.kg = HierarchicalMMKG(
            args=self.args,
        )
    
    def load_emb_model(self):
        if self.emb_model is None:
            logger.info(f"Loading embedding model:")
            self.emb_model = SentenceTransformer(
                self.args.emb_model,
                device=self.args.device,
                tokenizer_kwargs={"padding_side": "left"},
            )
        
    
    async def process_dataset(self,dataset: Dataset):
        """
        Args:
            folder_path (str): The path to the folder containing the files that need to be extracted for the knowledge graph.
            file_paths (Optional[List[str]], optional): Defaults to None. You can choose to input a list of file paths to process.

        """
        #pipeline_status_lock = asyncio.Lock()
        vlm = LLMRegistry.get(self.args.kg_vlm)

        # Semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(self.max_concurrency)
        valid_entries = []

        async def process_entry(idx:int):
            async with semaphore:
                entry = dataset[idx]
                chunks = multimodal_chunking_func(entry.image_path, entry.context, self.tokenizer)
                #print(f"Chunks: {chunks}")
                if not chunks:
                    return

                # Log the chunk details for debugging
                print(f"Processing file: {entry.image_path}")
                print(f"Chunks: {chunks}")

                # Extract entities and relations from chunk
                subgraph = await self.generate_sub_graph(chunks, client=vlm)
                if subgraph.id2entity:
                    await self.merge_graph(subgraph)
                    valid_entries.append(idx)

        # Create a list of tasks for processing all files in parallel
        tasks = [
            process_entry(idx)
            for idx in range(len(dataset))
        ]
        # tasks = [
        #     process_entry(idx)
        #     for idx in range(10)
        # ]
        #tasks = tasks[:10] # for test

        # Gather all tasks and wait for their completion
        #await asyncio.gather(*tasks)
        await async_task_runner(tasks,
                                max_concurrent=8,
                                describe="Processing files")
        return valid_entries
    
    async def process_chart_dataset(self, dataset:ChartMRag, sample_size=None, max_charts=None, max_texts=None):
        text_map = dataset.merged_text_map
        chart_map = dataset.chart_map
        chart_text_map = dataset.chart_text_map
        vlm = LLMRegistry.get(self.args.kg_vlm)

        # Semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(self.max_concurrency)

        temp_save_path = f"{self.args.save_path}_temp.pkl"
    
        if os.path.exists(temp_save_path):
            self.kg = HierarchicalMMKG.load(temp_save_path, args =self.args)
            print(f"Resuming from checkpoint, loaded {len(self.kg.valid_entries)} processed entries")
        else:
            self.kg.valid_entries = []

        entries_lock = asyncio.Lock()
        count_lock = asyncio.Lock()
        save_interval = 40 
        
        chart_count = 0
        text_count = 0
        
        # Time statistics
        chart_times = []
        text_times = []
        time_lock = asyncio.Lock()
        
        async def process_entry(idx:int):
            nonlocal chart_count, text_count
            async with semaphore:
                # Check if limits are reached
                async with count_lock:
                    if max_charts is not None and chart_count >= max_charts and max_texts is not None and text_count >= max_texts:
                        return
                
                entry = dataset[idx]
                entry_start_time = time.perf_counter()
                print(f"Processing page: {entry.ori_id}")
                chunks = []
                
                # Process text chunks with limit
                for id in entry.context:
                    async with count_lock:
                        if max_texts is not None and text_count >= max_texts:
                            break
                    chunks.append(
                        MMChunk(
                            id=id,
                            type="text",
                            context=text_map[id],
                            source=id
                        )
                    )
                    async with count_lock:
                        text_count += 1
                        if max_texts is not None:
                            print(f"  Added text chunk: {id} (text {text_count}/{max_texts})")
                
                # Process image chunks with limit
                for id in entry.image_paths:
                    async with count_lock:
                        if max_charts is not None and chart_count >= max_charts:
                            break
                    chunks.append(
                        MMChunk(
                            id=id,
                            type="image",
                            image_path=chart_map[id],
                            context=chart_text_map[id],
                            source=id
                        )
                    )
                    async with count_lock:
                        chart_count += 1
                        if max_charts is not None:
                            print(f"  Added chart chunk: {id} (chart {chart_count}/{max_charts})")
                
                #print(f"Chunks: {chunks}")
                if not chunks:
                    return
                
                # Extract entities and relations from chunk
                print(f"  Starting extraction for {len(chunks)} chunks...")
                chunk_start_time = time.perf_counter()
                subgraph = await self.generate_sub_graph(chunks, client=vlm)
                chunk_end_time = time.perf_counter()
                chunk_elapsed = chunk_end_time - chunk_start_time
                
                # Count chunks by type
                num_charts = len([c for c in chunks if c.type == "image"])
                num_texts = len([c for c in chunks if c.type == "text"])
                
                # Record time per chunk type (distribute time proportionally)
                async with time_lock:
                    if num_charts > 0:
                        avg_chart_time = chunk_elapsed / num_charts if num_charts > 0 else 0
                        chart_times.extend([avg_chart_time] * num_charts)
                    if num_texts > 0:
                        avg_text_time = chunk_elapsed / num_texts if num_texts > 0 else 0
                        text_times.extend([avg_text_time] * num_texts)
                
                entry_elapsed = chunk_end_time - entry_start_time
                print(f"  Processed {len(chunks)} chunks ({num_charts} charts, {num_texts} texts) in {chunk_elapsed:.2f}s (avg {chunk_elapsed/len(chunks):.2f}s per chunk)")
                
                if subgraph.id2entity:
                    await self.merge_graph(subgraph)
                    async with entries_lock:
                        self.kg.valid_entries.append(idx)  # 记录已处理的 entry_id
                        
                        # 每隔 save_interval 个 entry 保存一次 KG
                        if len(self.kg.valid_entries) % save_interval == 0:
                            self.kg.save(temp_save_path)
                            print(f"Saved KG checkpoint at {temp_save_path} (processed {len(self.kg.valid_entries)} entries)")
        all_indices = set(range(len(dataset)))
        processed_indices = set(self.kg.valid_entries)
        remaining_indices = sorted(all_indices - processed_indices)
        
        print(f"Total entries: {len(dataset)}, already processed: {len(processed_indices)}, remaining: {len(remaining_indices)}")
        if not sample_size:
        # 只处理未完成的 entry
            # Note: We create tasks for all remaining entries, but each task will check limits internally
            # The actual number of entries processed may be less than 267 if limits are reached early
            tasks = [
                process_entry(idx)
                for idx in remaining_indices
            ]
        else:
            # 随机抽取 sample_size 个未完成的 entry
            tasks = [
                process_entry(idx)
                for idx in random.sample(remaining_indices, min(sample_size, len(remaining_indices)))
            ]
        await async_task_runner(tasks, max_concurrent=1, describe="Processing Entries")
        
        # Print time statistics
        async with time_lock:
            if chart_times:
                avg_chart_time = sum(chart_times) / len(chart_times)
                print(f"\n=== Time Statistics ===")
                print(f"Charts processed: {len(chart_times)}")
                print(f"Average time per chart: {avg_chart_time:.2f}s")
                print(f"Total chart time: {sum(chart_times):.2f}s")
            
            if text_times:
                avg_text_time = sum(text_times) / len(text_times)
                print(f"Texts processed: {len(text_times)}")
                print(f"Average time per text: {avg_text_time:.2f}s")
                print(f"Total text time: {sum(text_times):.2f}s")
            
            if chart_times or text_times:
                total_chunks = len(chart_times) + len(text_times)
                total_time = sum(chart_times) + sum(text_times)
                print(f"Total chunks: {total_chunks}")
                print(f"Total processing time: {total_time:.2f}s")
                print(f"Average time per chunk: {total_time/total_chunks:.2f}s")

        return 


    async def insert(self,
                     stage: Literal["raw", "emb", "all"] = "raw",
                     dataset = None,
                     ):
        start = time.perf_counter()
        if stage in ("raw", "all"):
            
            if dataset:
                if dataset.name=="openwiki":
                    await self.process_dataset(dataset)

                elif dataset.name=="chartmrag":
                    max_charts = getattr(self.args, 'max_charts', None)
                    max_texts = getattr(self.args, 'max_texts', None)
                    await self.process_chart_dataset(dataset, sample_size=self.args.sample_size, max_charts=max_charts, max_texts=max_texts)

            #await self.process_folder(folder_path)
            

            #self.kg.get_layer(0).visualize(save_path=f"{self.args.save_path}.png")
            process_end = time.perf_counter()

            self.kg.save(f"{self.args.save_path}_no_resolution.pkl")
            print(f"Processing time: {process_end - start:.2f} seconds")

        if stage in ("emb", "all"):
            # Load unresolved KG if not already in memory
            self.kg = HierarchicalMMKG.load(path= f"{self.args.save_path}_no_resolution.pkl", args=self.args)

            # Resolve entities and compute embeddings
            self.load_emb_model()
            await self.resolve_entities()

            report = await self.kg.validate_and_repair_relations(
                repair=True,
                verbose=True
            )

            logger.info(f"[KG Validation Report] {report}")
            self.kg.emb(self.emb_model)

            

            self.kg.save(f"{self.args.save_path}.pkl")

            resolve_end = time.perf_counter()
            #print(f"time: {resolve_end - process_end:.2f} seconds")
            print(f"Total time: {resolve_end - start:.2f} seconds")
    
    def query_test(self, queries, top_k, rerank=False, corpus=None):
        """
        queries: dict | List[dict]
        topk: int
        rerank: bool
        corpus: {id:emb}
        return: List[dict] 对齐输入 queries

        query example:
            {
                "query_id": qa_id,
                "gold": qa_info["gt_ids"],  
                "query": qa_info["query"]
            }
        """
        if isinstance(queries, dict):  # 单个 query
            queries = [queries]
            single = True
        else:
            single = False

        self.load_emb_model()
        text_queries = [q["query"] for q in queries]
        query_embeddings = self.emb_model.encode(text_queries)

        del self.emb_model
        self.emb_model = None
        with torch.cuda.device(self.args.device):
            torch.cuda.empty_cache() 
        #TODO
        results = []
        for query, query_embedding in tqdm(zip(queries, query_embeddings), 
                                      total=len(queries),
                                      desc="Processing queries"):
            query["embedding"] = query_embedding
            result = self.kg.graph_search(
                query=query,
            )
            
            associated_chunks = result['associated_chunks']
            if rerank and corpus and associated_chunks:
                valid_chunks = []
                chunk_embeddings = []
                
                for chunk_id in associated_chunks:
                    if chunk_id in corpus:
                        valid_chunks.append(chunk_id)
                        chunk_embeddings.append(corpus[chunk_id])
                if valid_chunks:
                    chunk_embeddings = np.array(chunk_embeddings)

                    similarities = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        chunk_embeddings
                    )[0]
                    

                    chunk_score_pairs = list(zip(valid_chunks, similarities))
                    chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
 
                    rerank_chunks = [chunk_id for chunk_id, _ in chunk_score_pairs]
                    if len(rerank_chunks) > top_k:
                        rerank_chunks = rerank_chunks[:top_k]
                    result["associated_chunks"] = rerank_chunks
                    results.append(result)

        return results[0] if single else results
        

    async def generate_sub_graph(self, chunks: List[MMChunk], client):
        kg = HierarchicalMMKG(args = self.args)
        extract_results = await extract_chart_entities(chunks, hierachy=self.args.hierarchy, client = client)
        await kg.init_graph(extract_results)
        return kg

    async def merge_graph(self, subkg: HierarchicalMMKG):
        async with self.base_graph_lock:
            main_graph = self.kg
            await main_graph.merge_graph(subkg)

    async def resolve_entities(self):
        resolver = EntityResolver(vlm = self.args.vlm, use_minhash=False, embedding_model = self.emb_model)
        await resolver(self.kg)
    
