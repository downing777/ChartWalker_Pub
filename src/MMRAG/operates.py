import src.utils.utils
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import yaml
from src.VLMs import QwenChat, LLMRegistry
import json
from src.utils import safe_parser, async_task_runner, Tokenizer
from collections import defaultdict
import os
from PIL import Image
import json as pyjson
from src.schema import MMChunk, EntityType, RelationType
from src.Dataset.ChartMRag import ChartMRag, ChartEntry
import uuid
from copy import deepcopy
import logging
import time
import traceback                                   
from tenacity import RetryError

logger = logging.getLogger('lightrag')


prompts_path = "src/KG/kg_prompts.yaml"
with open(prompts_path, 'r') as file:
    PROMPTS = yaml.load(file, Loader=yaml.FullLoader)



def is_image(file_path: str) -> bool:
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

def is_text(file_path: str) -> bool:
    valid_extensions = ['.txt']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

def is_json(file_path: str) -> bool:
    return file_path.lower().endswith('.json')


def read_json_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = pyjson.load(f)
    #TODO deal with json or other data structure
    if isinstance(data, dict):
        return '\n'.join(str(v) for v in data.values() if isinstance(v, str))
    elif isinstance(data, list):
        return '\n'.join(str(item) for item in data if isinstance(item, str))
    return str(data)


def read_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def text_chunk(
    tokenizer: Tokenizer,
    content: str,
    source: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[MMChunk]:
    tokens = tokenizer.encode(content)
    results =  []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + max_token_size]
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            # results.append(
            #     {
            #         "tokens": _len,
            #         "content": chunk.strip(),
            #         "chunk_order_index": index,
            #         "type": "text",
            #     }
            # )
            results.append(MMChunk(
                id = f"{source}_text_{index}",
                source = source, 
                type = "text", 
                content = chunk.strip(),
            ))
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + max_token_size])
            results.append(
                MMChunk(
                id = f"{source}_text_{index}",
                source = source, 
                type = "text", 
                content = chunk_content.strip(),
                )
            )
    return results


def multimodal_chunking_func(file_path: str, context:Union[str, list[str]] = None, tokenizer: Tokenizer = None) -> List[MMChunk]:
    """
    image and textual chunking function, image itself is a chunk, text is chunked by tokenizer

    Args:
        file_path (str): The path to the file to be chunked.
        tokenizer (Tokenizer, optional): The tokenizer to use for text chunking. Defaults to None
    Returns:
    """
    chunks = []
    if not os.path.exists(file_path):
        return chunks
    if is_image(file_path):
        # TODO: find the corresponding context for other modal chunk
        if type(context) == list:
            context = "\n".join(context)
        chunk = image_chunk(file_path, context)
        chunks.append(chunk)
    elif is_text(file_path):
        if tokenizer is None:
            raise ValueError("Tokenizer required for text chunking")
        content = read_txt(file_path)
        for idx, chunk in enumerate(text_chunk(tokenizer, content)):
            chunk.source = file_path
            chunks.append(chunk)
    elif is_json(file_path):
        if tokenizer is None:
            raise ValueError("Tokenizer required for json chunking")
        content = read_json_text(file_path)
        for idx, chunk in enumerate(text_chunk(tokenizer, content)):
            chunk.source = file_path
            chunks.append(chunk)
    return chunks

# def chart_chunk_func() -> List[MMChunk]:
#     chunks = []
#     pass

async def extract_entities(
    chunks: List[MMChunk], hierachy: bool = False, client = None
) -> List[Dict[str, Any]]:
    """
    Args: chunks: List[MMChunk]
    Return: [{"chunk_id":..., "entities":..., "relationships":...}]
    """
    results = []
    # 构造prompt模板
    tasks = []
    for chunk in chunks:
        
        #logger.info(f"entity_extract_template: {entity_extract_template}")
        async def process_chunk(chunk:MMChunk):
            #file_path = getattr(chunk, "image_path", None) or getattr(chunk, "source", None)
            assert chunk.context is not None or chunk.image_path
            extraction_base = dict(
                entity_types=EntityType.description(),
                relation_types=RelationType.description(),
                language=PROMPTS["DEFAULTS"]["LANGUAGE"],
                input_text=getattr(chunk, "context", ""),
            )
            if hierachy:
                entity_extract_template = PROMPTS['prompts']["hierarchical_entity_extraction"].format(**extraction_base)
            else:
                entity_extract_template = PROMPTS['prompts']["entity_extraction"].format(**extraction_base)
            file_path = getattr(chunk, "image_path", None)
            prompt = entity_extract_template
            parsed = {}
            for i in range(3):
                try:
                    result = await client.async_chat(prompt=prompt, image_data=file_path, temperature=0.1)
                    parsed = safe_parser(result) 
                    
                    if parsed and isinstance(parsed, dict): 
                        break
                    else:
                        raise ValueError("Invalid response format")
                except Exception as e:
                    logger.warning(f"Retry {i+1}/3 failed for chunk {getattr(chunk, 'id', None)}: {e}")
            if not parsed:
                parsed = {"entities": [], "relationships": []}
                logger.info(f"chunk {chunk.id}, extraction failed")
            return {
                "chunk_id": getattr(chunk, "id", None),
                "entities": parsed.get("entities", []),
                "relationships": parsed.get("relationships", [])
            }
        tasks.append(process_chunk(chunk))
    #results = await asyncio.gather(*tasks)
    results = await async_task_runner(tasks, max_concurrent = 12)
    return results

# TODO 
async def extract_chart_entities(
    chunks: List[MMChunk],
    hierachy: bool = False,
    client=None,
    max_rounds: int = 3,
):
    # TODO 这里应该区分一下文本或者图片
    """
    Multi-round entity & relation extraction for the chart data.
    Args:
        chunks: List[MMChunk]
        hierachy: whether to use hierarchical extraction
        client: async_chat client
        max_rounds: number of extraction iterations
    Return:
        [{"chunk_id":..., "entities":..., "relationships":...}]
    """
    
    async def process_chunk(chunk:MMChunk):
        chunk_start_time = time.perf_counter()
        assert chunk.context is not None or chunk.image_path, f"Chunk {chunk.id} missing context/image"
        image_path = getattr(chunk, "image_path", None)
        chunk_type = "image" if image_path else "text"
        logger.info(f"[Extract] Starting {chunk_type} chunk: {chunk.id}")
        
            # ---- Base extraction prompt ----
        extraction_base = dict(
            entity_types=EntityType.description(),
            relation_types=RelationType.description(),
            input_text=getattr(chunk, "context", ""),
        )
        if chunk.type=="image":
            base_prompt = (
                PROMPTS["prompts"]["chart_hierarchical_entity_extraction"]
                if hierachy
                else PROMPTS["prompts"]["entity_extraction"]
            ).format(**extraction_base)
        else:
            base_prompt = (
                PROMPTS["prompts"]["hierarchical_entity_extraction"]
                if hierachy
                else PROMPTS["prompts"]["entity_extraction"]
            ).format(**extraction_base)


        all_entities, all_relationships = [], []
        history_messages = None  

        for round_idx in range(max_rounds):
            parsed = {}
            current_prompt = (
                base_prompt if round_idx == 0 else PROMPTS["prompts"]["continual_extraction"]
            )

            for retry_i in range(3):
                try:
                    max_tks = 32000 if round_idx == 0 else 16000
                    result = await client.async_chat(
                        prompt=current_prompt,
                        history_messages=history_messages,
                        image_data=image_path,
                        temperature=0.1,
                        max_tokens=max_tks,
                    )
                    parsed = safe_parser(result)
                    if parsed and isinstance(parsed, dict):
                        break
                    else:
                        # print(f"result: {result}")
                        # print(f"Parsed: {parsed}")
                        raise ValueError("Invalid response format")
                except Exception as e:
                    if isinstance(e, RetryError):
                        logger.error(
                            f"[{chunk.id}] Round {round_idx+1} Retry {retry_i+1}/3 RetryError"
                        )

                        # 1️⃣ 打印 RetryError 自身 traceback
                        logger.error("RetryError traceback:")
                        logger.error(traceback.format_exc())

                        # 2️⃣ 尝试取出最后一次真实异常
                        last_exc = None
                        if hasattr(e, "last_attempt") and e.last_attempt:
                            last_exc = e.last_attempt.exception()

                        if last_exc is not None:
                            logger.error("Underlying exception type: %s", type(last_exc))
                            logger.error("Underlying exception message: %s", last_exc)
                            logger.error("Underlying exception traceback:")
                            logger.error(
                                "".join(
                                    traceback.format_exception(
                                        type(last_exc),
                                        last_exc,
                                        last_exc.__traceback__,
                                    )
                                )
                            )
                    else:
                        logger.error(
                            f"[{chunk.id}] Round {round_idx+1} Retry {retry_i+1}/3 Exception"
                        )
                        logger.error(traceback.format_exc())


            if not parsed:
                logger.info(f"[{chunk.id}] Round {round_idx+1} extraction failed.")
                continue

            new_entities = parsed.get("entities", [])
            new_relationships = parsed.get("relationships", [])

            if not new_entities and not new_relationships:
                logger.info(
                    f"[{chunk.id}] No new entities/relations found after round {round_idx+1}. Stop early."
                )
                if round_idx == 0:
                    logger.debug(f"PROMPT:")
                    logger.debug(current_prompt)
                    logger.debug(f"Image Path:{image_path}")
                    logger.debug(f"LLM Result: {result}")
                break

            all_entities.extend(new_entities)
            all_relationships.extend(new_relationships)

            if history_messages is None:
                history_messages = [
                    {"role": "user", "content": current_prompt},
                    {"role": "assistant", "content": result},
                ]
            else:
                history_messages.extend([
                    {"role": "user", "content": current_prompt},
                    {"role": "assistant", "content": result},
                ])

           

        chunk_end_time = time.perf_counter()
        chunk_elapsed = chunk_end_time - chunk_start_time
        logger.info(f"[Extract] Completed chunk {chunk.id} in {chunk_elapsed:.2f}s (found {len(all_entities)} entities, {len(all_relationships)} relations)")
        
        return {
            "chunk_id": getattr(chunk, "id", None),
            "entities": all_entities,
            "relationships": all_relationships,
        }
    
    tasks = [process_chunk(chunk) for chunk in chunks]
    # Reduce concurrency to avoid API rate limiting
    # For thinking models (even with thinking disabled), use lower concurrency (2)
    # For regular models, use moderate concurrency (3) to avoid rate limits
    # Note: Even with thinking disabled, thinking models may still have rate limits
    model_name = getattr(client, 'model_name', '')
    is_thinking_model = 'thinking' in model_name.lower()
    max_concurrent = 2 if is_thinking_model else 3  # Reduced from 4 to 3 to avoid rate limits
    results = await async_task_runner(tasks, max_concurrent=max_concurrent)
    return results



if __name__ == "__main__":
   pass

