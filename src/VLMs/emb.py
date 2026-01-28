from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Any, Dict
import logging
from torch.cuda import OutOfMemoryError

logger = logging.getLogger(__name__)

# 全局加载模型（避免重复加载）
_LOCAL_EMBEDDING_MODEL = None
MODEL_PATH = "/share/project/tangning/model_hub/BAAI/bge-large-zh-v1.5"

def _get_embedding_model(model_path: str = MODEL_PATH):
    """单例模式获取模型"""
    global _LOCAL_EMBEDDING_MODEL
    if _LOCAL_EMBEDDING_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _LOCAL_EMBEDDING_MODEL = SentenceTransformer(
                model_path, 
                device=device
            )
            # 半精度加速（如果GPU支持）
            if device == "cuda":
                _LOCAL_EMBEDDING_MODEL = _LOCAL_EMBEDDING_MODEL.half()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    return _LOCAL_EMBEDDING_MODEL

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)  # bge-large-zh是1024维
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(OutOfMemoryError)  # 显存不足时重试
        | retry_if_exception_type(RuntimeError)    # 其他运行时错误
    ),
)
async def local_embed(
    texts: List[str],
    model: str = MODEL_PATH,  # 保持参数名一致，但实际使用本地模型
    base_url: str = None,  # 保持接口兼容性（忽略）
    api_key: str = None,  # 保持接口兼容性（忽略）
    client_configs: Dict[str, Any] = None,  # 可传入模型配置
    normalize: bool = True,  # 新增参数控制归一化
    **kwargs
) -> np.ndarray:
    """本地嵌入函数（兼容OpenAI接口）
    
    参数说明：
    - model: 实际为本地模型路径或名称
    - client_configs: 可包含 {
        "model_path": "自定义模型路径",
        "device": "cuda/cpu",
        "precision": "fp16/fp32"
      }
    """
    try:
        # 解析配置
        config = {
            "model_path": model,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "precision": "fp16",
            **(client_configs or {})
        }
        
        # 获取模型实例
        model = _get_embedding_model(config["model_path"])
        
        # 切换设备（如果配置不同）
        current_device = str(next(model.parameters()).device)
        if config["device"] != current_device.split(":")[0]:
            model.to(config["device"])
        
        # 精度处理
        if config["precision"] == "fp16" and config["device"] == "cuda":
            model.half()
        else:
            model.float()
        
        # 执行嵌入
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=32,  # 可根据显存调整
                **kwargs
            )
        
        return embeddings.astype(np.float32)  # 确保返回float32兼容OpenAI格式
    
    except OutOfMemoryError:
        logger.warning("GPU out of memory, retrying with smaller batch size...")
        # 自动降级处理
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=8,  # 减小batch size
                **kwargs
            )
        return embeddings.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise