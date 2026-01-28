# 修改image_path、message内容即可
import dataclasses
from openai import OpenAI, AsyncOpenAI
import base64
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
import async_timeout
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any
from PIL import Image
import io
from src.VLMs.llm_registry import LLM, LLMRegistry
from dataclasses import dataclass
import base64
import random
from mimetypes import guess_type

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    url: List[str]
    max_len: int
    key:str=None
    temperature: float = 0.8
    top_p: float = 0.9
    retry_attempts: int = 20
    timeout: int = 200
    think_bool: bool = False
    openai_client: Optional[Any] = None

MODEL_CONFIGS = {
    "qwen3-vl-8b": ModelConfig(
        url=["YOUR_API_URL"],
        key="EMPTY",
        max_len=8192,
        temperature=0.6,
    ),
}

@LLMRegistry.register('QwenChat')
class QwenChat(LLM):
    def __init__(
        self, 
        model_name: str ="qwen3-vl-8b",
        default_max_tokens: int = 6000,
        default_temperature: float = 0.7
    ):
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        # Configure HTTP client with longer timeout for thinking models
        import httpx
        timeout_config = httpx.Timeout(
            connect=30.0,  # Connection timeout
            read=3600.0 if self.model_config.think_bool else 200.0,  # Read timeout (60 min for thinking, 200s for regular)
            write=30.0,  # Write timeout
            pool=30.0  # Pool timeout
        )
        
        self.aclients = [AsyncOpenAI(
            api_key=self.model_config.key,
            base_url=url,
            http_client=httpx.AsyncClient(timeout=timeout_config),
        ) for url in self.model_config.url]
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
    
    @staticmethod
    def _image_to_data_url_from_bytes(data: bytes, mime: str) -> str:
        return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

        
    @staticmethod
    def _process_image(image_input: Union[str, Image.Image]) -> str:
        if isinstance(image_input, Image.Image):
            buf = io.BytesIO()
            image_input.save(buf, format="PNG")
            return QwenChat._image_to_data_url_from_bytes(buf.getvalue(), "image/png")

        if image_input.startswith(("http://", "https://", "data:image")):
            return image_input

        mime, _ = guess_type(image_input)
        if mime is None:
            mime = "application/octet-stream"

        with open(image_input, "rb") as f:
            return QwenChat._image_to_data_url_from_bytes(f.read(), mime)
    
    def _build_message(
        self,
        content: Union[str, List[Dict]],
        role: str = "user"
    ) -> Dict:
        return {"role": role, "content": content}
    
    def prepare_messages(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        image_data: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None
    ) -> List[Dict]:
        """
        Prepares the messages for the chat completion request.
        """
        messages = []
        if system_prompt:
            messages.append(self._build_message(system_prompt, "system"))
        
        if history_messages:
            messages.extend(history_messages)
        
        if prompt or image_data:
            content = []
            
            if prompt:
                content.append({"type": "text", "text": prompt})

            if image_data:
                if not isinstance(image_data, list):
                    image_data = [image_data]
                image_messages = []
                for img in image_data:
                    url = self._process_image(img)
                    image_messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        }
                    })
                content.extend(image_messages)
            if len(content) == 1 and content[0]["type"] == "text":
                messages.append(self._build_message(content[0]["text"], "user"))
            else:
                messages.append(self._build_message(content, "user"))
        return messages
    
    @retry(
        wait=wait_random_exponential(multiplier=2, min=4, max=120),  # Longer wait for rate limits (4-120s)
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, Exception))  # Retry on rate limits and other errors
    )
    async def async_chat(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        image_data: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        show_tokens: bool = False,
        **kwargs
    ) -> str:
        """
        Asynchronously sends a chat completion request to the OpenAI API.
        """
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature

        messages = self.prepare_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            image_data=image_data
        )
        aclient = random.choice(self.aclients)
        try:
            # For thinking models, use much longer timeout (3600 seconds = 60 minutes)
            # Regular models use model_config timeout (200 seconds)
            # Note: Some thinking model requests can take 50+ minutes
            timeout_seconds = 3600 if self.model_config.think_bool else self.model_config.timeout
            async with async_timeout.timeout(timeout_seconds):
                response = await aclient.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={
                        "top_k": 20, 
                        "chat_template_kwargs": {
                            #"enable_thinking": self.model_config.think_bool,
                            "thinking": self.model_config.think_bool
                        }
                        
                    },
                    **kwargs
                )
            if show_tokens:
                print({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                })
            msg = response.choices[0].message
            content = msg.content or ""

            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                return reasoning + "\n</think>\n" + content

            return content

        except RateLimitError as e:
            print(f"Rate limit error (429): {e}. Will retry with exponential backoff...")
            raise
        except Exception as e:
            error_msg = str(e)
            # Check if it's a 429 error even if not RateLimitError
            if "429" in error_msg or "rate limit" in error_msg.lower() or "Too Many Requests" in error_msg:
                print(f"Rate limit error detected: {error_msg}. Will retry with exponential backoff...")
            else:
                print(f"Error during async chat: {error_msg}")
            raise 

    
    def generate_response(self, messages, max_tokens=None, temperature=None):
        '''
        deprecated, use async_chat instead
        '''
        raise NotImplementedError("This class does not support generate_response method.")

    async def generate_response_async(self, messages, max_tokens=None, temperature=None):
        '''
        deprecated, use async_chat instead
        '''
        raise NotImplementedError("This class does not support async generation.")
