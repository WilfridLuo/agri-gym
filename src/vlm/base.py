"""
VLM Interface - Abstract Base Classes for Vision Language Models

支持多种 VLM 后端：
- Qwen3 VL 系列
- Qwen2.5 VL 系列  
- Gemma 系列
- API Models (Claude, GPT-4V, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request


class VLMType(Enum):
    """VLM 类型"""
    QWEN3_VL = "qwen3_vl"
    QWEN2_5_VL = "qwen2.5_vl"
    GEMMA_VL = "gemma_vl"
    CLAUDE_API = "claude_api"
    GPT4V_API = "gpt4v_api"
    GEMINI_API = "gemini_api"
    VLLM_API = "vllm_api"
    LOCAL = "local"


@dataclass
class VLMConfig:
    """VLM 配置"""
    model_type: VLMType
    model_name: str
    model_path: Optional[str] = None  # 本地模型路径
    api_key: Optional[str] = None  # API 密钥
    api_base: Optional[str] = None  # API 基础 URL
    device: str = "cuda"  # cuda, cpu
    dtype: str = "bfloat16"  # 数据类型
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    local_files_only: bool = False


@dataclass
class VLMMessage:
    """VLM 消息格式"""
    role: str  # "user", "assistant", "system"
    content: Union[str, List[Dict[str, Any]]]  # 文本或包含图像的内容


@dataclass
class VLMResponse:
    """VLM 响应"""
    text: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = None


class BaseVLM(ABC):
    """
    Vision Language Model 基类
    
    所有 VLM 实现都应该继承这个类
    """
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """初始化模型和 tokenizer"""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """
        生成文本响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Returns:
            VLMResponse
        """
        pass
    
    @abstractmethod
    def encode_image(self, image_path: str) -> str:
        """
        编码图像为模型可以接受的格式
        
        Args:
            image_path: 图像路径
            
        Returns:
            编码后的图像（通常是 base64 或 tensor）
        """
        pass
    
    def chat(
        self,
        query: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        简化的对话接口
        
        Args:
            query: 用户查询
            image_path: 图像路径（可选）
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            模型生成的文本
        """
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append(VLMMessage(role="system", content=system_prompt))
        
        # 构建用户消息
        if image_path:
            # 多模态消息
            content = [
                {"type": "image", "image": self.encode_image(image_path)},
                {"type": "text", "text": query}
            ]
        else:
            # 纯文本消息
            content = query
        
        messages.append(VLMMessage(role="user", content=content))
        
        # 生成响应
        response = self.generate(messages, **kwargs)
        return response.text
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "type": self.config.model_type.value,
            "name": self.config.model_name,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "max_tokens": self.config.max_tokens
        }


class LocalVLM(BaseVLM):
    """本地 VLM 的基类（Qwen, Gemma 等）"""

    def _resolve_model_id(self) -> str:
        """Resolve model identifier from config."""
        model_id = self.config.model_path or self.config.model_name
        if not model_id:
            raise ValueError("model_path/model_name cannot both be empty for local VLM")
        return model_id

    def _resolve_torch_dtype(self, torch_module):
        """Resolve torch dtype from config string."""
        dtype_map = {
            "float32": torch_module.float32,
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch_module.bfloat16)
    
    def _initialize(self):
        """初始化本地模型"""
        try:
            import transformers
            from transformers import AutoProcessor, AutoTokenizer
            import torch
            
            model_id = self._resolve_model_id()
            dtype = self._resolve_torch_dtype(torch)

            print(f"Loading VLM from {model_id}...")

            # For VLM, processor is the primary interface.
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=self.config.local_files_only,
            )

            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if self.config.device == "cuda":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "cpu"

            # Prefer explicit vision model classes by model type first.
            errors = []
            candidate_class_names: List[str] = []

            if self.config.model_type == VLMType.QWEN3_VL:
                candidate_class_names.extend([
                    "Qwen3VLForConditionalGeneration",
                    "AutoModelForImageTextToText",
                    "Qwen2_5_VLForConditionalGeneration",
                    "Qwen2VLForConditionalGeneration",
                ])
            elif self.config.model_type == VLMType.QWEN2_5_VL:
                candidate_class_names.extend([
                    "Qwen2_5_VLForConditionalGeneration",
                    "AutoModelForImageTextToText",
                    "Qwen2VLForConditionalGeneration",
                ])
            elif self.config.model_type == VLMType.GEMMA_VL:
                candidate_class_names.extend([
                    "PaliGemmaForConditionalGeneration",
                    "Gemma3ForConditionalGeneration",
                    "AutoModelForImageTextToText",
                ])

            # Version-compatible generic fallbacks.
            candidate_class_names.extend([
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
                "AutoModelForCausalLM",
            ])

            candidate_classes = []
            for name in candidate_class_names:
                cls = getattr(transformers, name, None)
                if cls is not None and cls not in candidate_classes:
                    candidate_classes.append(cls)

            self.model = None
            for cls in candidate_classes:
                try:
                    self.model = cls.from_pretrained(model_id, **model_kwargs)
                    break
                except Exception as e:
                    errors.append(f"{cls.__name__}: {e}")

            if self.model is None:
                raise RuntimeError(
                    "Failed to load VLM model using vision-capable classes. "
                    f"Tried: {[c.__name__ for c in candidate_classes]}\n"
                    + "\n".join(errors)
                )

            # Keep tokenizer compatibility for legacy callers.
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=self.config.local_files_only,
                )

            print(f"✓ VLM loaded successfully: {self.model.__class__.__name__}")
        
        except ImportError as e:
            raise ImportError(
                f"Required packages for LocalVLM not installed: {e}\n"
                "Install with: pip install transformers torch"
            )
    
    def encode_image(self, image_path: str) -> str:
        """编码图像"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _decode_image_if_needed(self, image_value: Any) -> Any:
        """Decode base64 image to PIL when needed for processor input."""
        if not isinstance(image_value, str):
            return image_value

        # Path/URL-like strings should be passed through.
        if image_value.startswith(("/", "http://", "https://")):
            return image_value

        try:
            from PIL import Image

            binary = base64.b64decode(image_value)
            return Image.open(BytesIO(binary)).convert("RGB")
        except Exception:
            # If not valid base64, keep raw string.
            return image_value

    def _messages_to_processor_inputs(self, messages: List[VLMMessage]):
        """Convert generic messages to processor chat format."""
        hf_messages: List[Dict[str, Any]] = []
        images: List[Any] = []

        for msg in messages:
            if isinstance(msg.content, str):
                content = [{"type": "text", "text": msg.content}]
            else:
                content = []
                for item in msg.content:
                    item_type = item.get("type")
                    if item_type == "text":
                        content.append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "image":
                        image_obj = self._decode_image_if_needed(item.get("image"))
                        content.append({"type": "image", "image": image_obj})
                        images.append(image_obj)

            hf_messages.append({"role": msg.role, "content": content})

        return hf_messages, images

    def _generate_with_processor(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> VLMResponse:
        """Generic VLM generation path based on processor + model."""
        import torch

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens
        do_sample = bool(kwargs.pop("do_sample", False))

        hf_messages, images = self._messages_to_processor_inputs(messages)

        # Prefer chat template when available.
        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "\n".join(
                [f"{m['role']}: " + " ".join([c.get("text", "") for c in m["content"] if c.get("type") == "text"]) for m in hf_messages]
            )

        processor_kwargs = {
            "text": [prompt_text],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images

        inputs = self.processor(**processor_kwargs)

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_tok,
            **kwargs,
        }
        if do_sample:
            generate_kwargs["top_p"] = self.config.top_p
        generate_kwargs["do_sample"] = do_sample
        if do_sample and temp is not None and temp > 0:
            generate_kwargs["temperature"] = temp

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = outputs[:, input_len:] if input_len > 0 else outputs

        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
            response_text = decoded[0] if decoded else ""
        else:
            response_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        return VLMResponse(
            text=response_text.strip(),
            tokens_used=int(generated.shape[-1]) if hasattr(generated, "shape") else 0,
            metadata={"model": self.config.model_name},
        )
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """生成响应"""
        raise NotImplementedError(
            "Subclasses must implement generate() method"
        )


class APIVLM(BaseVLM):
    """API-based VLM 的基类（Claude, GPT-4V 等）"""
    
    def _initialize(self):
        """初始化 API 连接"""
        if not self.config.api_key:
            raise ValueError(f"API key required for {self.config.model_type.value}")
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image as a standard JPEG data URL for API providers."""
        return self._image_to_data_url(image_path)

    @classmethod
    def _image_to_data_url(cls, image_value: Any) -> str:
        mime_type, encoded = cls._image_to_mime_and_base64(image_value)
        if mime_type == "url":
            return encoded
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _split_data_url(value: str) -> tuple[str, str] | None:
        if not value.startswith("data:") or "," not in value:
            return None
        header, encoded = value.split(",", 1)
        mime_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
        return mime_type, encoded

    @classmethod
    def _image_to_mime_and_base64(cls, image_value: Any) -> tuple[str, str]:
        """Normalize HF/PIL/path/base64/data-url images to API-safe JPEG/PNG/WebP/GIF."""
        if isinstance(image_value, str):
            if image_value.startswith(("http://", "https://")):
                return "url", image_value
            data_url = cls._split_data_url(image_value)
            if data_url is not None:
                mime_type, encoded = data_url
                if mime_type in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
                    return mime_type, encoded
                return cls._binary_image_to_jpeg(base64.b64decode(encoded))
            if os.path.exists(image_value):
                with open(image_value, "rb") as f:
                    return cls._binary_image_to_jpeg(f.read())
            try:
                raw = base64.b64decode(image_value, validate=True)
                return cls._binary_image_to_jpeg(raw)
            except Exception:
                raise ValueError(f"Unsupported image string for API VLM: {image_value[:80]}")

        try:
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise ImportError("Pillow and numpy are required for API image normalization") from e

        if isinstance(image_value, Image.Image):
            return cls._pil_image_to_jpeg(image_value)
        if isinstance(image_value, np.ndarray):
            return cls._pil_image_to_jpeg(Image.fromarray(image_value))

        raise TypeError(f"Unsupported image input for API VLM: {type(image_value)}")

    @staticmethod
    def _binary_image_to_jpeg(raw: bytes) -> tuple[str, str]:
        from PIL import Image

        with Image.open(BytesIO(raw)) as img:
            return APIVLM._pil_image_to_jpeg(img)

    @staticmethod
    def _pil_image_to_jpeg(image: Any) -> tuple[str, str]:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "image/jpeg", encoded
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """生成响应"""
        raise NotImplementedError(
            "Subclasses must implement generate() method"
        )


# ============================================================================
# Concrete Implementations
# ============================================================================

class Qwen3VL(LocalVLM):
    """Qwen3 Vision Language Model"""
    
    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.QWEN3_VL:
            config.model_type = VLMType.QWEN3_VL
        super().__init__(config)
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """Generate response using Qwen3 VL"""
        return self._generate_with_processor(messages, temperature, max_tokens, **kwargs)
    
    def _format_messages(self, messages: List[VLMMessage]) -> str:
        """将消息格式化为 Qwen3 可以接受的格式"""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n"
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    formatted += f"User: {msg.content}\n"
                else:
                    # 处理多模态内容
                    for item in msg.content:
                        if item.get("type") == "text":
                            formatted += f"User: {item['text']}\n"
                        elif item.get("type") == "image":
                            formatted += f"[Image]\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
        return formatted


class Qwen2_5VL(LocalVLM):
    """Qwen2.5 Vision Language Model"""
    
    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.QWEN2_5_VL:
            config.model_type = VLMType.QWEN2_5_VL
        super().__init__(config)
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """Generate response using Qwen2.5 VL"""
        return self._generate_with_processor(messages, temperature, max_tokens, **kwargs)
    
    def _format_messages(self, messages: List[VLMMessage]) -> str:
        """将消息格式化为 Qwen2.5 可以接受的格式"""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n"
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    formatted += f"User: {msg.content}\n"
                else:
                    for item in msg.content:
                        if item.get("type") == "text":
                            formatted += f"User: {item['text']}\n"
                        elif item.get("type") == "image":
                            formatted += f"[Image]\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
        return formatted


class GemmaVL(LocalVLM):
    """Gemma Vision Language Model"""
    
    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.GEMMA_VL:
            config.model_type = VLMType.GEMMA_VL
        super().__init__(config)
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """Generate response using Gemma VL"""
        return self._generate_with_processor(messages, temperature, max_tokens, **kwargs)
    
    def _format_messages(self, messages: List[VLMMessage]) -> str:
        """将消息格式化为 Gemma 可以接受的格式"""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n"
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    formatted += f"User: {msg.content}\n"
                else:
                    for item in msg.content:
                        if item.get("type") == "text":
                            formatted += f"User: {item['text']}\n"
                        elif item.get("type") == "image":
                            formatted += f"[Image]\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
        return formatted


class ClaudeAPI(APIVLM):
    """Claude API (Claude 3 Vision)"""
    
    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.CLAUDE_API:
            config.model_type = VLMType.CLAUDE_API
        super().__init__(config)
    
    def _initialize(self):
        """初始化 Claude API"""
        super()._initialize()
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package required for Claude API\n"
                "Install with: pip install anthropic"
            )
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """Generate response using Claude API"""
        temp = temperature or self.config.temperature
        max_tok = max_tokens or self.config.max_tokens
        
        # 将消息转换为 Claude 格式
        formatted_messages = self._format_messages_for_api(messages)
        system_prompt = self._extract_system_prompt(messages)
        request_kwargs = dict(kwargs)
        if system_prompt:
            request_kwargs["system"] = system_prompt
        
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=max_tok,
            temperature=temp,
            messages=formatted_messages,
            **request_kwargs
        )
        
        return VLMResponse(
            text=response.content[0].text,
            tokens_used=response.usage.output_tokens,
            metadata={"model": "claude", "usage": response.usage}
        )
    
    def _format_messages_for_api(self, messages: List[VLMMessage]) -> List[Dict]:
        """将消息转换为 Claude API 格式"""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                # Claude 用 system 参数而不是消息
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            formatted.append({
                "role": role,
                "content": self._format_content_for_api(msg.content)
            })
        return formatted

    @staticmethod
    def _extract_system_prompt(messages: List[VLMMessage]) -> str:
        parts = []
        for msg in messages:
            if msg.role != "system":
                continue
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
        return "\n\n".join(part for part in parts if part)

    @classmethod
    def _format_content_for_api(cls, content: Any) -> Any:
        if isinstance(content, str):
            return content
        formatted = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                formatted.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                media_type, encoded = cls._image_to_mime_and_base64(item.get("image", ""))
                if media_type == "url":
                    formatted.append({
                        "type": "image",
                        "source": {"type": "url", "url": encoded},
                    })
                    continue
                formatted.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded,
                    },
                })
        return formatted or str(content)


class GPT4VAPI(APIVLM):
    """GPT-4 Vision API"""
    
    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.GPT4V_API:
            config.model_type = VLMType.GPT4V_API
        super().__init__(config)
    
    def _initialize(self):
        """初始化 GPT-4V API"""
        super()._initialize()
        try:
            import openai
            client_kwargs = {"api_key": self.config.api_key}
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "OpenAI package required for GPT-4V API\n"
                "Install with: pip install openai"
            )
    
    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> VLMResponse:
        """Generate response using GPT-4V API"""
        temp = temperature or self.config.temperature
        max_tok = max_tokens or self.config.max_tokens
        
        formatted_messages = self._format_messages_for_api(messages)
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=formatted_messages,
            max_tokens=max_tok,
            temperature=temp,
            **kwargs
        )
        
        return VLMResponse(
            text=response.choices[0].message.content,
            tokens_used=response.usage.completion_tokens,
            metadata={"model": "gpt4v", "usage": response.usage}
        )
    
    def _format_messages_for_api(self, messages: List[VLMMessage]) -> List[Dict]:
        """将消息转换为 GPT-4V API 格式"""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": self._format_content_for_api(msg.content)
            })
        return formatted

    @classmethod
    def _format_content_for_api(cls, content: Any) -> Any:
        if isinstance(content, str):
            return content
        formatted = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                formatted.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                url = cls._image_to_data_url(item.get("image", ""))
                formatted.append({"type": "image_url", "image_url": {"url": url}})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    formatted.append(item)
                else:
                    formatted.append({"type": "image_url", "image_url": {"url": cls._image_to_data_url(url)}})
        return formatted or str(content)


class GeminiAPI(APIVLM):
    """Google Gemini API via REST generateContent."""

    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.GEMINI_API:
            config.model_type = VLMType.GEMINI_API
        super().__init__(config)

    def _initialize(self):
        super()._initialize()
        base = self.config.api_base or "https://generativelanguage.googleapis.com/v1beta"
        self.api_base = base.rstrip("/")

    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> VLMResponse:
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens
        system_instruction, contents = self._format_messages_for_api(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tok,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        endpoint = f"{self.api_base}/models/{self.config.model_name}:generateContent?key={self.config.api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=kwargs.get("timeout", 600)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed with HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Gemini API at {self.api_base}: {e}") from e

        candidates = data.get("candidates") or []
        content = (candidates[0].get("content") or {}) if candidates else {}
        parts = content.get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        usage = data.get("usageMetadata") or {}
        return VLMResponse(
            text=text,
            tokens_used=int(usage.get("candidatesTokenCount") or usage.get("totalTokenCount") or 0),
            finish_reason=(candidates[0].get("finishReason") if candidates else "stop") or "stop",
            metadata={"model": "gemini", "usage": usage},
        )

    def _format_messages_for_api(self, messages: List[VLMMessage]) -> tuple[str, List[Dict[str, Any]]]:
        system_parts = []
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(self._content_to_text(msg.content))
                continue
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": self._content_to_parts(msg.content)})
        return "\n\n".join([p for p in system_parts if p]), contents

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(texts)

    @classmethod
    def _content_to_parts(cls, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image":
                mime_type, image_data = cls._image_to_mime_and_base64(item.get("image", ""))
                if mime_type == "url":
                    parts.append({"text": f"[Image URL omitted for Gemini REST request: {image_data}]"})
                    continue
                parts.append({"inline_data": {"mime_type": mime_type, "data": image_data}})
        return parts or [{"text": str(content)}]


class VLLMAPI(BaseVLM):
    """OpenAI-compatible vLLM server client.

    This client intentionally uses stdlib HTTP instead of the openai package so
    benchmark inference can target a vLLM server without adding another Python
    dependency to the agent process.
    """

    def __init__(self, config: VLMConfig):
        if config.model_type != VLMType.VLLM_API:
            config.model_type = VLMType.VLLM_API
        super().__init__(config)

    def _initialize(self):
        self.api_base = (
            self.config.api_base
            or os.environ.get("VLLM_API_BASE")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.api_key = self.config.api_key or os.environ.get("VLLM_API_KEY") or "EMPTY"
        self.endpoint = f"{self.api_base}/chat/completions"

    def encode_image(self, image_path: str) -> str:
        return self._image_to_data_url(image_path)

    def _image_to_data_url(self, image_value: Any) -> str:
        if isinstance(image_value, str):
            if image_value.startswith(("http://", "https://", "data:image/")):
                return image_value

            if os.path.exists(image_value):
                mime_type = mimetypes.guess_type(image_value)[0] or "image/jpeg"
                with open(image_value, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded}"

            try:
                base64.b64decode(image_value, validate=True)
                return f"data:image/jpeg;base64,{image_value}"
            except Exception:
                return image_value

        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required to send in-memory images to vLLM") from e

        if isinstance(image_value, Image.Image):
            buffer = BytesIO()
            image_value.convert("RGB").save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

        raise TypeError(f"Unsupported image input for vLLM: {type(image_value)}")

    def _format_content_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_type = item.get("type")
        if item_type == "text":
            return {"type": "text", "text": item.get("text", "")}
        if item_type == "image":
            return {
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(item.get("image"))},
            }
        if item_type == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, str):
                image_url = {"url": image_url}
            return {"type": "image_url", "image_url": image_url}
        return {"type": "text", "text": str(item)}

    def _merge_api_content(self, left: Any, right: Any) -> Any:
        if isinstance(left, str) and isinstance(right, str):
            return left.rstrip() + "\n\n" + right.lstrip()

        def as_parts(value: Any) -> List[Dict[str, Any]]:
            if isinstance(value, str):
                return [{"type": "text", "text": value}]
            if isinstance(value, list):
                return [dict(item) if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            return [{"type": "text", "text": str(value)}]

        return as_parts(left) + as_parts(right)

    def _coalesce_chat_roles(self, messages: List[VLMMessage]) -> List[VLMMessage]:
        """Make chat-template input robust for stricter vLLM model templates."""
        merged: List[VLMMessage] = []
        for msg in messages:
            role = msg.role if msg.role in {"system", "user", "assistant"} else "user"
            current = VLMMessage(role=role, content=msg.content)
            if not merged:
                merged.append(current)
                continue

            if merged[-1].role == current.role:
                merged[-1] = VLMMessage(
                    role=merged[-1].role,
                    content=self._merge_api_content(merged[-1].content, current.content),
                )
                continue

            if current.role == "system":
                # Most vLLM chat templates expect system only at the beginning.
                # Late system messages are treated as user-visible instructions.
                current = VLMMessage(role="user", content=current.content)
                if merged[-1].role == current.role:
                    merged[-1] = VLMMessage(
                        role=merged[-1].role,
                        content=self._merge_api_content(merged[-1].content, current.content),
                    )
                    continue

            merged.append(current)

        first_non_system = next((idx for idx, msg in enumerate(merged) if msg.role != "system"), None)
        if first_non_system is not None and merged[first_non_system].role == "assistant":
            merged.insert(first_non_system, VLMMessage(role="user", content="Continue the task."))
        return merged

    def _format_messages_for_api(self, messages: List[VLMMessage]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in self._coalesce_chat_roles(messages):
            if isinstance(msg.content, str):
                content = msg.content
            else:
                content = [self._format_content_item(item) for item in msg.content]
            formatted.append({"role": msg.role, "content": content})
        return formatted

    def generate(
        self,
        messages: List[VLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> VLMResponse:
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        payload = {
            "model": self.config.model_name,
            "messages": self._format_messages_for_api(messages),
            "max_tokens": max_tok,
            "temperature": temp,
            "top_p": self.config.top_p,
        }
        allowed_extra = {
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "stop",
            "top_k",
            "min_p",
            "seed",
        }
        for key, value in kwargs.items():
            if key in allowed_extra and value is not None:
                payload[key] = value

        response_data = self._post_chat_completion(payload, timeout=kwargs.get("timeout", 600))

        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response_data.get("usage") or {}
        return VLMResponse(
            text=(message.get("content") or "").strip(),
            tokens_used=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason") or "stop",
            metadata={"model": self.config.model_name, "api_base": self.api_base, "usage": usage},
        )

    def _post_chat_completion(self, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        current_payload = dict(payload)
        retries = []
        for _ in range(4):
            data = json.dumps(current_payload).encode("utf-8")
            request = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                    if retries:
                        response_data["_agri_retry"] = {"max_tokens_retries": retries}
                    return response_data
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                retry_payload = self._context_length_retry_payload(current_payload, body)
                if retry_payload is None:
                    raise RuntimeError(f"vLLM request failed with HTTP {e.code}: {body}") from e
                retries.append(retry_payload.get("max_tokens"))
                current_payload = retry_payload
            except urllib.error.URLError as e:
                raise RuntimeError(f"Could not reach vLLM server at {self.endpoint}: {e}") from e

        raise RuntimeError(f"vLLM request exceeded context after max_tokens retries: {retries}")

    @staticmethod
    def _context_length_retry_payload(payload: Dict[str, Any], error_body: str) -> Optional[Dict[str, Any]]:
        if "maximum context length" not in error_body:
            return None

        max_ctx_match = re.search(r"maximum context length is\s+(\d+)", error_body)
        input_match = re.search(r"prompt contains at least\s+(\d+)\s+input tokens", error_body)
        if not max_ctx_match or not input_match:
            return None

        max_ctx = int(max_ctx_match.group(1))
        input_tokens = int(input_match.group(1))
        allowed = max_ctx - input_tokens - 256
        if allowed < 32:
            strict_allowed = max_ctx - input_tokens - 1
            if strict_allowed < 32:
                return None
            allowed = min(64, strict_allowed)

        old_max = int(payload.get("max_tokens") or allowed)
        new_max = min(old_max, allowed)
        if new_max >= old_max:
            return None

        retry_payload = dict(payload)
        retry_payload["max_tokens"] = max(32, new_max)
        return retry_payload


# ============================================================================
# VLM Factory
# ============================================================================

class VLMFactory:
    """VLM 工厂类"""
    
    _vlm_classes = {
        VLMType.QWEN3_VL: Qwen3VL,
        VLMType.QWEN2_5_VL: Qwen2_5VL,
        VLMType.GEMMA_VL: GemmaVL,
        VLMType.CLAUDE_API: ClaudeAPI,
        VLMType.GPT4V_API: GPT4VAPI,
        VLMType.GEMINI_API: GeminiAPI,
        VLMType.VLLM_API: VLLMAPI,
    }
    
    @classmethod
    def create(cls, config: VLMConfig) -> BaseVLM:
        """创建 VLM 实例"""
        vlm_class = cls._vlm_classes.get(config.model_type)
        if not vlm_class:
            raise ValueError(f"Unknown VLM type: {config.model_type}")
        return vlm_class(config)
    
    @classmethod
    def register(cls, vlm_type: VLMType, vlm_class: type):
        """注册自定义 VLM"""
        cls._vlm_classes[vlm_type] = vlm_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出可用的 VLM"""
        return [vt.value for vt in cls._vlm_classes.keys()]
