"""
MIRAGE MMST Dataset Loader

从 HuggingFace 加载 MIRAGE MMST 数据集
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from urllib.parse import quote


@dataclass
class MMSTItem:
    """MIRAGE MMST 数据项"""
    id: str
    question: str
    reference_answer: str
    image_path: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    image_inputs: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MMSTDatasetLoader:
    """MIRAGE MMST 数据集加载器"""
    
    def __init__(self, split: str = "train", image_cache_dir: Optional[str] = None):
        """
        初始化加载器
        
        Args:
            split: 数据集分割（train/test）
        """
        self.split = split
        self.data: List[MMSTItem] = []
        self.hf_dataset = None
        self.image_cache_dir = image_cache_dir
        self.subset = "MMST_Standard"
    
    def load_from_huggingface(self, subset: str = "MMST_Standard", image_cache_dir: Optional[str] = None):
        """
        从 HuggingFace 加载 MMST 数据
        
        Args:
            subset: 子集名称（MMST_Standard, MMMT_Direct, etc）
        """
        if image_cache_dir:
            self.image_cache_dir = image_cache_dir
        self.subset = subset

        try:
            from datasets import load_dataset
            
            print(f"Loading MIRAGE {subset}/{self.split} from HuggingFace...")
            
            self.hf_dataset = load_dataset(
                "MIRAGE-Benchmark/MIRAGE",
                subset,
                split=self.split,
                streaming=False
            )
            
            print(f"✓ Loaded {len(self.hf_dataset)} samples")
            self.data = []
        
        except ImportError:
            raise ImportError(
                "HuggingFace datasets required\n"
                "Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load from HuggingFace: {e}")
    
    def _convert_from_hf(self, limit: Optional[int] = None, offset: int = 0):
        """将 HF 数据集转换为 MMSTItem"""
        self.data = []

        source = self.hf_dataset
        if limit is not None or offset:
            start = max(0, offset)
            stop = len(self.hf_dataset) if limit is None else min(len(self.hf_dataset), start + limit)
            source = self.hf_dataset.select(range(start, stop))

        for idx, item in enumerate(source):
            item_id = str(item.get("id", str(idx)))
            image_refs, image_inputs = self._extract_hf_images(item, item_id=item_id)
            mmst_item = MMSTItem(
                id=item_id,
                question=item.get("question", ""),
                reference_answer=item.get("answer", ""),
                image_path=image_refs[0] if image_refs else None,
                image_paths=image_refs,
                image_inputs=image_inputs,
                metadata={
                    k: v for k, v in item.items()
                    if k not in ["id", "question", "answer", "image", "image_1", "image_2", "image_3"]
                }
            )
            self.data.append(mmst_item)

    def _extract_hf_images(self, item: Dict[str, Any], item_id: str) -> tuple[List[str], List[Any]]:
        """Keep decoded HF Image columns in memory and expose stable logical refs."""
        refs: List[str] = []
        inputs: List[Any] = []
        image_keys = [k for k in ("image", "image_1", "image_2", "image_3") if item.get(k) is not None]

        for key in image_keys:
            value = item.get(key)
            image_input = self._normalize_hf_image_value(value)
            if image_input is None:
                continue
            ref = f"mirage://{quote(self.subset, safe='')}/{quote(self.split, safe='')}/{quote(str(item_id), safe='')}/{key}"
            refs.append(ref)
            inputs.append(image_input)

        return refs, inputs

    @staticmethod
    def _normalize_hf_image_value(value: Any) -> Optional[Any]:
        if isinstance(value, dict):
            value = value.get("image") or value.get("bytes") or value.get("path")
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                from io import BytesIO
                from PIL import Image
                image = Image.open(BytesIO(value)).convert("RGB")
                return image if MMSTDatasetLoader._is_usable_image(image) else None
            except Exception:
                return None
        if hasattr(value, "convert"):
            try:
                image = value.convert("RGB")
            except Exception:
                image = value
            if not MMSTDatasetLoader._is_usable_image(image):
                return None
            return image
        return value

    @staticmethod
    def _is_usable_image(image: Any) -> bool:
        size = getattr(image, "size", None)
        if not size or len(size) != 2:
            return True
        width, height = [int(x) for x in size]
        if width < 16 or height < 16:
            return False
        ratio = max(width / max(1, height), height / max(1, width))
        return ratio <= 20
    
    def load_from_json(self, json_path: str):
        """从 JSON 文件加载"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.data = []
            for item in data:
                mmst_item = MMSTItem(
                    id=item.get("id", ""),
                    question=item.get("question", ""),
                    reference_answer=item.get("reference_answer", item.get("answer", "")),
                    image_path=item.get("image_path"),
                    image_paths=item.get("image_paths") or ([item.get("image_path")] if item.get("image_path") else []),
                    image_inputs=[],
                    metadata=item.get("metadata", {})
                )
                self.data.append(mmst_item)
            
            print(f"✓ Loaded {len(self.data)} samples from {json_path}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load from JSON: {e}")
    
    def get_items(self, limit: Optional[int] = None) -> List[MMSTItem]:
        """
        获取数据项
        
        Args:
            limit: 限制数量
            
        Returns:
            MMSTItem 列表
        """
        if self.hf_dataset is not None and (not self.data or (limit is not None and len(self.data) < limit)):
            self._convert_from_hf(limit=limit)
        if limit:
            return self.data[:limit]
        return self.data
    
    def get_by_id(self, item_id: str) -> Optional[MMSTItem]:
        """根据 ID 获取数据项"""
        for item in self.data:
            if item.id == item_id:
                return item
        return None
    
    def get_batch(self, batch_size: int = 32, offset: int = 0) -> List[MMSTItem]:
        """
        获取一批数据
        
        Args:
            batch_size: 批大小
            offset: 偏移量
            
        Returns:
            MMSTItem 列表
        """
        end = min(offset + batch_size, len(self.data))
        if self.hf_dataset is not None:
            self._convert_from_hf(limit=batch_size, offset=offset)
            return self.data
        return self.data[offset:end]
    
    def stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            "total_items": len(self.hf_dataset) if self.hf_dataset is not None else len(self.data),
            "materialized_items": len(self.data),
            "split": self.split,
            "items_with_images": sum(1 for item in self.data if item.image_path),
            "items_without_images": sum(1 for item in self.data if not item.image_path),
            "sample_questions": [item.question for item in self.data[:3]]
        }
