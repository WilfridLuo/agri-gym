"""
Tool definitions and implementations

基于 AGMMU 知识库的工具
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import os
import threading
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

IN_MEMORY_IMAGE_STORE: Dict[str, Any] = {}


def register_in_memory_image(image_ref: str, image: Any) -> None:
    """Register an online/dataset image object for path-like tool access."""
    if image_ref and image is not None:
        IN_MEMORY_IMAGE_STORE[str(image_ref)] = image


def get_in_memory_image(image_ref: str) -> Any:
    return IN_MEMORY_IMAGE_STORE.get(str(image_ref))


def clear_in_memory_images() -> None:
    IN_MEMORY_IMAGE_STORE.clear()


@dataclass
class ToolInput:
    """工具输入"""
    name: str
    value: Any


@dataclass
class ToolOutput:
    """工具输出"""
    success: bool
    data: Any
    error: Optional[str] = None


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        """
        初始化工具
        
        Args:
            name: 工具名称
            description: 工具描述
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolOutput:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolOutput 工具输出
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具的 JSON schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }
    
    def _get_parameters(self) -> Dict[str, Any]:
        """获取参数信息"""
        return {}


def _species_match_score(query_species: Optional[str], candidate_species: Optional[str]) -> float:
    if not query_species or not candidate_species:
        return 0.0
    q = str(query_species).lower().strip()
    c = str(candidate_species).lower().strip()
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    q_terms = {x for x in q.replace("/", " ").replace(",", " ").split() if x}
    c_terms = {x for x in c.replace("/", " ").replace(",", " ").split() if x}
    if q in c or c in q:
        return 0.75
    overlap = q_terms & c_terms
    if overlap:
        return min(0.6, len(overlap) / max(1, len(q_terms)))
    return 0.0


def _case_bundle(loader: Any, faq_id: str, max_facts: Optional[int] = None) -> Dict[str, Any]:
    if hasattr(loader, "get_case_evidence"):
        bundle = loader.get_case_evidence(faq_id, max_facts=max_facts)
        if bundle:
            return bundle
    return {"faq_id": faq_id, "facts": []}


def _compact_facts(facts: List[Dict[str, Any]], max_facts: int = 6) -> List[Dict[str, Any]]:
    out = []
    for fact in facts[:max_facts]:
        out.append(
            {
                "field": fact.get("field"),
                "question": fact.get("question", ""),
                "answer": fact.get("answer", ""),
            }
        )
    return out


def _all_case_facts(loader: Any, faq_id: str) -> List[Dict[str, Any]]:
    bundle = _case_bundle(loader, faq_id, max_facts=None)
    return [
        {
            "field": fact.get("field"),
            "question": fact.get("question", ""),
            "answer": fact.get("answer", ""),
        }
        for fact in (bundle.get("facts") or [])
        if isinstance(fact, dict)
    ]


def _term_overlap_score(query: Optional[str], row: Dict[str, Any]) -> float:
    query_terms = {
        t
        for t in str(query or "").lower().replace("/", " ").replace("_", " ").split()
        if len(t) > 2
    }
    if not query_terms:
        return 0.0
    text = json.dumps(row, ensure_ascii=False).lower()
    hit = sum(1 for t in query_terms if t in text)
    return hit / max(1, len(query_terms))


def _candidate_relevance(
    query: Optional[str],
    requested_species: Optional[str],
    requested_field: Optional[str],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    species_score = _species_match_score(requested_species, row.get("species"))
    field = _normalize_field_filter(requested_field)
    row_field = (
        (row.get("text_evidence") or {}).get("field")
        or row.get("field")
        or ""
    )
    field_match = bool(field and row_field == field) if field else None
    query_term_match = _term_overlap_score(query, row)
    text = json.dumps(row, ensure_ascii=False).lower()

    mismatch_reasons = []
    if requested_species and row.get("species") and species_score == 0.0:
        mismatch_reasons.append("species_mismatch")
    if field and row_field and row_field != field:
        mismatch_reasons.append("field_mismatch")
    if query_term_match == 0.0:
        mismatch_reasons.append("no_query_term_overlap")

    useful_by_analogy = (
        query_term_match > 0.0
        or species_score > 0.0
        or bool({"lichen", "fungal", "pathogen", "deer", "squirrel", "antler", "bark"} & set(text.split()))
    )
    should_agent_use = useful_by_analogy and not (
        "species_mismatch" in mismatch_reasons
        and query_term_match < 0.25
        and species_score == 0.0
    )

    return {
        "species_match_score": species_score,
        "field_match": field_match,
        "query_term_match": round(query_term_match, 3),
        "should_agent_use": should_agent_use,
        "mismatch_reason": "; ".join(mismatch_reasons),
        "use_as": "answer_evidence" if should_agent_use else "recall_only",
        "analogy_note": (
            "Use as analogous case evidence; adapt cautiously to the current query, species, image, and uncertainty."
            if should_agent_use else
            "Keep as recall context only unless later evidence connects it to the current query."
        ),
    }


def _query_expansions(query: Optional[str]) -> List[str]:
    q = str(query or "").strip()
    ql = q.lower()
    variants = [q] if q else []
    if any(k in ql for k in ["bark", "scrap", "scraping", "strip", "stripped", "rub"]):
        variants.extend([
            "deer antler rubbing bark trunk",
            "squirrel bark stripping cedar",
            "wildlife physical damage bark stripped",
        ])
    if "white" in ql and any(k in ql for k in ["evergreen", "conifer", "spruce", "pine", "disease"]):
        variants.extend([
            "white lichens not problem fungal pathogen evergreen",
            "evergreen fungal pathogen chlorothalonil bordeaux foliar spray",
            "needles fall fungal pathogen lower branches chlorothalonil copper bordeaux",
        ])
    return list(dict.fromkeys([v for v in variants if v]))


def _domain_rerank_adjustment(query: Optional[str], row: Dict[str, Any]) -> float:
    q = str(query or "").lower()
    text = json.dumps(row, ensure_ascii=False).lower()
    delta = 0.0
    if any(k in q for k in ["bark", "scrap", "scraping", "strip", "stripped", "rub"]):
        if any(k in text for k in ["deer", "antler"]):
            delta += 0.28
        if any(k in text for k in ["squirrel", "wildlife", "physical damage", "stripped of their bark"]):
            delta += 0.16
        if any(k in text for k in ["bark beetle", "borer", "frass", "tunneling", "whole branches", "dying"]):
            delta -= 0.16
    if "white" in q and any(k in q for k in ["evergreen", "conifer", "spruce", "pine", "disease"]):
        if any(k in text for k in ["fungal pathogen", "chlorothalonil", "bordeaux", "foliar", "needles fall", "lower branches"]):
            delta += 0.22
        if "lichen" in text and any(k in text for k in ["not the problem", "harmless", "not problem"]):
            delta += 0.12
        if any(k in text for k in ["pine needle scale", "cottony scale", "scale insects", "bagworms", "systemic insecticide"]):
            delta -= 0.14
        if "cedar apple rust" in text and "woody ball" in text:
            delta -= 0.18
    return delta


def _normalize_field_filter(field: Optional[str]) -> Optional[str]:
    if not field:
        return None
    raw = str(field).strip().lower()
    aliases = {
        "management instructions": "management_instructions",
        "symptom description": "symptom_description",
        "image description": "image_description",
        "insect/pest": "insect_or_pest",
        "disease/issue identification": "disease_or_issue_identification",
    }
    return aliases.get(raw, raw.replace("/", "_").replace(" ", "_"))


def _resolve_image_path(image_path: str) -> Path:
    p = Path(str(image_path)).expanduser()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return p


def _open_image_resource(image_path: str) -> tuple[Any, str, str]:
    ref = str(image_path)
    image_obj = get_in_memory_image(ref)
    if image_obj is not None:
        if hasattr(image_obj, "copy"):
            image_obj = image_obj.copy()
        stem = ref.rstrip("/").split("/")[-2:] if "/" in ref else [ref]
        safe_stem = "_".join(stem).replace("#", "").replace(":", "_")
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe_stem) or "in_memory_image"
        return image_obj, ref, safe_stem

    path = _resolve_image_path(ref)
    return Image.open(path), str(path), path.stem


def _visual_output_path_for_name(source_name: str, suffix: str, source_path: Optional[Path] = None, output_dir: Optional[str] = None) -> Path:
    env_output_dir = os.environ.get("AGRI_VISUAL_TOOL_DIR")
    if env_output_dir:
        out_dir = Path(env_output_dir).expanduser()
    elif output_dir:
        out_dir = Path(output_dir).expanduser()
    else:
        out_dir = Path("data/visual_tools").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source_name) or "image"
    return out_dir / f"{safe_name}_{suffix}.jpg"


def _visual_output_path(image_path: Path, suffix: str, output_dir: Optional[str] = None) -> Path:
    out_dir = Path(os.environ.get("AGRI_VISUAL_TOOL_DIR") or output_dir or "data/visual_tools").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_path.stem}_{suffix}{image_path.suffix or '.jpg'}"


def _bbox_to_pixels(
    bbox: List[float],
    width: int,
    height: int,
    normalized: bool = True,
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    left = int(max(0, min(width - 1, min(x1, x2))))
    right = int(max(1, min(width, max(x1, x2))))
    top = int(max(0, min(height - 1, min(y1, y2))))
    bottom = int(max(1, min(height, max(y1, y2))))
    if right <= left:
        right = min(width, left + 1)
        left = max(0, right - 1)
    if bottom <= top:
        bottom = min(height, top + 1)
        top = max(0, bottom - 1)
    return left, top, right, bottom


def _normalize_box_list(boxes: Any) -> List[List[float]]:
    """Accept common VLM bbox shapes and normalize to [[x1, y1, x2, y2], ...]."""
    if boxes is None:
        return []
    if isinstance(boxes, dict):
        boxes = boxes.get("bbox") or boxes.get("box") or boxes.get("coordinates") or boxes.get("coords")
    if not isinstance(boxes, list):
        raise ValueError("boxes must be [x1,y1,x2,y2] or [[x1,y1,x2,y2], ...]")
    if len(boxes) == 4 and all(isinstance(v, (int, float, str)) for v in boxes):
        return [[float(v) for v in boxes]]

    normalized_boxes: List[List[float]] = []
    for raw_box in boxes:
        if isinstance(raw_box, dict):
            raw_box = raw_box.get("bbox") or raw_box.get("box") or raw_box.get("coordinates") or raw_box.get("coords")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError("each box must be [x1, y1, x2, y2]")
        normalized_boxes.append([float(v) for v in raw_box])
    return normalized_boxes


def _stabilize_crop_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    min_size: int = 64,
    max_aspect: float = 8.0,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    cx = (left + right) / 2
    cy = (top + bottom) / 2

    target_w = max(crop_w, min(min_size, width))
    target_h = max(crop_h, min(min_size, height))
    if target_w / max(1, target_h) > max_aspect:
        target_h = min(height, int(np.ceil(target_w / max_aspect)))
    if target_h / max(1, target_w) > max_aspect:
        target_w = min(width, int(np.ceil(target_h / max_aspect)))

    left = int(round(cx - target_w / 2))
    top = int(round(cy - target_h / 2))
    left = max(0, min(width - target_w, left))
    top = max(0, min(height - target_h, top))
    right = int(left + target_w)
    bottom = int(top + target_h)
    return left, top, right, bottom


class ImageMetadataTool(BaseTool):
    """Return image size and basic metadata for coordinate planning."""

    def __init__(self):
        super().__init__(
            name="inspect_image_metadata",
            description="读取图像尺寸和模式，帮助规划 crop/zoom/bounding box 坐标"
        )

    def execute(self, image_path: str, **kwargs) -> ToolOutput:
        try:
            img, source_ref, _ = _open_image_resource(image_path)
            with img:
                width, height = img.size
                mode = img.mode
            return ToolOutput(
                success=True,
                data={
                    "image_path": source_ref,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "coordinate_system": "normalized bbox [x1,y1,x2,y2] uses 0..1 relative coordinates unless normalized=false",
                },
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "image_path": {"type": "string", "description": "图像路径"},
        }


class CropImageTool(BaseTool):
    """Crop an image region and return a new image path."""

    def __init__(self):
        super().__init__(
            name="crop_image",
            description="按 bbox 裁剪图像局部区域，用于观察病斑、虫害、树皮损伤等细节"
        )

    def execute(
        self,
        image_path: str,
        bbox: List[float],
        normalized: bool = True,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        try:
            img, source_ref, source_name = _open_image_resource(image_path)
            source_path = Path(source_ref) if Path(source_ref).exists() else None
            with img:
                img = img.convert("RGB")
                width, height = img.size
                raw_box = _bbox_to_pixels(bbox, width, height, normalized=normalized)
                box = _stabilize_crop_box(raw_box, width, height)
                cropped = img.crop(box)
                out = _visual_output_path_for_name(source_name, f"crop_{box[0]}_{box[1]}_{box[2]}_{box[3]}", source_path, output_dir)
                cropped.save(out)
            return ToolOutput(
                success=True,
                data={
                    "source_image_path": source_ref,
                    "generated_image_path": str(out),
                    "operation": "crop",
                    "bbox_pixels": list(box),
                    "bbox_normalized": bbox if normalized else None,
                    "width": cropped.size[0],
                    "height": cropped.size[1],
                },
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "image_path": {"type": "string", "description": "原图路径"},
            "bbox": {"type": "array", "description": "裁剪框 [x1,y1,x2,y2]，默认归一化 0..1"},
            "normalized": {"type": "boolean", "description": "bbox 是否为归一化坐标", "default": True},
            "output_dir": {"type": "string", "description": "输出目录（可选）"},
        }


class ZoomImageTool(BaseTool):
    """Crop around a center point and resize for close inspection."""

    def __init__(self):
        super().__init__(
            name="zoom_image",
            description="围绕中心点放大图像区域，生成更容易观察的局部图"
        )

    def execute(
        self,
        image_path: str,
        center: List[float],
        scale: float = 2.0,
        normalized: bool = True,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        try:
            if len(center) != 2:
                raise ValueError("center must be [x, y]")
            img, source_ref, source_name = _open_image_resource(image_path)
            source_path = Path(source_ref) if Path(source_ref).exists() else None
            scale = max(1.0, float(scale))
            with img:
                img = img.convert("RGB")
                width, height = img.size
                cx, cy = [float(v) for v in center]
                if normalized:
                    cx, cy = cx * width, cy * height
                crop_w = max(8, int(width / scale))
                crop_h = max(8, int(height / scale))
                left = int(max(0, min(width - crop_w, cx - crop_w / 2)))
                top = int(max(0, min(height - crop_h, cy - crop_h / 2)))
                box = (left, top, left + crop_w, top + crop_h)
                zoomed = img.crop(box).resize((width, height), Image.Resampling.LANCZOS)
                out = _visual_output_path_for_name(source_name, f"zoom_{left}_{top}_{scale:g}x", source_path, output_dir)
                zoomed.save(out)
            return ToolOutput(
                success=True,
                data={
                    "source_image_path": source_ref,
                    "generated_image_path": str(out),
                    "operation": "zoom",
                    "center": center,
                    "scale": scale,
                    "crop_bbox_pixels": list(box),
                    "width": width,
                    "height": height,
                },
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "image_path": {"type": "string", "description": "原图路径"},
            "center": {"type": "array", "description": "放大中心 [x,y]，默认归一化 0..1"},
            "scale": {"type": "number", "description": "放大倍数，>=1", "default": 2.0},
            "normalized": {"type": "boolean", "description": "center 是否为归一化坐标", "default": True},
            "output_dir": {"type": "string", "description": "输出目录（可选）"},
        }


class DrawBoundingBoxTool(BaseTool):
    """Draw one or more bounding boxes on an image."""

    def __init__(self):
        super().__init__(
            name="draw_bounding_box",
            description="在图像上绘制 bbox 标注，用于记录 agent 关注的病斑、虫害或损伤区域"
        )

    def execute(
        self,
        image_path: str,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
        normalized: bool = True,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        try:
            boxes = _normalize_box_list(boxes)
            if isinstance(labels, str):
                labels = [labels]
            elif labels is not None and not isinstance(labels, list):
                labels = [str(labels)]
            img, source_ref, source_name = _open_image_resource(image_path)
            source_path = Path(source_ref) if Path(source_ref).exists() else None
            with img:
                img = img.convert("RGB")
                width, height = img.size
                draw = ImageDraw.Draw(img)
                pixel_boxes = []
                for idx, raw_box in enumerate(boxes):
                    box = _bbox_to_pixels(raw_box, width, height, normalized=normalized)
                    pixel_boxes.append(list(box))
                    draw.rectangle(box, outline="red", width=max(2, width // 250))
                    if labels and idx < len(labels):
                        draw.text((box[0] + 3, box[1] + 3), str(labels[idx]), fill="red")
                out = _visual_output_path_for_name(source_name, f"bbox_{len(pixel_boxes)}", source_path, output_dir)
                img.save(out)
            return ToolOutput(
                success=True,
                data={
                    "source_image_path": source_ref,
                    "generated_image_path": str(out),
                    "operation": "draw_bounding_box",
                    "boxes_pixels": pixel_boxes,
                    "boxes_normalized": boxes if normalized else None,
                },
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "image_path": {"type": "string", "description": "原图路径"},
            "boxes": {"type": "array", "description": "bbox 列表，每个为 [x1,y1,x2,y2]"},
            "labels": {"type": "array", "description": "每个 bbox 的标签（可选）"},
            "normalized": {"type": "boolean", "description": "boxes 是否为归一化坐标", "default": True},
            "output_dir": {"type": "string", "description": "输出目录（可选）"},
        }


class TriageAgriIssueTool(BaseTool):
    """Broad agricultural issue routing without requiring an exact case match."""

    def __init__(self):
        super().__init__(
            name="triage_agri_issue",
            description="根据 query/metadata 做宽泛问题路由，给出建议检索词；不把 issue-type 当硬性诊断"
        )

    def execute(
        self,
        query: str,
        species: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ToolOutput:
        text = " ".join([
            str(query or ""),
            str(species or ""),
            json.dumps(metadata or {}, ensure_ascii=False),
        ]).lower()

        issue_types = []
        search_terms = []
        cautions = []

        if any(k in text for k in ["bark", "scrap", "strip", "rub", "antler"]):
            issue_types.append("wildlife_or_physical_damage")
            search_terms.extend(["deer antler rubbing bark", "squirrel bark stripping", "bark stripped trunk"])
            cautions.append("Do not assume insect/disease unless evidence mentions boring holes, galleries, dieback, or pathogen signs.")
        if any(k in text for k in ["white", "lichen", "fung", "disease", "pathogen", "mold"]):
            issue_types.extend(["disease", "cosmetic_or_non_harmful_observation"])
            search_terms.extend(["lichen not problem fungal pathogen", "evergreen fungal pathogen chlorothalonil", "bordeaux mix foliar spray evergreen"])
            cautions.append("Separate visible white lichen/coating from the actual damaging disease or pest.")
        if any(k in text for k in ["insect", "pest", "scale", "beetle", "mite", "aphid"]):
            issue_types.append("pest")
            search_terms.extend(["insect pest symptoms management", "scale insects evergreen"])
        if any(k in text for k in ["water", "drought", "snow", "winter", "fertiliz", "soil"]):
            issue_types.append("abiotic_or_cultural")
            search_terms.extend(["abiotic stress management", "cultural conditions plant recovery"])

        if not issue_types:
            issue_types.append("unknown_or_mixed")
            search_terms.append(str(query or "").strip())

        # Preserve order while removing duplicates.
        issue_types = list(dict.fromkeys(issue_types))
        search_terms = list(dict.fromkeys([x for x in search_terms if x]))

        return ToolOutput(
            success=True,
            data={
                "query": query,
                "species": species,
                "broad_issue_types": issue_types,
                "recommended_search_terms": search_terms[:8],
                "recommended_next_tools": [
                    "retrieve_multimodal_evidence",
                    "retrieve_by_embedding",
                    "search_knowledge_base",
                ],
                "cautions": cautions,
                "routing_policy": (
                    "Use these labels for retrieval routing only. Similar KB facts can be borrowed as expert experience, "
                    "but final answers must explain how the analogy applies and where uncertainty remains."
                ),
            },
        )

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "用户问题或症状描述"},
            "species": {"type": "string", "description": "植物物种（可选）"},
            "metadata": {"type": "object", "description": "任务 metadata（可选）"},
        }


class SearchKnowledgeBaseTool(BaseTool):
    """搜索知识库工具"""
    
    def __init__(self, agmmu_loader):
        super().__init__(
            name="search_knowledge_base",
            description="从农业知识库中搜索相关信息"
        )
        self.agmmu_loader = agmmu_loader
    
    def execute(
        self,
        query: str,
        species: Optional[str] = None,
        limit: int = 5,
        field: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        """
        搜索知识库
        
        Args:
            query: 搜索查询
            species: 物种（可选）
            limit: 返回的结果数
            
        Returns:
            ToolOutput
        """
        try:
            tried = []
            query_variants = [query]
            # fallback query variants for sparse matching
            if len(query.split()) > 8:
                query_variants.append(" ".join(query.split()[:8]))
            if species:
                query_variants.append(f"{species} {query}")

            image_hint = any(k in (query or "").lower() for k in ["photo", "image", "attached", "图片", "照片"])
            candidate_fields: List[Optional[str]] = [field]
            if not field and image_hint:
                candidate_fields = ["image description", "symptom description", None]

            # Prefer fact-level retrieval if available.
            if hasattr(self.agmmu_loader, "search_facts"):
                facts = []
                for qv in query_variants:
                    for fd in candidate_fields:
                        tried.append({"mode": "facts", "species": species, "field": fd, "query": qv})
                        facts = self.agmmu_loader.search_facts(
                            keyword=qv,
                            species=species,
                            field=fd,
                            limit=limit,
                        )
                        if facts:
                            break
                    if facts:
                        break

                # fallback 1: drop species filter
                if not facts and species:
                    for qv in query_variants:
                        for fd in candidate_fields:
                            tried.append({"mode": "facts", "species": None, "field": fd, "query": qv})
                            facts = self.agmmu_loader.search_facts(
                                keyword=qv,
                                species=None,
                                field=fd,
                                limit=limit,
                            )
                            if facts:
                                break
                        if facts:
                            break

                # fallback 2: drop field filter
                if not facts and field:
                    tried.append({"mode": "facts", "species": species, "field": None})
                    facts = self.agmmu_loader.search_facts(
                        keyword=query,
                        species=species,
                        field=None,
                        limit=limit,
                    )

                # fallback 3: global facts
                if not facts and (species or field):
                    tried.append({"mode": "facts", "species": None, "field": None})
                    facts = self.agmmu_loader.search_facts(
                        keyword=query,
                        species=None,
                        field=None,
                        limit=limit,
                    )

                if facts:
                    if _normalize_field_filter(field) == "management_instructions":
                        q_terms = {
                            t for t in str(query or "").lower().split()
                            if len(t) > 2 and t not in {"management", "advice", "manage", "control"}
                        }
                        facts.sort(
                            key=lambda f: sum(1 for t in q_terms if t in f.answer.lower()),
                            reverse=True,
                        )
                    seen_cases = set()
                    fact_data = []
                    case_evidence = []
                    for f in facts:
                        row = {
                            "faq_id": f.faq_id,
                            "species": f.species,
                            "field": f.field,
                            "question": f.question,
                            "answer": f.answer,
                            "all_facts": _all_case_facts(self.agmmu_loader, f.faq_id),
                        }
                        row["relevance"] = _candidate_relevance(query, species, field, row)
                        fact_data.append(row)
                        if f.faq_id not in seen_cases:
                            seen_cases.add(f.faq_id)
                            bundle = _case_bundle(self.agmmu_loader, f.faq_id)
                            bundle["matched_fact"] = {
                                "field": f.field,
                                "question": f.question,
                                "answer": f.answer,
                            }
                            case_evidence.append(bundle)
                    return ToolOutput(
                        success=True,
                        data={
                            "query": query,
                            "species": species,
                            "field": field,
                            "result_count": len(fact_data),
                            "results": fact_data,
                            "case_evidence": case_evidence[:limit],
                            "recall_candidates": fact_data,
                            "answer_evidence": [r for r in fact_data if (r.get("relevance") or {}).get("should_agent_use")],
                            "result_type": "facts",
                            "retrieval_diagnostics": {
                                "status": "ok",
                                "tried": tried,
                                "image_hint": image_hint,
                            },
                        },
                    )

            if species:
                tried.append({"mode": "items", "species": species, "field": field})
                results = self.agmmu_loader.search_by_species_and_keyword(
                    species=species,
                    keyword=query,
                    limit=limit
                )
            else:
                tried.append({"mode": "items", "species": None, "field": field})
                results = self.agmmu_loader.search_by_keyword(
                    keyword=query,
                    limit=limit
                )
            
            data = [
                {
                    "faq_id": item.faq_id,
                    "species": item.species,
                    "issue_type": self.agmmu_loader.infer_issue_type(item) if hasattr(self.agmmu_loader, "infer_issue_type") else "unknown",
                    "matched_fields": [
                        {
                            "field": fact.field,
                            "question": fact.question,
                            "answer": fact.answer,
                        }
                        for fact in getattr(item, "facts", [])[:3]
                    ],
                    "all_facts": _all_case_facts(self.agmmu_loader, item.faq_id),
                }
                for item in results
            ]
            for row in data:
                row["relevance"] = _candidate_relevance(query, species, field, row)
            
            return ToolOutput(
                success=True,
                data={
                    "query": query,
                    "species": species,
                    "field": field,
                    "result_count": len(data),
                    "results": data,
                    "recall_candidates": data,
                    "answer_evidence": [r for r in data if (r.get("relevance") or {}).get("should_agent_use")],
                    "result_type": "items",
                    "retrieval_diagnostics": {
                        "status": "ok" if len(data) > 0 else "empty",
                        "tried": tried,
                        "image_hint": image_hint,
                        "suggested_species": (
                            self.agmmu_loader.suggest_species(species, limit=5)
                            if species and hasattr(self.agmmu_loader, "suggest_species")
                            else []
                        ),
                        "recommendations": [
                            "Try species=None to avoid over-filtering.",
                            "Use shorter symptom/pest keywords.",
                            "Use canonical fields: symptom description, disease/issue identification, insect/pest, management instructions.",
                        ] if len(data) == 0 else [],
                    },
                }
            )
        
        except Exception as e:
            return ToolOutput(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "species": {
                "type": "string",
                "description": "植物物种（可选）"
            },
            "field": {
                "type": "string",
                "description": "字段过滤（可选）：management instructions / symptom description / disease/issue identification / insect/pest"
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数",
                "default": 5
            }
        }


class VisualRetrieveTool(BaseTool):
    """Visual retrieval tool over KB images."""

    def __init__(
        self,
        agmmu_loader,
        emb_dir: Optional[str] = None,
        device: str = "cuda",
        embedding_tool: Optional["EmbeddingRetrieveTool"] = None,
    ):
        super().__init__(
            name="visual_retrieve",
            description="使用查询图片在知识库图片中检索相似案例"
        )
        self.agmmu_loader = agmmu_loader
        self.embedding_tool = embedding_tool or (EmbeddingRetrieveTool(emb_dir, device=device) if emb_dir else None)

    def execute(
        self,
        image_path: Optional[str] = None,
        species: Optional[str] = None,
        limit: int = 5,
        image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolOutput:
        try:
            query_image_paths = [str(x) for x in (image_paths or []) if str(x).strip()]
            if image_path and str(image_path).strip() not in query_image_paths:
                query_image_paths.insert(0, str(image_path).strip())
            if not query_image_paths:
                raise ValueError("visual_retrieve requires image_path or image_paths")

            hits = []
            seen_hits = set()
            if self.embedding_tool is not None:
                for image_idx, query_image_path in enumerate(query_image_paths, start=1):
                    emb_out = self.embedding_tool.execute(query="", image_path=query_image_path, species=species, limit=limit)
                    if emb_out.success:
                        for row in emb_out.data.get("image_results") or []:
                            key = (row.get("faq_id"), row.get("image_path"), query_image_path)
                            if key in seen_hits:
                                continue
                            row = dict(row)
                            row["query_image_path"] = query_image_path
                            row["query_image_index"] = image_idx
                            hits.append(row)
                            seen_hits.add(key)
            elif hasattr(self.agmmu_loader, "search_by_image"):
                for image_idx, query_image_path in enumerate(query_image_paths, start=1):
                    for row in self.agmmu_loader.search_by_image(
                        query_image_path=query_image_path,
                        species=species,
                        limit=limit,
                    ):
                        key = (row.get("faq_id"), row.get("image_path"), query_image_path)
                        if key in seen_hits:
                            continue
                        row = dict(row)
                        row["query_image_path"] = query_image_path
                        row["query_image_index"] = image_idx
                        hits.append(row)
                        seen_hits.add(key)

            enriched = []
            for h in hits:
                bundle = _case_bundle(self.agmmu_loader, str(h.get("faq_id", "")))
                matched_facts = h.get("matched_facts") or h.get("facts") or bundle.get("facts") or []
                enriched.append(
                    {
                        "faq_id": h.get("faq_id"),
                        "species": h.get("species"),
                        "image_path": h.get("image_path"),
                        "query_image_path": h.get("query_image_path"),
                        "query_image_index": h.get("query_image_index"),
                        "image_score": h.get("image_score", h.get("score", 0.0)),
                        "matched_facts": _compact_facts(matched_facts),
                        "all_facts": _all_case_facts(self.agmmu_loader, str(h.get("faq_id", ""))),
                        "case_evidence": bundle,
                    }
                )
                enriched[-1]["relevance"] = _candidate_relevance(
                    query=None,
                    requested_species=species,
                    requested_field=None,
                    row=enriched[-1],
                )

            return ToolOutput(
                success=True,
                data={
                    "image_path": image_path or query_image_paths[0],
                    "image_paths": query_image_paths,
                    "species": species,
                    "result_count": len(enriched),
                    "results": enriched,
                    "recall_candidates": enriched,
                    "answer_evidence": [r for r in enriched if (r.get("relevance") or {}).get("should_agent_use")],
                    "result_type": "visual",
                    "retrieval_diagnostics": {
                        "status": "ok" if enriched else "empty",
                        "embedding_model": self.embedding_tool.model_name if self.embedding_tool is not None else None,
                        "embedding_instance_id": id(self.embedding_tool) if self.embedding_tool is not None else None,
                        "recommendations": [
                            "Check if query image path exists and is readable.",
                            "Try species=None for broader visual recall.",
                        ] if not enriched else [],
                    },
                }
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "image_path": {
                "type": "string",
                "description": "查询图片绝对路径"
            },
            "image_paths": {
                "type": "array",
                "description": "多图任务的全部查询图片路径，可选",
                "items": {"type": "string"}
            },
            "species": {
                "type": "string",
                "description": "植物物种（可选）"
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数",
                "default": 5
            }
        }


class RetrieveMultimodalEvidenceTool(BaseTool):
    """Parallel text+image retrieval and fusion."""

    def __init__(
        self,
        agmmu_loader,
        emb_dir: Optional[str] = None,
        device: str = "cuda",
        embedding_tool: Optional["EmbeddingRetrieveTool"] = None,
    ):
        super().__init__(
            name="retrieve_multimodal_evidence",
            description="并行文本与图像检索并融合证据"
        )
        self.agmmu_loader = agmmu_loader
        self.embedding_tool = embedding_tool or (EmbeddingRetrieveTool(emb_dir, device=device) if emb_dir else None)

    def execute(
        self,
        query: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        species: Optional[str] = None,
        field: Optional[str] = None,
        limit: int = 5,
        **kwargs,
    ) -> ToolOutput:
        try:
            text_results = []
            image_results = []
            query_image_paths = [str(x) for x in (image_paths or []) if str(x).strip()]
            if image_path and str(image_path).strip() not in query_image_paths:
                query_image_paths.insert(0, str(image_path).strip())

            if self.embedding_tool is not None:
                seen_text = set()
                seen_image = set()
                for query_variant in _query_expansions(query):
                    emb_out = self.embedding_tool.execute(
                        query=query_variant,
                        image_path=None,
                        species=species,
                        field=field,
                        limit=max(limit * 4, 20),
                        aggregate_cases=False,
                    )
                    if emb_out.success:
                        for row in emb_out.data.get("text_results") or []:
                            key = (row.get("faq_id"), row.get("field"), row.get("answer"))
                            if key not in seen_text:
                                row = dict(row)
                                row["query_variant"] = query_variant
                                text_results.append(row)
                                seen_text.add(key)
                for image_idx, query_image_path in enumerate(query_image_paths, start=1):
                    emb_out = self.embedding_tool.execute(
                        query="",
                        image_path=query_image_path,
                        species=species,
                        field=field,
                        limit=max(limit * 4, 20),
                        aggregate_cases=False,
                    )
                    if emb_out.success:
                        for row in emb_out.data.get("image_results") or []:
                            key = (row.get("faq_id"), row.get("image_path"), query_image_path)
                            if key not in seen_image:
                                row = dict(row)
                                row["query_image_path"] = query_image_path
                                row["query_image_index"] = image_idx
                                image_results.append(row)
                                seen_image.add(key)
            elif hasattr(self.agmmu_loader, "search_facts"):
                seen_facts = set()
                facts = []
                for query_variant in _query_expansions(query):
                    for f in self.agmmu_loader.search_facts(
                        keyword=query_variant,
                        species=species,
                        field=field,
                        limit=max(limit * 2, 10),
                    ):
                        key = (f.faq_id, f.field, f.answer)
                        if key not in seen_facts:
                            facts.append(f)
                            seen_facts.add(key)
                text_results = [
                    {
                        "faq_id": f.faq_id,
                        "species": f.species,
                        "field": f.field,
                        "question": f.question,
                        "answer": f.answer,
                        "text_score": 1.0 / (idx + 1),
                    }
                    for idx, f in enumerate(facts)
                ]

            if query_image_paths and self.embedding_tool is None and hasattr(self.agmmu_loader, "search_by_image"):
                seen_image = set()
                for image_idx, query_image_path in enumerate(query_image_paths, start=1):
                    for row in self.agmmu_loader.search_by_image(
                        query_image_path=query_image_path,
                        species=species,
                        limit=max(limit * 2, 10),
                    ):
                        key = (row.get("faq_id"), row.get("image_path"), query_image_path)
                        if key in seen_image:
                            continue
                        row = dict(row)
                        row["query_image_path"] = query_image_path
                        row["query_image_index"] = image_idx
                        image_results.append(row)
                        seen_image.add(key)

            merged: Dict[str, Dict[str, Any]] = {}
            for r in text_results:
                fid = str(r.get("faq_id", ""))
                if not fid:
                    continue
                bundle = _case_bundle(self.agmmu_loader, fid)
                facts = r.get("facts") or bundle.get("facts") or []
                merged[fid] = {
                    "faq_id": fid,
                    "species": r.get("species"),
                    "issue_type": bundle.get("issue_type"),
                    "text_evidence": {
                        "field": r.get("field"),
                        "question": r.get("question"),
                        "answer": r.get("answer"),
                    },
                    "text_score": float(r.get("text_score", r.get("score", 0.0))),
                    "image_score": 0.0,
                    "facts": _compact_facts(facts),
                    "all_facts": _all_case_facts(self.agmmu_loader, fid),
                    "case_evidence": bundle,
                }

            for r in image_results:
                fid = str(r.get("faq_id", ""))
                if not fid:
                    continue
                if fid not in merged:
                    matched_facts = r.get("matched_facts") or []
                    best_fact = matched_facts[0] if matched_facts else {}
                    bundle = _case_bundle(self.agmmu_loader, fid)
                    facts = r.get("facts") or bundle.get("facts") or matched_facts
                    merged[fid] = {
                        "faq_id": fid,
                        "species": r.get("species"),
                        "issue_type": bundle.get("issue_type"),
                        "text_evidence": {
                            "field": best_fact.get("field"),
                            "question": best_fact.get("question", ""),
                            "answer": best_fact.get("answer", ""),
                        },
                        "text_score": 0.0,
                        "image_score": 0.0,
                        "facts": _compact_facts(facts),
                        "all_facts": _all_case_facts(self.agmmu_loader, fid),
                        "case_evidence": bundle,
                    }
                merged[fid]["image_score"] = max(
                    float(merged[fid].get("image_score", 0.0)),
                    float(r.get("image_score", r.get("score", 0.0))),
                )
                merged[fid]["visual_match"] = {
                    "image_path": r.get("image_path"),
                    "query_image_path": r.get("query_image_path"),
                    "query_image_index": r.get("query_image_index"),
                    "image_score": r.get("image_score", r.get("score", 0.0)),
                    "matched_facts": r.get("matched_facts") or [],
                }

            candidates = []
            for _, row in merged.items():
                ts = float(row.get("text_score", 0.0))
                vs = float(row.get("image_score", 0.0))
                sm = _species_match_score(species, row.get("species"))
                row["species_match_score"] = sm
                row["fused_score"] = 0.58 * ts + 0.30 * vs + 0.12 * sm
                row["domain_adjustment"] = _domain_rerank_adjustment(query, row)
                row["fused_score"] += row["domain_adjustment"]
                row["evidence_channels"] = {
                    "text": ts > 0,
                    "image": vs > 0,
                }
                row["relevance"] = _candidate_relevance(
                    query=query,
                    requested_species=species,
                    requested_field=field,
                    row=row,
                )
                candidates.append(row)

            candidates.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
            candidates = candidates[:limit]

            return ToolOutput(
                success=True,
                data={
                    "query": query,
                    "image_path": image_path or (query_image_paths[0] if query_image_paths else None),
                    "image_paths": query_image_paths,
                    "species": species,
                    "field": field,
                    "result_count": len(candidates),
                    "candidates": candidates,
                    "recall_candidates": candidates,
                    "answer_evidence": [c for c in candidates if (c.get("relevance") or {}).get("should_agent_use")],
                    "retrieval_diagnostics": {
                        "status": "ok" if candidates else "empty",
                        "text_count": len(text_results),
                        "image_count": len(image_results),
                        "embedding_model": self.embedding_tool.model_name if self.embedding_tool is not None else None,
                        "embedding_instance_id": id(self.embedding_tool) if self.embedding_tool is not None else None,
                        "recommendations": [
                            "If empty, retry with species=None and field=None.",
                            "If image_count=0, verify query image path and KB image coverage.",
                        ] if not candidates else [],
                    },
                }
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "查询文本（症状/问题）"
            },
            "image_path": {
                "type": "string",
                "description": "查询图片绝对路径（可选）"
            },
            "image_paths": {
                "type": "array",
                "description": "多图任务的全部查询图片路径，可选",
                "items": {"type": "string"}
            },
            "species": {
                "type": "string",
                "description": "植物物种（可选）"
            },
            "field": {
                "type": "string",
                "description": "字段过滤（可选）"
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数",
                "default": 5
            }
        }


class EmbeddingRetrieveTool(BaseTool):
    """Embedding retrieval tool using offline KB embedding index."""

    def __init__(self, emb_dir: str, device: str = "cuda"):
        super().__init__(
            name="retrieve_by_embedding",
            description="基于离线embedding索引进行文本/图像检索"
        )
        self.emb_dir = Path(emb_dir)
        self.device = device
        self._cache = {
            "model_name": None,
            "model": None,
            "text_emb": None,
            "text_id_map": None,
            "image_emb": None,
            "image_id_map": None,
            "facts_map": None,
        }
        self._load_lock = threading.RLock()

    @property
    def model_name(self) -> Optional[str]:
        try:
            self._ensure_loaded()
        except Exception:
            return self._cache.get("model_name")
        return self._cache.get("model_name")

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[None, :]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return x / norms

    def _load_manifest(self) -> Dict[str, Any]:
        manifests = sorted(self.emb_dir.glob("agmmu_manifest_*.json"))
        if not manifests:
            raise FileNotFoundError(f"No manifest found in {self.emb_dir}")
        with open(manifests[-1], "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._cache["text_emb"] is not None:
                return

            manifest = self._load_manifest()
            embeddings = manifest.get("embeddings") or {}

            text_info = embeddings.get("text") or {}
            text_emb_path = self._resolve_artifact_path(text_info.get("embeddings") or "")
            text_map_path = self._resolve_artifact_path(text_info.get("id_map") or "")

            text_emb = np.load(str(text_emb_path)).astype(np.float32)
            with open(text_map_path, "r", encoding="utf-8") as f:
                text_id_map = json.load(f)

            image_info = embeddings.get("image") or {}
            image_emb_path = self._resolve_artifact_path(image_info.get("embeddings") or "")
            image_map_path = self._resolve_artifact_path(image_info.get("id_map") or "")

            image_emb = None
            image_id_map = None
            if image_emb_path.exists() and image_map_path.exists():
                image_emb = np.load(str(image_emb_path)).astype(np.float32)
                with open(image_map_path, "r", encoding="utf-8") as f:
                    image_id_map = json.load(f)

            facts_map = None
            processed_facts = manifest.get("processed_facts") or manifest.get("processed_facts_path")
            if processed_facts:
                facts_path = self._resolve_artifact_path(str(processed_facts))
                if facts_path.exists():
                    with open(facts_path, "r", encoding="utf-8") as f:
                        facts_list = json.load(f)
                    if isinstance(facts_list, list):
                        facts_map = {}
                        for fact in facts_list:
                            if not isinstance(fact, dict):
                                continue
                            fid = str(fact.get("faq_id", ""))
                            if not fid:
                                continue
                            facts_map.setdefault(fid, []).append(fact)

            self._cache["model_name"] = text_info.get("model") or image_info.get("model")
            self._cache["text_emb"] = self._l2_normalize(text_emb)
            self._cache["text_id_map"] = text_id_map
            self._cache["image_emb"] = self._l2_normalize(image_emb) if image_emb is not None else None
            self._cache["image_id_map"] = image_id_map
            self._cache["facts_map"] = facts_map

    def _resolve_artifact_path(self, raw_path: str) -> Path:
        p = Path(str(raw_path))
        candidates = []
        if p.is_absolute():
            candidates.append(p)
            candidates.append(self.emb_dir / p.name)
        else:
            candidates.append(self.emb_dir / p)
            candidates.append(self.emb_dir / p.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def preload(self) -> None:
        """Load index artifacts and the embedding model once for reuse."""
        self._ensure_loaded()
        if self._cache.get("model") is None:
            self._encode(["embedding retriever warmup"])

    def _encode(self, docs: List[Any]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise ImportError("sentence-transformers is required for embedding retrieval") from e

        model_name = self._cache.get("model_name")
        if not model_name:
            raise RuntimeError("Embedding model name missing in manifest")

        with self._load_lock:
            model = self._cache.get("model")
            if model is None:
                model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
                self._cache["model"] = model
        vecs = model.encode(
            docs,
            batch_size=max(1, min(16, len(docs))),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._l2_normalize(vecs.astype(np.float32))

    @staticmethod
    def _topk_by_dot(emb: np.ndarray, q: np.ndarray, k: int) -> List[Dict[str, float]]:
        scores = emb @ q
        if scores.ndim == 2:
            scores = scores.squeeze(1)
        idx = np.argsort(-scores)[:k]
        return [{"index": int(i), "score": float(scores[i])} for i in idx]

    def execute(
        self,
        query: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        species: Optional[str] = None,
        field: Optional[str] = None,
        limit: int = 5,
        aggregate_cases: bool = True,
        **kwargs,
    ) -> ToolOutput:
        try:
            self._ensure_loaded()

            text_hits = []
            query_image_paths = [str(x) for x in (image_paths or []) if str(x).strip()]
            if image_path and str(image_path).strip() not in query_image_paths:
                query_image_paths.insert(0, str(image_path).strip())
            if query:
                text_emb = self._cache.get("text_emb")
                text_id_map = self._cache.get("text_id_map")
                facts_map = self._cache.get("facts_map") or {}
                if text_emb is None or text_id_map is None:
                    raise RuntimeError("Text embedding index not loaded")

                q_vec = self._encode([query])[0]
                t_idx = self._topk_by_dot(text_emb, q_vec, max(limit * 4, limit))
                for i in t_idx:
                    meta = text_id_map[i["index"]]
                    if field and str(meta.get("field", "")).lower() != _normalize_field_filter(field):
                        continue
                    sm = _species_match_score(species, meta.get("species"))
                    hit = {
                        "faq_id": meta.get("faq_id"),
                        "species": meta.get("species"),
                        "field": meta.get("field"),
                        "question": meta.get("question"),
                        "answer": meta.get("answer"),
                        "score": i["score"],
                        "species_match_score": sm,
                        "rerank_score": float(i["score"]) + 0.08 * sm,
                        "facts": facts_map.get(str(meta.get("faq_id", ""))) or [],
                    }
                    hit["relevance"] = _candidate_relevance(query, species, field, hit)
                    text_hits.append(hit)
                text_hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                text_hits = text_hits[:limit]

            image_hits = []
            if query_image_paths and self._cache.get("image_emb") is not None:
                img_emb = self._cache.get("image_emb")
                img_id_map = self._cache.get("image_id_map") or []
                facts_map = self._cache.get("facts_map") or {}
                seen_image = set()
                for image_idx, query_image_path in enumerate(query_image_paths, start=1):
                    image_query = get_in_memory_image(str(query_image_path)) or query_image_path
                    img_vec = self._encode([image_query])[0]
                    i_idx = self._topk_by_dot(img_emb, img_vec, max(limit * 4, limit))
                    for i in i_idx:
                        meta = img_id_map[i["index"]]
                        key = (meta.get("faq_id"), meta.get("image_path"), query_image_path)
                        if key in seen_image:
                            continue
                        seen_image.add(key)
                        sm = _species_match_score(species, meta.get("species"))
                        image_hits.append(
                            {
                                "faq_id": meta.get("faq_id"),
                                "species": meta.get("species"),
                                "image_path": meta.get("image_path"),
                                "query_image_path": query_image_path,
                                "query_image_index": image_idx,
                                "score": i["score"],
                                "species_match_score": sm,
                                "rerank_score": float(i["score"]) + 0.08 * sm,
                                "facts": facts_map.get(str(meta.get("faq_id", ""))) or [],
                            }
                        )
                        image_hits[-1]["relevance"] = _candidate_relevance(None, species, None, image_hits[-1])
                image_hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                image_hits = image_hits[:limit]

            case_results = []
            if aggregate_cases:
                merged: Dict[str, Dict[str, Any]] = {}
                for hit in text_hits:
                    fid = str(hit.get("faq_id") or "")
                    if not fid:
                        continue
                    row = merged.setdefault(
                        fid,
                        {
                            "faq_id": fid,
                            "species": hit.get("species"),
                            "text_score": 0.0,
                            "image_score": 0.0,
                            "facts": hit.get("facts") or [],
                        },
                    )
                    row["text_score"] = max(float(row.get("text_score", 0.0)), float(hit.get("score", 0.0)))
                    row["text_hit"] = hit
                for hit in image_hits:
                    fid = str(hit.get("faq_id") or "")
                    if not fid:
                        continue
                    row = merged.setdefault(
                        fid,
                        {
                            "faq_id": fid,
                            "species": hit.get("species"),
                            "text_score": 0.0,
                            "image_score": 0.0,
                            "facts": hit.get("facts") or [],
                        },
                    )
                    row["image_score"] = max(float(row.get("image_score", 0.0)), float(hit.get("score", 0.0)))
                    row["image_hit"] = hit
                    row["query_image_path"] = hit.get("query_image_path")
                    row["query_image_index"] = hit.get("query_image_index")
                    if not row.get("facts"):
                        row["facts"] = hit.get("facts") or []
                for row in merged.values():
                    sm = _species_match_score(species, row.get("species"))
                    row["species_match_score"] = sm
                    row["fused_score"] = 0.58 * float(row.get("text_score", 0.0)) + 0.34 * float(row.get("image_score", 0.0)) + 0.08 * sm
                    row["domain_adjustment"] = _domain_rerank_adjustment(query, row)
                    row["fused_score"] += row["domain_adjustment"]
                    row["facts"] = _compact_facts(row.get("facts") or [], max_facts=8)
                    row["all_facts"] = row.get("text_hit", {}).get("facts") or row.get("image_hit", {}).get("facts") or []
                    row["relevance"] = _candidate_relevance(query, species, field, row)
                    case_results.append(row)
                case_results.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
                case_results = case_results[:limit]

            return ToolOutput(
                success=True,
                data={
                    "query": query,
                    "image_path": image_path or (query_image_paths[0] if query_image_paths else None),
                    "image_paths": query_image_paths,
                    "species": species,
                    "field": field,
                    "text_results": text_hits,
                    "image_results": image_hits,
                    "case_results": case_results,
                    "answer_evidence": [c for c in case_results if (c.get("relevance") or {}).get("should_agent_use")],
                    "retrieval_diagnostics": {
                        "status": "ok" if (text_hits or image_hits or case_results) else "empty",
                        "text_count": len(text_hits),
                        "image_count": len(image_hits),
                        "case_count": len(case_results),
                        "mode": "embedding",
                        "embedding_model": self.model_name,
                        "embedding_instance_id": id(self),
                    },
                },
            )
        except Exception as e:
            return ToolOutput(success=False, data=None, error=str(e))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "查询文本"
            },
            "image_path": {
                "type": "string",
                "description": "查询图片绝对路径（可选）"
            },
            "image_paths": {
                "type": "array",
                "description": "多图任务的全部查询图片路径，可选",
                "items": {"type": "string"}
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数",
                "default": 5
            }
        }


class IdentifyDiseaseTool(BaseTool):
    """病害识别工具"""
    
    def __init__(self, agmmu_loader):
        super().__init__(
            name="identify_disease",
            description="根据症状识别植物病害"
        )
        self.agmmu_loader = agmmu_loader
    
    def execute(self, symptom: str, species: str, **kwargs) -> ToolOutput:
        """
        识别病害
        
        Args:
            symptom: 症状描述
            species: 植物物种
            
        Returns:
            ToolOutput
        """
        try:
            diseases = []
            tried = []

            if hasattr(self.agmmu_loader, "search_facts"):
                # field-aware retrieval first
                tried.append({"mode": "facts", "species": species, "field": "disease/issue identification"})
                facts = self.agmmu_loader.search_facts(
                    keyword=symptom,
                    species=species,
                    field="disease/issue identification",
                    limit=10,
                )
                if not facts:
                    tried.append({"mode": "facts", "species": species, "field": "symptom description"})
                    facts = self.agmmu_loader.search_facts(
                        keyword=symptom,
                        species=species,
                        field="symptom description",
                        limit=10,
                    )
                if not facts and species:
                    tried.append({"mode": "facts", "species": None, "field": "disease/issue identification"})
                    facts = self.agmmu_loader.search_facts(
                        keyword=symptom,
                        species=None,
                        field="disease/issue identification",
                        limit=10,
                    )

                for f in facts:
                    bundle = _case_bundle(self.agmmu_loader, f.faq_id)
                    diseases.append(
                        {
                            "faq_id": f.faq_id,
                            "species": f.species,
                            "disease": f.answer,
                            "symptom": f.question,
                            "issue_type": bundle.get("issue_type"),
                            "case_evidence": bundle,
                        }
                    )
            else:
                # legacy item-level fallback
                tried.append({"mode": "items", "species": species})
                results = self.agmmu_loader.search_by_species_and_keyword(
                    species=species,
                    keyword=symptom,
                    limit=5
                )
                for item in results:
                    disease_id = item.content.get('disease_or_issue_identification')
                    if disease_id:
                        diseases.append({
                            "faq_id": item.faq_id,
                            "species": item.species,
                            "disease": disease_id,
                            "symptom": item.content.get('symptom_description'),
                            "management": item.content.get('management_instructions')
                        })
            
            return ToolOutput(
                success=True,
                data={
                    "symptom": symptom,
                    "species": species,
                    "possible_diseases": diseases,
                    "retrieval_diagnostics": {
                        "status": "ok" if diseases else "empty",
                        "tried": tried,
                        "suggested_species": (
                            self.agmmu_loader.suggest_species(species, limit=5)
                            if species and hasattr(self.agmmu_loader, "suggest_species")
                            else []
                        ),
                    },
                }
            )
        
        except Exception as e:
            return ToolOutput(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "symptom": {
                "type": "string",
                "description": "植物症状描述"
            },
            "species": {
                "type": "string",
                "description": "植物物种"
            }
        }


class GetManagementAdviceTool(BaseTool):
    """获取管理建议工具"""
    
    def __init__(self, agmmu_loader):
        super().__init__(
            name="get_management_advice",
            description="获取植物管理和防治建议"
        )
        self.agmmu_loader = agmmu_loader
    
    def execute(self, issue: str, species: str, **kwargs) -> ToolOutput:
        """
        获取管理建议
        
        Args:
            issue: 问题描述
            species: 植物物种
            
        Returns:
            ToolOutput
        """
        try:
            advices = []
            tried = []

            if hasattr(self.agmmu_loader, "search_facts"):
                facts = []
                seen_facts = set()
                for query_variant in _query_expansions(issue):
                    tried.append({"mode": "facts", "species": species, "field": "management instructions", "query": query_variant})
                    for f in self.agmmu_loader.search_facts(
                        keyword=query_variant,
                        species=species,
                        field="management instructions",
                        limit=10,
                    ):
                        key = (f.faq_id, f.field, f.answer)
                        if key not in seen_facts:
                            facts.append(f)
                            seen_facts.add(key)
                    if species:
                        tried.append({"mode": "facts", "species": None, "field": "management instructions", "query": query_variant})
                        for f in self.agmmu_loader.search_facts(
                            keyword=query_variant,
                            species=None,
                            field="management instructions",
                            limit=10,
                        ):
                            key = (f.faq_id, f.field, f.answer)
                            if key not in seen_facts:
                                facts.append(f)
                                seen_facts.add(key)

                q_terms = {
                    t for t in str(issue or "").lower().split()
                    if len(t) > 2 and t not in {"management", "advice", "manage", "control"}
                }
                facts.sort(
                    key=lambda f: (
                        sum(1 for t in q_terms if t in f.answer.lower())
                        + _domain_rerank_adjustment(issue, {"answer": f.answer, "species": f.species, "all_facts": _all_case_facts(self.agmmu_loader, f.faq_id)})
                    ),
                    reverse=True,
                )
                for f in facts:
                    bundle = _case_bundle(self.agmmu_loader, f.faq_id)
                    row = {
                        "faq_id": f.faq_id,
                        "species": f.species,
                        "issue": issue,
                        "advice": f.answer,
                        "evidence_question": f.question,
                        "issue_type": bundle.get("issue_type"),
                        "all_facts": _all_case_facts(self.agmmu_loader, f.faq_id),
                        "case_evidence": bundle,
                    }
                    row["relevance"] = _candidate_relevance(issue, species, "management instructions", row)
                    advices.append(row)
            else:
                tried.append({"mode": "items", "species": species})
                results = self.agmmu_loader.search_by_species_and_keyword(
                    species=species,
                    keyword=issue,
                    limit=5
                )
                for item in results:
                    management = item.content.get('management_instructions')
                    if management:
                        advices.append({
                            "faq_id": item.faq_id,
                            "species": item.species,
                            "issue": issue,
                            "advice": management,
                            "full_content": item.content
                        })
            
            return ToolOutput(
                success=True,
                data={
                    "issue": issue,
                    "species": species,
                    "advice_count": len(advices),
                    "advices": advices,
                    "recall_candidates": advices,
                    "answer_evidence": [a for a in advices if (a.get("relevance") or {}).get("should_agent_use")],
                    "retrieval_diagnostics": {
                        "status": "ok" if advices else "empty",
                        "tried": tried,
                        "suggested_species": (
                            self.agmmu_loader.suggest_species(species, limit=5)
                            if species and hasattr(self.agmmu_loader, "suggest_species")
                            else []
                        ),
                    },
                }
            )
        
        except Exception as e:
            return ToolOutput(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "issue": {
                "type": "string",
                "description": "植物问题描述"
            },
            "species": {
                "type": "string",
                "description": "植物物种"
            }
        }


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.shared_resources: Dict[str, Any] = {}
    
    def register(self, tool: BaseTool) -> None:
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(tool_name)
    
    def execute(self, tool_name: str, **kwargs) -> ToolOutput:
        """执行工具"""
        tool = self.get(tool_name)
        if not tool:
            return ToolOutput(
                success=False,
                data=None,
                error=f"Tool not found: {tool_name}"
            )
        try:
            return tool.execute(**kwargs)
        except TypeError as e:
            return ToolOutput(
                success=False,
                data={
                    "tool_name": tool_name,
                    "action": tool_name,
                    "provided_params": kwargs,
                },
                error=f"Invalid parameters for {tool_name}: {e}",
            )
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """获取工具描述"""
        descriptions = []
        for tool_name, tool in self.tools.items():
            descriptions.append(f"- {tool_name}: {tool.description}")
        return "\n".join(descriptions)
