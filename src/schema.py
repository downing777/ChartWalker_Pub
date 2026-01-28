

from enum import Enum


from dataclasses import dataclass, field
from typing import Any, Union, TypedDict, List, Dict, Optional, Literal, Set


from typing import Dict, List, Optional
import networkx as nx
from enum import Enum
from dataclasses import dataclass, field


class EntityType(Enum):
    '''
    Enum: Generic enumerations
    For fixed values, usage: e.name, e.value for e in EntityType
    '''
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    EVENT = "Event"
    OBJECT = "Object"
    CONCEPT = "Concept"
    WORK = "Work"  
    #TIME = "Time"
    #MULTIMODAL = "Multimodal" # other modality entity e.g. image video audio 
    @classmethod
    def description(cls) -> str:
        values = [e.value for e in cls]
        return f"{', '.join(values)}"

class RelationType(Enum):
    TAXONOMIC = "Taxonomic"  # (is-a, instance-of)
    MERONYMIC = "Meronymic"  # (part-of)
    ATTRIBUTIVE = "Attributive"  # (has-property)
    TEMPORAL = "Temporal"  # (before, after) time related
    SPATIAL = "Spatial"  # (near, inside)
    CAUSAL = "Causal"  # (causes, leads-to)
    SOCIAL = "Social"  # (knows, works-with)
    SEMANTIC = "Semantic"  #  (synonym, antonym)
#    HIERARCHICAL = "Hierarchical"  # (parent-child)

    @classmethod
    def description(cls) -> str:
        values = [e.value for e in cls]
        return f"{', '.join(values)}"


@dataclass
class NodeAttributes:
    '''
    params:
        description: Entity description
        source: List[str] chunk id
        embedding: Optional[List[float]] = None
        metadata: Dict[str, Any] 
        confidence: float = 1.0
    '''
    description: str = ""
    source: List[str] = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class Entity:
    id: str
    name: str
    type: EntityType
    attributes: NodeAttributes
    level: int = 0  # the layer level of the HKG
    source: Set[str] = field(default_factory=set)  # the source of the entity

    def summary(self) -> str:
        return {"name": self.name, "source": self.source}

    def __repr__(self) -> str:
        #src_str = ",".join(sorted(self.source)) if self.source else "N/A"
        attrs_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        src_num = len(self.source)
        return f"{self.name}(attr:{attrs_str}, src_num: {src_num})"


@dataclass
class Relation:
    source_id: str
    target_id: str
    id: str
    type: RelationType
    attributes: Dict[str, Any] = field(default_factory=dict)
    layer: int = 0  # 0 is the finest layer
    #source: Set[str] = field(default_factory=set)


@dataclass
class GraphLayer:
    graph: nx.MultiDiGraph
    layer_level: int
    node_mapping: Dict[str, List[str]]  # coarse node to fine nodes mapping

@dataclass
class MMChunk:
    """
    Minium process unit
    a chunk should be a subparagraph of a text document, a picture, a table or a video with necessary textual context
    Params:
        source: str file path
        type: Literal['text', 'image', 'table']
        content: str #original text content or summary of other modality data
        image_path
        context 
    """
    id: str 
    type: Literal['text', 'image', 'table']
    #content: Optional[str] = None #original text content or summary of other modality data
    image_path: Optional[str] = None
    context: Optional[str] = None
    source: str = None
    #entities: List[Entity]
    #relationships: List[Relation]