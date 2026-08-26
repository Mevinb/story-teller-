"""
Structured data schemas for Grok-Style Visual Analysis & Rating.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Demographics:
    apparent_age_range: str = "Unknown"
    apparent_gender: str = "Unspecified"
    aesthetic_vibe: str = "Neutral"


@dataclass
class FacialFeatures:
    hair: str = "Unspecified"
    eyes: str = "Unspecified"
    face_shape: str = "Unspecified"
    expression: str = "Neutral"
    distinctive_marks: str = "None noted"


@dataclass
class BodyStructure:
    build: str = "Average"
    body_shape: str = "Unspecified"
    bust_chest: str = "Unspecified"
    waist_midsection: str = "Unspecified"
    hips_thighs: str = "Unspecified"
    glutes_lower_body: str = "Unspecified"
    physique_appreciation: str = "Unspecified"
    bust_details: str = "Unspecified"
    waist_hips_details: str = "Unspecified"
    legs_thighs_details: str = "Unspecified"
    glutes_rear_details: str = "Unspecified"
    overall_look_details: str = "Unspecified"
    height_estimate: str = "Medium"
    posture: str = "Upright"
    upper_body: str = "Unspecified"
    lower_body: str = "Unspecified"


@dataclass
class ClothingStyle:
    outfit: str = "Casual"
    style_category: str = "Modern"
    colors: List[str] = field(default_factory=list)
    accessories: List[str] = field(default_factory=list)


@dataclass
class RatingBreakdown:
    face: int = 0
    body: int = 0
    style: int = 0
    vibe: int = 0
    overall: int = 0


@dataclass
class PersonAnalysisResult:
    person_detected: bool = True
    summary: str = ""
    rating: int = 0
    rating_breakdown: RatingBreakdown = field(default_factory=RatingBreakdown)
    rating_explanation: str = ""
    demographics: Demographics = field(default_factory=Demographics)
    facial_features: FacialFeatures = field(default_factory=FacialFeatures)
    body_structure: BodyStructure = field(default_factory=BodyStructure)
    clothing_style: ClothingStyle = field(default_factory=ClothingStyle)
    roast: str = ""
    best_feature: str = ""
    advice: str = ""
    narrative_description: str = ""
    consistency_tags: List[str] = field(default_factory=list)
    backend_used: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonAnalysisResult":
        demo_data = data.get("demographics", {})
        face_data = data.get("facial_features", {})
        body_data = data.get("body_structure", {})
        cloth_data = data.get("clothing_style", {})
        rating_data = data.get("rating_breakdown", {})

        # Handle rating - could be int or string
        rating = data.get("rating", 0)
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            rating = 0

        return cls(
            person_detected=data.get("person_detected", True),
            summary=data.get("summary", ""),
            rating=rating,
            rating_breakdown=RatingBreakdown(**rating_data) if isinstance(rating_data, dict) else RatingBreakdown(),
            rating_explanation=data.get("rating_explanation", ""),
            demographics=Demographics(**demo_data) if isinstance(demo_data, dict) else Demographics(),
            facial_features=FacialFeatures(**face_data) if isinstance(face_data, dict) else FacialFeatures(),
            body_structure=BodyStructure(**body_data) if isinstance(body_data, dict) else BodyStructure(),
            clothing_style=ClothingStyle(**cloth_data) if isinstance(cloth_data, dict) else ClothingStyle(),
            roast=data.get("roast", ""),
            best_feature=data.get("best_feature", ""),
            advice=data.get("advice", ""),
            narrative_description=data.get("narrative_description", ""),
            consistency_tags=data.get("consistency_tags", []),
            backend_used=data.get("backend_used", "unknown")
        )
