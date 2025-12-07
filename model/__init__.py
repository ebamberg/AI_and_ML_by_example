from datetime import date
from pydantic import Field, BaseModel
from typing import Optional, List

class ProductDescription(BaseModel):
    product_name: str = Field (description="The name of the product.")
    description: str = Field (description="The description of the product.")
    release_date: Optional[date] = Field (description="the planned release date of the product")

class MarketingText(BaseModel):
    product_name: str = Field (description="The name of the product.")
    marketing_text: str = Field (description="The Marketing text for that product")

class Critique(BaseModel):
    critique_points: List[str] = Field (description="List of critiques. Each item represents one critique")

class Score(BaseModel):
    score: float = Field (description="The score as integer value.")

