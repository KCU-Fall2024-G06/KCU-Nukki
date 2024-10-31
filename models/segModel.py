from pydantic import BaseModel, conlist
from typing import List

class Pair(BaseModel):
    __root__: conlist(int, min_items=2, max_items=2)