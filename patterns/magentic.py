import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


class MagenticPattern:
    def __init__(self, specialists: Dict[str, Callable]):
        self.specialists = specialists

    def execute(self, context: str, input_data: Any) -> Any:
        logger.info(f"MagenticPattern: Delegating '{context}' to specialist.")
        specialist = self.specialists.get(
            context, self.specialists.get("default"))
        return specialist(input_data) if specialist else input_data

