import logging
from typing import List, Callable, Any

logger = logging.getLogger(__name__)


class SequencePattern:
    def execute(self, tasks: List[Callable], initial_input: Any) -> List[Any]:
        current_input = initial_input
        results = []
        for task in tasks:
            name = getattr(task, '__name__', task.__class__.__name__)
            logger.info(f"SequencePattern: Executing {name}")
            current_input = task(current_input)
            results.append(current_input)
        return results

