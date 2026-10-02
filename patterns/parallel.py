import logging
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ParallelPattern:
    def execute(self, tasks: List[Callable], input_data: Any) -> Dict[str, Any]:
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for task in tasks:
                name = getattr(task, '__name__', task.__class__.__name__)
                futures[executor.submit(task, input_data)] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = f"Error: {str(e)}"
        return results

