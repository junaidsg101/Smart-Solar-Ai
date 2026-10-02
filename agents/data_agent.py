import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DataAgent:
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("DataAgent: Starting data ingestion and preprocessing...")
        processed_data = input_data.copy()
        processed_data['panel_temp_c'] = float(
            processed_data.get('panel_temp_c', 25.0))
        processed_data['voltage_l1_v'] = float(
            processed_data.get('voltage_l1_v', 230.0))
        processed_data['ping_ms'] = float(processed_data.get('ping_ms', 15.0))
        return processed_data

