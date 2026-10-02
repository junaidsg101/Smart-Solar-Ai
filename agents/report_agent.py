import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportAgent:
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("ReportAgent: Compiling final diagnostic report...")
        input_data['final_report'] = {
            "timestamp": datetime.now().isoformat(),
            "site_id": input_data.get('site_id', 'UNKNOWN'),
            "diagnosis": input_data.get('diagnosis', {}),
            "repair_plan": input_data.get('repair_plan', []),
            "status": "Action Required" if input_data.get('diagnosis', {}).get('fault_type') != 'Normal' else "Operational"
        }
        return input_data

