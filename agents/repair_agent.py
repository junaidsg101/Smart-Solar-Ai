import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RepairAgent:
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        fault = input_data.get('diagnosis', {}).get('fault_type', 'Unknown')
        logger.info(f"RepairAgent: Generating repair plan for {fault}")

        steps = {
            "Wiring_Fault": ["1. Turn off DC disconnect.", "2. Inspect junction box.", "3. Test continuity.", "4. Replace damaged wiring."],
            "Grid_Voltage_Fluctuation": ["1. Check grid connection.", "2. Verify inverter profile.", "3. Monitor voltage 24h.", "4. Contact utility."],
            "Monitoring_Offline": ["1. Check ethernet/Wi-Fi.", "2. Reboot gateway.", "3. Verify network creds.", "4. Ping gateway."],
            "Loose_Corroded_Wiring": ["1. Inspect MC4 connectors.", "2. Clean corrosion.", "3. Tighten connections.", "4. Apply dielectric grease."]
        }
        input_data['repair_plan'] = steps.get(
            fault, ["1. Escalate to senior technician for manual review."])
        return input_data

