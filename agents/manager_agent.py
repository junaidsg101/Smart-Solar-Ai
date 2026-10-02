import logging
from typing import Dict, Any
from patterns.sequence import SequencePattern
from patterns.parallel import ParallelPattern
from agents.data_agent import DataAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.repair_agent import RepairAgent
from agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)


class ManagerAgent:
    def __init__(self):
        self.data_agent = DataAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.repair_agent = RepairAgent()
        self.report_agent = ReportAgent()

        self.specialists = {
            "Wiring_Fault": self._handle_wiring_fault,
            "Grid_Voltage_Fluctuation": self._handle_grid_fluctuation,
            "Monitoring_Offline": self._handle_monitoring_offline,
            "Loose_Corroded_Wiring": self._handle_corroded_wiring,
            "default": self._handle_default
        }

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("ManagerAgent: Received new diagnostic task.")
        data = self.data_agent(input_data)
        data = self.diagnostic_agent(data)
        fault_type = data.get('diagnosis', {}).get('fault_type', 'default')

        target_workflow = self.specialists.get(
            fault_type, self.specialists["default"])
        data = target_workflow(data)
        return self.report_agent(data)

    def _handle_wiring_fault(self, data: Dict[str, Any]) -> Dict[str, Any]:
        seq = SequencePattern()
        return seq.execute([self._check_junction_box, self.repair_agent], data)[-1]

    def _handle_grid_fluctuation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        par = ParallelPattern()
        data['parallel_checks'] = par.execute(
            [self._check_inverter, self._check_grid], data)
        return self.repair_agent(data)

    def _handle_monitoring_offline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        seq = SequencePattern()
        return seq.execute([self._reboot_gateway, self._verify_connection, self.repair_agent], data)[-1]

    def _handle_corroded_wiring(self, data: Dict[str, Any]) -> Dict[str, Any]:
        seq = SequencePattern()
        return seq.execute([self._inspect_connectors, self._clean_and_tighten, self.repair_agent], data)[-1]

    def _handle_default(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.repair_agent(data)

    def _check_junction_box(self, d): return d
    def _check_inverter(self, d): return "Inverter OK"
    def _check_grid(self, d): return "Grid OK"
    def _reboot_gateway(self, d): return d
    def _verify_connection(self, d): return d
    def _inspect_connectors(self, d): return d
    def _clean_and_tighten(self, d): return d

