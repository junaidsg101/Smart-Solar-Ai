import logging
from typing import Dict, Any
from ml.predict import SolarFaultPredictor

logger = logging.getLogger(__name__)


class DiagnosticAgent:
    def __init__(self):
        self.predictor = SolarFaultPredictor()

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("DiagnosticAgent: Running fault classification...")
        try:
            prediction_result = self.predictor.predict(input_data)
            input_data['diagnosis'] = {
                'fault_type': prediction_result['fault_type'],
                'confidence': prediction_result['confidence'],
                'all_probabilities': prediction_result['all_probabilities']
            }
            logger.info(
                f"DiagnosticAgent: Predicted {prediction_result['fault_type']} ({prediction_result['confidence']:.2f})")
        except Exception as e:
            logger.error(f"DiagnosticAgent: Error during diagnosis: {str(e)}")
            input_data['diagnosis'] = {
                'fault_type': 'Unknown', 'confidence': 0.0, 'error': str(e)}
        return input_data

