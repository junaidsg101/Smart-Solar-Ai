import os
import sys
import json
import argparse
from utils.logger import setup_logger
from utils.helpers import load_config, get_project_root
from agents.manager_agent import ManagerAgent

config = load_config()
log_level = config.get('app', {}).get('log_level', 'INFO')
logger = setup_logger("SolarAI", os.path.join(
    get_project_root(), "logs", "app.log"), level=log_level)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Solar AI Multi-Agent Diagnostic Pipeline")
    parser.add_argument("--site", default="SITE-01",
                        help="Site ID to diagnose")
    parser.add_argument("--voltage", type=float,
                        default=230.0, help="L1 Voltage")
    parser.add_argument("--ping", type=float, default=15.0,
                        help="Network Ping (ms)")
    args = parser.parse_args()

    logger.info(f"Starting diagnostic pipeline for {args.site}")

    payload = {
        "site_id": args.site,
        "panel_temp_c": 35.0,
        "irradiance_wm2": 850,
        "power_output_kw": 10.0,
        "voltage_l1_v": args.voltage,
        "resistance_ohm": 2.50 if args.voltage < 200 else 0.05,
        "ping_ms": args.ping
    }

    manager = ManagerAgent()
    result = manager(payload)

    print("\n" + "="*50)
    print("🌞 FINAL DIAGNOSTIC REPORT")
    print("="*50)
    print(json.dumps(result.get('final_report', {}), indent=4))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

