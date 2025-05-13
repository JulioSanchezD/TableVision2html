import sys
import yaml
import logging
import argparse

# Set up logging configuration
logging.addLevelName(35, 'INFO')
logging.basicConfig(level='INFO', format='%(asctime)s - %(levelname)s - %(message)s')

# Importing main functions from the modules
from src.inference import main as inference
from src.models.granite_vision.finetune import main as granite_vision_finetune


def run(config_path: str):
    """
    Run modules based on the configuration file.
    :param config_path: Path to the YAML configuration file.
    """
    # Load the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    modules = config.get('modules', [])
    if not modules:
        logging.error("No modules configuration found in the YAML file.")
        sys.exit(1)

    # Map module names to their functions
    module_map = {
        'inference': lambda params: inference(**params),
        'granite_vision_finetune': lambda params: granite_vision_finetune(**params),
    }

    try:
        for step in modules:
            module = step.get('module')
            run = step.get('run', False)
            params = step.get('params', {})

            if not run:
                logging.log(35, f"Skipping module: {module}...")
                continue

            if module not in module_map:
                logging.error(f"Unknown module: {module}")
                continue

            logging.log(35, f"Running module: {module}{f' with params: {params}' if params else ''}...")
            module_map[module](params)

        logging.log(35, "Modules execution completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the module execution: {e}", exc_info=True)
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the desired modules using a YAML configuration.")
    parser.add_argument('--config', type=str, required=False, help="Path to the YAML configuration file.",
                        default='run_config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args.config)