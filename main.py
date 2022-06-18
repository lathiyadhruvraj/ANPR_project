import argparse
from training import training

def start_training(parsed_args):
	training(config_path=parsed_args.config)

if __name__ == "__main__":
	args = argparse.ArgumentParser()

	args.add_argument("--config", "-c", default="v5_config.yaml")

	parsed_args = args.parse_args()

	start_training(parsed_args)

