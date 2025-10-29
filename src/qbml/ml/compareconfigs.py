import yaml
from pathlib import Path


def compareconfigs(models: tuple):
    for fpath in models:
        model_path = Path(fpath)
        config_path = Path("hydra") / "config.yaml"
        with open(model_path / config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(model_path)
            print(config)
            print("\n")



if __name__ == "__main__":
    main()
