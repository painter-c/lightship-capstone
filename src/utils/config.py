import tomli

CONFIG_PATH = 'config.toml'

def load():
    with open(CONFIG_PATH, 'rb') as f:
        return tomli.load(f)