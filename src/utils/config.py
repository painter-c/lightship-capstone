import tomli

CONFIG_PATH = 'config.toml'
CREATOR_BLACKLIST_PATH = 'blacklist_creator.txt'
ASSIGNEE_BLACKLIST_PATH = 'blacklist_assignee.txt'

def load():
    with open(CONFIG_PATH, 'rb') as f:
        return tomli.load(f)
    
def load_creator_blacklist():
    with open(CREATOR_BLACKLIST_PATH, 'rb') as f:
        return [s.decode('UTF-8') for s in list(f)]
    
def load_assignee_blacklist():
    with open(ASSIGNEE_BLACKLIST_PATH, 'rb') as f:
        return [s.decode('UTF-8') for s in list(f)]