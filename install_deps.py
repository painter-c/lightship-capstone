import sys
import subprocess

packages = [
    'numpy',
    'pandas',
    'sklearn',
    'gensim',
    'nltk',
    'tomli',
    'scipy',
    'bertopic',
    'umap'
]

def install(pkg):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', pkg])

for pkg in packages:
    install(pkg)
