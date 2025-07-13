# distributionally-robust-recommendation

## Setup
This repository is using uv.  
If you want to use this repository, please run the following command.

1. install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. install dependencies and create a virtual environment
```
uv sync
```
3. activate the virtual environment (optional)
```
source .venv/bin/activate
```

In addition, this repository requires a mosek licence.
If necessary, obtain the appropriate licence from [here](https://www.mosek.com/).


## Formatter
```
./bin/format.sh
# or run separately
uv run ruff format ./
uv run ruff check ./ --fix
```

