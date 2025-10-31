# dify_plugin_grounding_with_google_search

```bash
uv pip compile pyproject.toml -o grounding_with_google_search/requirements.txt
```

```bash
uv run black . -C -l 100 && uv run ruff check --fix
```

```bash
uv run black . -C -l 100 && uv run ruff check --fix
uv pip compile pyproject.toml -o grounding_with_google_search/requirements.txt
mkdir -p difypkg
./dify-plugin-windows-amd64.exe plugin package grounding_with_google_search/ -o difypkg/grounding_with_google_search-0.1.0.difypkg
```