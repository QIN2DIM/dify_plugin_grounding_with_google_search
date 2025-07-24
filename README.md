# dify_plugin_grounding_with_google_search

```bash
mkdir -p difypkg
./dify-plugin-windows-amd64.exe plugin package grounding_with_google_search/ -o difypkg/grounding_with_google_search-0.0.1.difypkg
```

```bash
uv pip compile pyproject.toml -o grounding_with_google_search/requirements.txt
```