# UNLP 2026: Team XXX solution

The repository includes the solution to the [UNLP-2026 shared task](https://github.com/unlp-workshop/unlp-2026-shared-task).


uv run pip download tantivy==0.25.1 \
  --only-binary=:all: \
  --platform manylinux2014_aarch64 \
  --python-version 3.12 \
  --implementation cp \
  --abi cp312 \
  -d wheels
