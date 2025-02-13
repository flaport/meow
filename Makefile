build:
	uv run python -m build --sdist --wheel

docker:
	docker build . -t flaport/meow:latest

nbrun:
	find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | xargs parallel -j `nproc --all` uv run papermill {} {} -k python3 :::

dockerpush:
	docker push flaport/meow:latest

.PHONY: docs
docs:
	uv run sphinx-apidoc --force --no-toc --no-headings --implicit-namespaces --module-first --maxdepth 1 --output-dir docs/source meow
	cd docs && make html

run:
	find examples -name "*.ipynb" | grep -v ipynb_checkpoints | xargs -I {} uv run papermill -k meow {} {}

clean:
	find . -name "dist" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*.so" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	find . -name ".mypy_cache" | xargs rm -rf
