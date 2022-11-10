build:
	python -m build --sdist --wheel

docker:
	docker build . -t flaport/meow:latest

dockerpush:
	docker push flaport/meow:latest

.PHONY: docs
docs:
	sphinx-apidoc --force --no-toc --no-headings --implicit-namespaces --module-first --maxdepth 1 --output-dir docs/source meow
	cd docs && make html

run:
	find examples -name "*.ipynb" | grep -v ipynb_checkpoints | xargs -I {} papermill -k meow {} {}
