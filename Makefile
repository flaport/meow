docker:
	docker build . -t flaport/meow:latest

dockerpush:
	docker push flaport/meow:latest

.PHONY: docs
docs:
	cd docs && make html
