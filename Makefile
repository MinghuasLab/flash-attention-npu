
clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

.PHONY: quality quality-changed quality-fix quality-docker quality-fix-docker \
	quality-all quality-all-docker quality-install

quality:
	ci/code_quality.sh --staged

quality-changed:
	@base="$${BASE_SHA:-$$(git rev-parse HEAD^)}"; \
	ci/code_quality.sh --base "$$base"

quality-fix:
	ci/code_quality.sh --staged

quality-docker:
	ci/run_code_quality_container.sh --staged

quality-fix-docker:
	ci/run_code_quality_container.sh --staged

quality-all:
	ci/code_quality.sh --all-files

quality-all-docker:
	ci/run_code_quality_container.sh --all-files

quality-install:
	python3 -m pip install -r ci/quality-requirements.txt
	python3 -m pre_commit install
