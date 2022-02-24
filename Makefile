TAG := $(shell python -c "import utils; print(utils.get_version())")

ifndef skip_test
	extra_dep_docker:= test

else
	extra_dep_docker :=
endif

docker: $(extra_dep_docker) docker-main docker-per-model

docker-main:
	mkdir -p docker-build
	cp setup.py docker-build/
	cp VERSION docker-build/ || echo "VERSION not found"
	cp utils.py docker-build/
	cp requirements.txt docker-build/
	cp -r slaid docker-build/
	cp -r bin docker-build/
	cd docker-build &&	docker build . -f ../docker/Dockerfile -t slaid:$(TAG) --build-arg $(BUILD_ARG)
	docker tag slaid:$(TAG) slaid
	tests/docker/test_docker.sh

install:
	pip install --use-feature=2020-resolver -e .
	touch install

test: install

	PYTHONPATH=$(PYTHONPATH):$(PWD)/tests tests/run_all_tests.sh
	touch test

docker-per-model:
	mkdir -p docker-build
	cd docker; PYTHONPATH=$(PYTHONPATH):$(PWD) ./docker_cmd.py -v $(TAG) $(DOCKER_ARGS) build --build-arg $(BUILD_ARG) -e $(EXTRA_TAGS)
clean:
	rm -f install
	rm -f test
	rm -rf docker-build

docker-push: docker
	cd docker/; PYTHONPATH=$(PYTHONPATH):$(PWD) ./docker-push.sh $(TAG) $(EXTRA_TAGS)

