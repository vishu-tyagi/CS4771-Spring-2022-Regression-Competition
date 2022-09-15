SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

IMG = dnn
BASE_IMG = dnn-base
TEST_IMG = dnn-test
IMG_TAG ?= latest

LOCAL_DATA_DIR = ${CWD}data
DOCKER_DATA_DIR = /usr/src/app/data
LOCAL_MODEL_DIR = ${CWD}model
DOCKER_MODEL_DIR = /usr/src/app/model
LOCAL_REPORTS_DIR = ${CWD}reports
DOCKER_REPORTS_DIR = /usr/src/app/reports

export

.PHONY: build-base
build-base:
	@docker build -t ${BASE_IMG}:${IMG_TAG} -f Dockerfile.base .

.PHONY: build
build: build-base
	docker build -t ${IMG}:${IMG_TAG} --build-arg IMAGE=${BASE_IMG}:${IMG_TAG} .

.PHONY: test-image
test-image: build-base
	docker build -t ${TEST_IMG}:${IMG_TAG} -f Dockerfile.test --build-arg IMAGE=${BASE_IMG}:${IMG_TAG} .

test: test-image
	docker run -t ${TEST_IMG}:${IMG_TAG}

train:
	docker run -t \
		-v ${LOCAL_DATA_DIR}:${DOCKER_DATA_DIR} \
		-v ${LOCAL_MODEL_DIR}:${DOCKER_MODEL_DIR} \
		-v ${LOCAL_REPORTS_DIR}:${DOCKER_REPORTS_DIR} \
		${IMG}:${IMG_TAG} train