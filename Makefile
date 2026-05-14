.PHONY: help build up down logs deploy undeploy lint test clean

IMAGE_PREFIX ?= cabflow
TAG ?= latest
NAMESPACE ?= cabflow

help:
	@echo "CabFlow Make targets"
	@echo "  build        Build docker images (api + dashboard)"
	@echo "  up           Start the full stack with docker compose"
	@echo "  down         Stop the docker compose stack"
	@echo "  logs         Tail docker compose logs"
	@echo "  deploy       kubectl apply -f k8s/"
	@echo "  undeploy     Delete the $(NAMESPACE) namespace (removes everything)"
	@echo "  lint         Validate k8s manifests with kubectl --dry-run=client"
	@echo "  test         Run pytest"
	@echo "  clean        Remove __pycache__, .pytest_cache, *.egg-info"

build:
	docker build -t $(IMAGE_PREFIX)/dashboard:$(TAG) -f Dockerfile .
	docker build -t $(IMAGE_PREFIX)/api:$(TAG) -f Dockerfile.api .

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

deploy:
	kubectl apply -f k8s/

undeploy:
	kubectl delete namespace $(NAMESPACE)

lint:
	kubectl apply --dry-run=client -f k8s/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
