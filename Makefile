export MLFLOW_DISABLE_AGENT_HINT := 1
export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR := false
API ?= http://localhost:8000
RUN := uv run

.PHONY: setup data train promote serve test lint drift simulate status rollback mlflow-ui monitoring repro image docker-serve traffic reset all

setup:          ## install the locked environment
	uv sync --locked

data:           ## download the TLC files and build zone x hour features (DVC pipeline)
	$(RUN) dvc repro

train:          ## train on the reference period, register as @staging
	$(RUN) python -m taxi.train

promote:        ## human approval: @staging -> @production (and reload the API if it runs)
	$(RUN) python -m taxi.registry promote --api $(API) || $(RUN) python -m taxi.registry promote

rollback:       ## @production -> the version it replaced (and reload the API)
	$(RUN) python -m taxi.registry rollback --api $(API) || $(RUN) python -m taxi.registry rollback

status:         ## versions, aliases, metrics, git commit
	$(RUN) python -m taxi.registry status

serve:          ## the API on :8000 (docs on /docs, metrics on /metrics/)
	$(RUN) uvicorn app.main:app --port 8000

drift:          ## drift check on the 24 h before END, e.g. make drift END="2020-03-16 23:00"
	$(RUN) python -m taxi.drift --end "$(END)"

simulate:       ## replay March-April 2020: drift alarm, gated retraining, approval
	$(RUN) python -m taxi.simulate --approve

mlflow-ui:      ## MLflow UI on :5001 (5000 is AirPlay on a Mac)
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

monitoring:     ## Prometheus :9090, Alertmanager :9093, Grafana :3000 (needs Docker and `make serve`)
	docker compose up -d

repro:          ## refit version V (default 1) from code + data and compare with the logged metrics (1e-6)
	$(RUN) python -m taxi.repro_check --version $(or $(V),1)

image:          ## build the API image, tagged with the commit
	docker build --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) -t taxi-api:$$(git rev-parse --short HEAD) .

docker-serve:   ## the API from its image on :8000 (repo mounted at the same path: registry paths are absolute)
	docker run --rm -p 8000:8000 -v "$(CURDIR):$(CURDIR)" -e MLFLOW_TRACKING_URI="sqlite:///$(CURDIR)/mlflow.db" \
		taxi-api:$$(git rev-parse --short HEAD)

traffic:        ## live demo traffic: make traffic PERIOD=normal  or  PERIOD=covid
	$(RUN) python -m taxi.traffic $(PERIOD) -n 300 --rate 50

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .

reset:          ## forget every run and model version (keeps the data)
	rm -rf mlflow.db mlruns mlartifacts state reports

all: setup data train promote simulate  ## fresh clone to full incident story
