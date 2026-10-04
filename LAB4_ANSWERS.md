# Lab 4 — Orchestrating the app with Docker Compose

## Question 1 — MLflow without persistent storage

When MLflow runs without a volume, everything written under `/mlflow-data` is stored in that container's writable filesystem. Removing the container deletes its SQLite database, experiments, registered models and artifacts.

Starting a new container from the same image creates a fresh MLflow installation. The UI contains only a new `Default` experiment because container images do not preserve data written while an earlier container was running.

## Question 2 — Named volume versus bind mount

A named volume is managed by Docker and is independent of the repository and any individual container. It is portable between containers on the same Docker host and avoids host-specific folder paths and permission details.

A bind mount would also preserve the MLflow database and artifacts, and it can be useful during local development because the files are directly visible on the host. However, it couples the Compose configuration to a host directory and may create Windows/Linux permission and path issues.

## Question 3 — Compose service discovery

Docker Compose creates a private network for the application. Each service name becomes a DNS hostname on that network, so the inference container can resolve `mlflow` and connect to `http://mlflow:5000`.

In Lab 3, MLflow ran on the host rather than inside the container network, so the inference container needed `host.docker.internal`.

## Question 4 — Configurable inference URL

The frontend reads `INFERENCE_URL` from an environment variable so the same image can run in different environments.

Inside Compose, it uses `http://inference:8000`. If the frontend is run independently, the value can be changed to an externally reachable API URL without editing or rebuilding the image.

## Question 5 — Published and internal ports

MLflow publishes port `5000` because a user needs to open its web interface from the host. The frontend publishes port `8501` because a user needs to open the Streamlit page.

Inference does not publish port `8000` to the host because only the frontend needs to contact it. Both containers are on the private Compose network, so the frontend reaches it through the service hostname `http://inference:8000`.

## Question 6 — Start order versus readiness

`depends_on` starts the MLflow container before the inference container, but it does not guarantee that the MLflow server is ready to accept requests.

If inference tries to load the registered model too early, its startup raises a connection error and the inference container exits. The error can be inspected with `docker compose logs inference`. It can then be restarted after MLflow becomes ready.

During this run, MLflow became ready before inference requested the model, so inference downloaded the artifacts and started successfully.

## Question 7 — Compose service status

`docker compose ps` showed that the MLflow, inference, and frontend services were all running.

MLflow publishes port `5000` and the frontend publishes port `8501`, so both can be accessed from the host computer. The inference service shows only `8000/tcp` because its port is not published to the host. It is accessed internally by the frontend through the Docker Compose network using `http://inference:8000`.

## Question 8 — Switching model versions

Changing the `champion` alias does not immediately change the model used by an already-running inference service. The service loads the model once during application startup and keeps it in memory.

After the `champion` alias was moved to Version 3, I ran:

`docker compose restart inference`

This restarted only the inference service. During startup, it resolved `models:/food11@champion` again and loaded Version 3. Rebuilding the Docker image was not necessary.

The prediction remained the same because Versions 2 and 3 reference the same trained model artifact.

## Question 9 — Model and image separation

The inference Docker image contains the serving code and its Python dependencies. The trained model is not embedded directly in the image.

At startup, the inference service connects to MLflow and loads the model referenced by `models:/food11@champion`. This allows the deployed model version to be changed by moving the alias and restarting the inference service, without rebuilding the Docker image.

## Question 10 — Volume persistence

Running `docker compose down` removed the containers and network but preserved the named volume. After starting the services again, the registered model, its versions, aliases, and artifacts were still available.

Running `docker compose down -v` also deleted the named volume. When the services started again, MLflow used a new empty database, so the `food11` registered model was no longer present.

This demonstrates that container deletion does not delete data stored in a named volume, but explicitly deleting the volume removes that persisted data.

## Question 11 — Production scaling

Docker Compose is useful for running this application on one machine, but it does not provide complete production orchestration across multiple machines.

For production, a platform such as Kubernetes could run multiple replicas of the frontend and inference services, restart failed containers, distribute traffic through load balancers, perform health checks, and deploy updates gradually.

MLflow would also need a shared production database instead of SQLite and shared artifact storage such as an object-storage service. This would allow multiple MLflow instances and inference replicas to access the same model registry and artifacts reliably.