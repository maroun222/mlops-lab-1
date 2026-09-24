# Lab 3 — Containerizing the model with Docker

## Question 1 — Registered model

The registered model is named `food11`, and MLflow assigned the initial registration Version `1`.

A logged model artifact is the model produced by a particular training run. A registered model gives that artifact a shared name and version, allowing it to be managed and served independently from the run that created it.

## Question 2 — Model aliases

MLflow replaced the deprecated built-in stages such as `Staging` and `Production` with user-defined aliases such as `champion` and `challenger`.

Registering a model separately from its training run gives it a stable name and version history that can be used for deployment. An alias is a mutable pointer to a model version. The alias can be reassigned to another version without changing the serving application's model URI.

The `champion` alias initially pointed to Version 1. It was later moved to Version 2, which references the same trained model through MLflow's HTTP artifact service so that it can be downloaded inside Docker.

## Question 3 — Loading through the Model Registry

Using `models:/food11@champion` separates the serving application from a particular model file and version. MLflow resolves the registered model name and alias, finds the corresponding artifact, and loads it using the model's saved flavor and environment information.

To serve a newer model, I can register the new version and move the `champion` alias to it. The serving code can continue using the same `models:/food11@champion` URI without changing a `.pth` path.

## Question 4 — Docker layer caching

The Dockerfile copies `pyproject.toml` and `uv.lock` and installs the dependencies before copying the source code. Docker can therefore cache the expensive dependency-installation layer.

If only a line in `serve.py` changes, Docker reuses the Python and dependency layers and only repeats the fast source-copy and final image layers. If the complete project were copied before dependency installation, every source-code change would invalidate the dependency layer and force all packages to be installed again.

## Question 5 — Image-size comparison

The multi-stage `food11-api:latest` image was approximately `2.06 GB`. The naive single-stage `food11-api:naive` image was approximately `2.20 GB`. The multi-stage image was therefore about `140 MB`, or approximately `6.4%`, smaller.

According to `docker history`, the largest layer in both images was the Python environment containing PyTorch, MLflow and the other dependencies. It was approximately `1.49 GB` in the multi-stage image and `1.53 GB` in the naive image. The naive image also retained the approximately `61 MB` layer used to install `uv`. The application source-code layer in the multi-stage image was only about `49.2 KB`.

## Question 6 — Docker build context

Without `.dockerignore`, Docker may send the virtual environment, dataset, MLflow artifacts, database, Git history and temporary Python files as build context. This increases context-transfer time, disk use and the risk of accidentally including local data or metadata in an image.

With the current multi-stage Dockerfile's selective `COPY` instructions, merely sending these folders would not necessarily put them in the final image. However, a Dockerfile using `COPY . .` could copy them. The local Windows `.venv` is the most likely to break or interfere with a Linux image because its executables and paths are incompatible. The large `data/` and `mlruns/` folders could also exhaust time, storage or memory. `.git/` and `__pycache__/` usually create unnecessary size rather than directly breaking the build.

## Question 7 — Reaching the host from Docker

Inside a container, `127.0.0.1` refers to the container itself, not the Windows host. Therefore, using `127.0.0.1:5000` inside the container would look for an MLflow server inside that same container.

On Docker Desktop, `host.docker.internal` resolves to the host computer, allowing the container to reach the MLflow server running on Windows.

## Question 8 — Recreating the container

After stopping and removing the first container, I started a new container from the same `food11-api:latest` image. The API started and `/health` returned `{"status":"ok"}` without rebuilding the image.

The Docker image contains the application code, Python environment and dependencies. The trained model is not baked into the image; the new container downloads the model at runtime from the MLflow server using the `food11@champion` alias.

## Question 9 — Distributing the Docker image

The Dockerfile is stored in Git, but the built `food11-api:latest` image currently exists only in the local Docker installation.

For another machine, CI runner or Kubernetes cluster to pull and run the exact image, the image must be tagged and pushed to a container registry such as Docker Hub or GitHub Container Registry. A unique version tag or immutable image digest should be used instead of relying only on the mutable `latest` tag.