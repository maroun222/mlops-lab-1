# Lab 1 — Git, DVC and Food-11 preparation

Run commands from this repository using PowerShell. Dependencies are managed by uv:

```powershell
uv sync
uv run python src/food11/data.py
```

The preparation script keeps raw images intact, preserves all three splits, converts images to RGB, resizes them to 128×128, and places them in category folders. The mini dataset takes the first 100 images in sorted filename order per category per split (or all images if fewer). It refuses to overwrite existing output folders to avoid stale files.

## Question 1 — Files created by uv init

In this project, `uv init` created `pyproject.toml` (project metadata, Python requirement and dependencies), `.python-version` (Python 3.12), `main.py` (starter program) and an empty `README.md` (project documentation). The `.git` folder already existed from cloning. A root `.gitignore` was absent at inspection; it was added during this solution to ignore the virtual environment and Python temporary files. `uv.lock` was subsequently created when dependencies were added; it records resolved versions for reproducibility.

## Question 2 — Files created by dvc init

`.dvc/config` holds shared DVC settings. `.dvc/.gitignore` excludes DVC cache, temporary state and local configuration from Git. `.dvcignore` controls files DVC should ignore when scanning. Commit these three files. Do not commit `.dvc/cache`, `.dvc/tmp` or `.dvc/config.local`; they are machine-specific or may contain secrets. The `.dvc` directory itself is a configuration directory, whereas `data.dvc` is a dataset pointer file.

## Question 3 — Credentials and configuration scopes

With `--global`, DVC stores settings in the user configuration, typically `%LOCALAPPDATA%\iterative\dvc\config` on Windows. Other scopes are `--system` (all users), `--project` (the default, `.dvc/config`) and `--local` (`.dvc/config.local`, ignored by Git). Precedence is local, project, global, then system.

Credentials must not be pushed to GitHub. Store the remote URL in project configuration so new clones know where to fetch data. Store the username and token locally. The lab's global-only remote URL would otherwise be missing on another computer.

## Question 4 — Change to .gitignore

`uv run dvc add data` adds `/data` to the root `.gitignore`. Git therefore ignores the actual image directory while DVC tracks its contents. Commit the updated `.gitignore` with `data.dvc`.

## Question 5 — Contents of data.dvc

`data.dvc` is a small YAML file describing the tracked directory. Its `outs` entry includes the path `data`, the hash algorithm and directory checksum, total size, and number of files. A directory checksum ends in `.dir` and identifies a cached manifest of its files. It contains no image bytes. The remote URL is configured separately in `.dvc/config`.

The actual raw-only snapshot is commit `d43356b`: 16,643 files, 1,188,442,712 bytes, checksum `a3a457d03c51ff8b037a833440f6ad13.dir`. There are 9,866 training, 3,347 evaluation and 3,430 validation images.

## Question 6 — GitHub and DagsHub

After a successful Git push, GitHub contains code, dependency files, non-secret DVC configuration, `.gitignore` and `data.dvc`. It does not contain the image directory. DVC combines the checksum in `data.dvc` with the configured remote URL to locate the data. After DagsHub is connected and `uv run dvc push` succeeds, the actual data objects are stored there and can be browsed through its DVC data interface. DagsHub upload is pending because no DagsHub repository has been created yet; this is the expected result, not a claim of an observed upload.

## Question 7 — Clone into a new folder

A fresh Git clone contains `data.dvc` but no `data` directory. Run `uv sync`, configure any required DVC credentials for that clone, then `uv run dvc pull`. Pull downloads the referenced data into the cache and restores it to the workspace. `dvc checkout` alone cannot download a missing cache from the remote.

Verified by cloning the GitHub repository into a new temporary directory: `data.dvc` existed and `data` did not. Remote pulling remains pending DagsHub setup.

```powershell
git clone https://github.com/maroun222/mlops-lab-1.git lab1-check
cd lab1-check
uv sync
# Configure local DagsHub authentication as described below.
uv run dvc pull
```

## Image preparation and ResNet

Outputs have the structure `data/food11_processed/<split>/<category>/<image>` and `data/food11_processed_mini/<split>/<category>/<image>`. Category names follow the lab exactly, including `Dairy product` and `Noodles-Pasta`.

This folder structure is expected by torchvision's `ImageFolder` loader; ResNet itself consumes image tensors, not directories. `ImageFolder` assigns labels alphabetically by folder name, so inspect `class_to_idx` instead of assuming it matches the raw filename labels (Egg and Fried food are ordered differently alphabetically). Use the loader's same class mapping across splits. Images are 128×128 as the lab requires. Pretrained ResNet inference normally uses the preprocessing supplied by its chosen weights, often including 224×224 crops and normalization; the lab's stored size is a separate requirement.

## Question 8 — Switching data versions

```powershell
git log --oneline -- data.dvc
git checkout <raw-only-commit-hash>
uv run dvc checkout
```

After checking out the raw-only pointer and running DVC checkout, `food11_processed` and `food11_processed_mini` disappear, leaving `food11_raw`. Git changes the pointer, and DVC changes the actual workspace data to match it. The data versions remain in the DVC cache. Return to the current version with:

```powershell
git checkout main
uv run dvc checkout
```

## Finish the DagsHub portion

Create a DagsHub account and connect the existing GitHub repository `maroun222/mlops-lab-1`. Copy the DVC remote configuration from its Remote → Data menu; use your actual DagsHub username, which may differ from the GitHub username. The following HTTP configuration matches the lab handout; use it only if the repository supplies that HTTP endpoint.

```powershell
uv run dvc remote add -d origin https://dagshub.com/<dagshub-user>/mlops-lab-1.dvc
uv run dvc remote modify --local origin auth basic
uv run dvc remote modify --local origin user <dagshub-user>
uv run dvc remote modify --local origin ask_password true
git add .dvc/config
git commit -m "Configure DagsHub DVC remote"
git push
uv run dvc push
```

Enter your DagsHub access token at the password prompt. The URL and default remote are committed; local authentication settings are ignored. In the fresh clone for Question 7, repeat the three local authentication commands before pulling.

Current DagsHub documentation shows an S3-compatible DVC configuration instead. If that is what your repository supplies, use its commands with `uv run` and install `uv add "dvc[s3]"`. The shared settings are `dvc remote add -d origin s3://dvc` and `dvc remote modify origin endpointurl https://dagshub.com/<dagshub-user>/mlops-lab-1.s3`. Keep `access_key_id` and `secret_access_key` in `--local` configuration, using your token as instructed by DagsHub. Commit the updated dependency files and `.dvc/config`, then push Git and DVC. Do not mix the HTTP and S3 authentication options.

After future data changes:

```powershell
uv run dvc add data
git add data.dvc .gitignore src/food11/data.py pyproject.toml uv.lock
git commit -m "Update Food-11 data and preparation"
git push
uv run dvc push
```

## References

- https://doc.dvc.org/command-reference/init
- https://doc.dvc.org/command-reference/config
- https://doc.dvc.org/command-reference/pull
- https://doc.dvc.org/command-reference/checkout
- https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
- https://pytorch.org/hub/pytorch_vision_resnet/
- https://dagshub.com/docs/integration_guide/dvc/
