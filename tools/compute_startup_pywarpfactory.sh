#!/bin/bash
set -euxo pipefail

exec > >(tee /var/log/pywarpfactory-startup.log) 2>&1

cleanup() {
  status=$?
  set +e
  echo "pyWarpFactory startup exit status: ${status}; shutting down VM."
  if command -v python >/dev/null 2>&1 && python -c "import google.cloud.storage" >/dev/null 2>&1; then
    python - <<PY || true
from google.cloud import storage

output_uri = "${OUTPUT_URI}".rstrip("/")
if output_uri.startswith("gs://"):
    bucket_name, _, prefix = output_uri[5:].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/compute-startup.log")
    blob.upload_from_filename("/var/log/pywarpfactory-startup.log")
    print(f"Uploaded gs://{bucket_name}/{prefix}/compute-startup.log")
PY
  fi
  shutdown -h now || true
}
trap cleanup EXIT

PACKAGE_URI="__PACKAGE_URI__"
OUTPUT_URI="__OUTPUT_URI__"
PROFILE="__PROFILE__"
STATIC_FLAG="__STATIC_FLAG__"
V_WARP="__V_WARP__"
NUM_VECS="__NUM_VECS__"
SAVE_ARRAYS="__SAVE_ARRAYS__"

echo "pyWarpFactory Compute startup"
date -u

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-pip python3-venv

python3 -m venv /opt/pywarpfactory-venv
source /opt/pywarpfactory-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install google-cloud-storage

mkdir -p /opt/pywarpfactory

python - <<PY
from google.cloud import storage

uri = "${PACKAGE_URI}"
if not uri.startswith("gs://"):
    raise SystemExit(f"PACKAGE_URI must start with gs://, got {uri}")

bucket_name, _, path = uri[5:].partition("/")
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(path)
blob.download_to_filename("/opt/pywarpfactory/pywarpfactory.tar.gz")
print(f"Downloaded {uri}")
PY

python -m pip install /opt/pywarpfactory/pywarpfactory.tar.gz

ARGS=(
  -m warpfactory.cloud.run_job
  --execution-target local
  --profile "${PROFILE}"
  --v-warp "${V_WARP}"
  --num-vecs "${NUM_VECS}"
  --energy-condition-method warpfactory
  --solver-method christoffel
  --local-output-dir /opt/pywarpfactory/output
  --output-uri "${OUTPUT_URI}"
)

if [ "${STATIC_FLAG}" = "true" ]; then
  ARGS+=(--static)
fi
if [ "${SAVE_ARRAYS}" = "true" ]; then
  ARGS+=(--save-arrays)
fi

python "${ARGS[@]}"

python - <<PY
from google.cloud import storage

output_uri = "${OUTPUT_URI}".rstrip("/")
bucket_name, _, prefix = output_uri[5:].partition("/")
client = storage.Client()
bucket = client.bucket(bucket_name)
for local_name in ("summary.json", "arrays.npz"):
    path = f"/opt/pywarpfactory/output/{local_name}"
    try:
        with open(path, "rb"):
            pass
    except FileNotFoundError:
        continue
    blob = bucket.blob(f"{prefix}/compute-{local_name}")
    blob.upload_from_filename(path)
    print(f"Uploaded gs://{bucket_name}/{prefix}/compute-{local_name}")
PY

echo "pyWarpFactory Compute job completed"
date -u
