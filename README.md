# Parkinson's Gait Analysis — research prototype

Flask application for a video-based Parkinson's gait classification experiment. **Not a medical device and not suitable for diagnosis or clinical decisions.**

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY='replace-with-a-long-random-secret'
python -m backend.server
```

The local entrypoint attempts to train the SVM from the small bundled example videos. There are only **three samples per class**, and no validated generalisation metrics; the classifier must not be represented as accurate or diagnostic. The sample data also needs provenance, licensing, and clinical-label review before distribution.

## Vercel caveats

`api/index.py` and `vercel.json` expose the Flask application for Vercel Python functions. This is only an experimental UI deployment: training is not performed on serverless cold start. The `/predict` endpoint returns HTTP 503 when no validated model is loaded, rather than reporting a bogus prediction. SQLite data, sessions, and uploads are not persistent across Vercel instances. Replace the database with a managed service, persist video files in object storage or combine upload/prediction in one request, use a managed model inference service, and configure `SECRET_KEY` and `DATABASE_URL` securely before public use. Avoid uploading patient-identifiable gait videos.

## Known limitations

* The project is an academic prototype, not clinically validated.
* Local training occurs only from the `__main__` entrypoint.
* Login backed by SQLite is not production-ready on Vercel.
* Frame features are padded/truncated at 1000 elements, discarding most optical-flow data; requires experimental redesign and evaluation.
