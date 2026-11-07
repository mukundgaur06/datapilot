# Deploying DataPilot

DataPilot ships as **one Docker image**: it builds the React app and serves it from the FastAPI
backend, so a deployment is a single web service with a single URL. Any host that runs a
Dockerfile works. Measured footprint: about 175 MB of RAM idle; training a model on the sample
datasets adds about 200 MB. A 512 MB instance is enough for the samples, with little headroom for
large uploads.

## Before you deploy: the LLM quota

Every visitor's question spends **your** Gemini quota. On the free tier that's about 20
requests per model per day, shared by everyone, and a question takes 2-4 requests. So a free-tier
demo handles roughly 5-10 questions a day in total; after that, visitors see "Daily quota used
up". For a demo you'll share widely, use a paid key (check Google's pricing page for current costs),
and keep the rate limits on (they are by default).

## Option A: Render (simplest; deploys from GitHub)

1. Sign in at [render.com](https://render.com) with GitHub, then **New > Web Service** and pick
   this repository. Render detects the `Dockerfile`.
2. Instance type: **Free** works (it sleeps after ~15 minutes idle; the first visit after that
   takes about a minute to wake).
3. **Environment variables**: add `GEMINI_API_KEY` (mark it secret). Optional: the
   `DATAPILOT_*` settings from the README.
4. **Health check path**: `/health`.
5. Deploy. Render builds the image and gives you a URL such as `https://datapilot-xyz.onrender.com`.
   Pushes to `main` redeploy automatically.

Render sets `$PORT` itself; the image listens on it.

## Option B: Hugging Face Spaces (more memory on the free tier)

Note: a **public** Space exposes its files, so your source code becomes public (your GitHub
repository is private). A private Space hides the code but also the app.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) with
   **SDK: Docker** (blank template).
2. In the Space's **Settings > Variables and secrets**, add the secret `GEMINI_API_KEY`.
3. A Space reads its settings from the YAML header of its own `README.md`. Push this repository
   to the Space, with this header at the top of the README that the Space receives:

   ```yaml
   ---
   title: DataPilot
   sdk: docker
   app_port: 8000
   ---
   ```

   (Keep that header out of the GitHub README by committing it on a separate `hf` branch and
   pushing that branch to the Space's `main`: `git push hf hf:main`.)

## Any other Docker host

```bash
docker build -t datapilot .
docker run -d -p 80:8000 -e GEMINI_API_KEY=your-key --restart unless-stopped datapilot
```

Behind your own reverse proxy (nginx, Caddy), make sure it forwards `X-Forwarded-For`; the
container trusts it so per-IP rate limits see real visitors. If you expose the container
directly to the internet without a proxy, override its start command so it trusts only
localhost, and clients can't spoof their IP:

```bash
docker run -d -p 80:8000 -e GEMINI_API_KEY=your-key datapilot \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips 127.0.0.1
```

## Serving the frontend separately (optional)

To host the UI on Vercel or Netlify and the API elsewhere: build the frontend with
`VITE_API_URL=https://your-api.example.com`, and set `DATAPILOT_CORS_ORIGINS` on the API to the
UI's origin. The single-container setup avoids all of this and is recommended.

## After deploying: checklist

- `https://your-url/health` returns `{"ok": true, ..., "llm_configured": true}`.
- Upload `sample_data/sms_spam.csv` and click a suggested question; an answer and a chart appear.
- Try it from a phone and in a private window (no cached state).
- Traces live in `/tmp` inside the container and disappear on restart; nothing about visitors
  is stored permanently.
