# Frontend

React + Vite UI for the Next Word Predictor. Deployed on Vercel.

## Running locally

```bash
npm install
npm run dev
```

Copy `.env.example` to `.env` and point it at your backend:
```
VITE_API_BASE_URL=http://127.0.0.1:8000
```
If unset, it defaults to `http://127.0.0.1:8000` (see `src/api/client.js`).

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Start the Vite dev server |
| `npm run build` | Production build (output to `dist/`) |
| `npm run lint` | ESLint |
| `npm run preview` | Preview the production build locally |

## Structure

```
src/
  api/client.js              fetch wrapper for POST /predict
  components/PredictForm.jsx   input form
  components/PredictionList.jsx  renders returned predictions
  App.jsx
```

## Deployment

Deployed on Vercel, building `npm run build` with `VITE_API_BASE_URL` set to the production backend URL (Railway). The backend's CORS config (`NWP_CORS_ORIGINS` on Railway) must include the exact Vercel URL for requests to succeed.

## CI

`.github/workflows/frontend-ci.yml` runs `npm ci`, `npm run lint`, and `npm run build` on every push/PR touching this directory — see the root README for the full CI/CD picture. There's currently no frontend test suite (no vitest/jest); lint + build-check is the current coverage.
