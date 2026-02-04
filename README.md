# Biosecurity × AI Risk Explorer

Interactive tool for exploring when AI model safeguards matter for biosecurity based on compute cost curves, model sizes, and safeguard resistance to adversarial fine-tuning.

## Live Demo

[Deploy to get your link]

## Quick Deploy Options

### Option 1: Vercel (Recommended - 2 minutes)

1. Push this folder to a GitHub repository
2. Go to [vercel.com](https://vercel.com) and sign in with GitHub
3. Click "New Project" → Import your repo
4. Vercel auto-detects Vite, just click "Deploy"
5. Get your `https://yourproject.vercel.app` link

### Option 2: Netlify (Also easy)

1. Push to GitHub
2. Go to [netlify.com](https://netlify.com)
3. "New site from Git" → Select repo
4. Build command: `npm run build`
5. Publish directory: `dist`
6. Deploy

### Option 3: GitHub Pages

1. Push to GitHub
2. Run locally: `npm install && npm run build`
3. Push the `dist` folder to a `gh-pages` branch
4. Enable Pages in repo settings

### Option 4: CodeSandbox (Instant, no GitHub needed)

1. Go to [codesandbox.io](https://codesandbox.io)
2. Create new sandbox → Upload this folder
3. Get instant shareable link

## Local Development

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173`

## Build for Production

```bash
npm run build
```

Output in `dist/` folder.

## What This Tool Does

Explores the question: **When do model safeguards stop mattering for biosecurity?**

Key features:
- **Crux 1**: Who can train a bio foundation model from scratch? (cost trajectories over time)
- **Crux 2**: How robust must safeguards be? (fine-tuning steps → dollars)
- **Chokepoint comparison**: Safeguards vs other barriers (lab setup, DNA synthesis, etc.)
- **Physical limits**: Decaying cost curves as we approach CMOS ceilings
- **Model details**: Full methodology explanation with references

## Key References

- Deep Ignorance (O'Brien, Casper et al., 2025) - arxiv:2508.06601
- Epoch AI hardware efficiency analysis
- Cost calibration from public cloud GPU pricing

## License

MIT - Use freely, attribution appreciated.
