# Fantasy Baseball Draft Helper

A statistically-driven rotisserie draft optimization tool that uses Standings Gain Points (SGP), Monte Carlo simulation, and real-time ESPN integration to recommend optimal draft picks.

## Features

- **SGP Engine**: Converts raw projections to Standings Gain Points — the gold standard for rotisserie valuation. Properly handles rate stats (AVG, ERA, WHIP) with volume weighting.
- **Positional Scarcity**: Computes Value Above Replacement (VAR) at each position. Catchers and shortstops get appropriate scarcity premiums.
- **ADP-Aware Recommendations**: Discounts players likely to be available at your next pick. Won't tell you to reach for a player no one else is targeting.
- **Monte Carlo Draft Simulation**: Simulates thousands of remaining-draft scenarios using opponent modeling (softmax/Boltzmann selection on ADP) to find picks that maximize total team SGP.
- **Pitcher Streaming Analysis**: Quantifies the benefit of rostering extra SPs to rotate through lineup slots. Simulates weekly start decisions to estimate ERA/WHIP improvement.
- **ESPN Integration**: Connect to your ESPN league with browser cookies. Poll for live draft picks (best-effort — ESPN's live draft API is limited) or enter picks manually.
- **Real-Time WebSocket Updates**: Frontend auto-updates as picks are recorded. Recommendations recalculate after every pick.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
# Open http://localhost:8000
```

## How It Works

### 1. Upload Projections

Download CSV projections from FanGraphs (Steamer, ZiPS, ATC, or THE BAT). Upload separate hitter and pitcher files. Optionally upload ADP data.

### 2. Configure League

Set your team count, draft position, number of rounds, and draft type (snake/linear).

### 3. Get Recommendations

The engine computes for each available player:

- **SGP Total**: Sum of standings gain points across all 10 rotisserie categories
- **VAR**: Value Above Replacement — SGP minus the replacement-level player at that position
- **Recommendation Score**: VAR adjusted for urgency (will this player be gone by your next pick?) and positional need

### 4. Draft in Real Time

Click "Draft" next to any player, or type a name to record picks. The board recalculates after every pick. Connect to ESPN for automatic pick detection.

## Statistical Models

### Standings Gain Points (SGP)

For counting stats: `SGP = projected_stat / SGP_denominator`

For rate stats (volume-adjusted):
- `SGP_AVG = (AVG - league_avg) * AB / (5500 * SGP_denom)`
- `SGP_ERA = -(ERA - league_ERA) * IP / (1300 * SGP_denom)`

### Positional Scarcity

Replacement level at each position = SGP of the (N+1)th best player, where N = league_size * roster_slots_at_position.

### ADP Availability Model

`P(available at pick p) = 1 / (1 + exp(0.8 * (p - ADP) / ADP_std))`

Players with high availability at your next pick get urgency discounted.

### Monte Carlo Draft Simulation

Opponent picks modeled with softmax selection:
`P(opponent drafts player i) = exp(-rank_i / temperature) / Σ exp(-rank_j / temperature)`

For each candidate, simulates the remaining draft N times, using greedy best-VAR for your picks and softmax-ADP for opponents.

### Pitcher Streaming

Simulates a full season of weekly start decisions. With M pitchers for K slots (M > K), selects the K best matchups each week. Compares ERA/WHIP/W/K against a fixed-rotation baseline and converts the improvement to SGP.

## ESPN Integration

1. Log into ESPN Fantasy Baseball in your browser
2. Open DevTools → Application → Cookies → espn.com
3. Copy `espn_s2` and `SWID` values
4. Paste into the ESPN tab in the app

Note: ESPN's REST API may not return picks in real-time during a live draft. The app supports both automatic polling and manual pick entry as a fallback.

## Project Structure

```
├── main.py                          # FastAPI app entry point
├── requirements.txt                 # Python dependencies
├── backend/
│   ├── engine/
│   │   ├── config.py                # League settings, category definitions
│   │   ├── sgp.py                   # Standings Gain Points calculations
│   │   ├── monte_carlo.py           # Monte Carlo draft simulation
│   │   ├── pitcher_streaming.py     # SP rotation value analysis
│   │   └── draft_optimizer.py       # Central orchestrator
│   ├── data/
│   │   └── projections.py           # CSV loader, FanGraphs format support
│   ├── espn/
│   │   └── client.py                # ESPN API client (auth, polling)
│   └── api/
│       └── routes.py                # REST + WebSocket endpoints
├── frontend/
│   ├── templates/
│   │   └── index.html               # Main UI (Alpine.js)
│   └── static/
│       ├── css/style.css             # Dark theme styles
│       └── js/app.js                 # Frontend logic
└── sample_data/
    ├── sample_hitters.csv            # Example hitter projections
    ├── sample_pitchers.csv           # Example pitcher projections
    └── sample_adp.csv               # Example ADP data
```

## Tech Stack

- **Backend**: FastAPI (async, native WebSocket support)
- **Frontend**: Alpine.js (15KB, no build step)
- **Statistics**: NumPy, Pandas, SciPy
- **ESPN**: httpx (async HTTP client)
