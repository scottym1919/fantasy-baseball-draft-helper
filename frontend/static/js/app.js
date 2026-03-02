/**
 * Fantasy Baseball Draft Helper — Frontend Application
 *
 * Vanilla JS with Alpine.js for reactivity.
 * Connects to the FastAPI backend via REST + WebSocket.
 */

document.addEventListener("alpine:init", () => {
    Alpine.data("draftApp", () => ({
        // ── State ──────────────────────────────────────────────
        // Setup
        setupComplete: false,
        numTeams: 12,
        myTeamIdx: 0,
        numRounds: 23,
        snakeDraft: true,

        // Projection files
        hittersFile: null,
        pitchersFile: null,
        adpFile: null,
        projectionSystem: "custom",
        projectionsLoaded: false,
        playersLoaded: 0,

        // ESPN
        espnConnected: false,
        espnPolling: false,
        espnLeagueId: "",
        espnSeason: 2025,
        espnS2: "",
        espnSwid: "",
        espnLeagueName: "",

        // Draft state
        currentPick: 1,
        currentRound: 1,
        isMyPick: false,
        totalSgp: 0,
        picks: [],
        myRoster: [],
        categoryTotals: {},

        // Recommendations
        recommendations: [],
        loadingRecs: false,
        useMonteCarlo: false,

        // Streaming analysis
        streamingAnalysis: null,

        // Keeper assistant
        keeperTiers: [],
        keeperTierName: "",
        keeperTierMax: 1,
        keeperTierRoundCost: "",
        keeperRosterFile: null,
        keeperTeamIdx: 0,
        keeperCandidates: [],
        keeperRecommendation: null,
        keeperMaxTotal: "",
        keeperApplied: false,

        // Player search
        searchQuery: "",
        searchResults: [],
        pickPlayerName: "",

        // UI
        activeTab: "setup",
        rightTab: "log",
        ws: null,
        toasts: [],

        // ── Initialization ─────────────────────────────────────
        init() {
            this.connectWebSocket();
        },

        // ── WebSocket ──────────────────────────────────────────
        connectWebSocket() {
            const protocol = window.location.protocol === "https:" ? "wss" : "ws";
            this.ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

            this.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                this.handleWsMessage(msg);
            };

            this.ws.onclose = () => {
                setTimeout(() => this.connectWebSocket(), 3000);
            };
        },

        handleWsMessage(msg) {
            switch (msg.type) {
                case "state":
                    this.updateDraftState(msg.summary);
                    break;
                case "pick":
                case "espn_pick":
                    this.addPickToLog(msg);
                    this.refreshDraftState();
                    this.refreshRecommendations();
                    this.showToast(
                        `Pick ${msg.pick || ""}: ${msg.player_name}`,
                        msg.type === "espn_pick" ? "info" : "success"
                    );
                    break;
                case "undo":
                    this.refreshDraftState();
                    this.refreshRecommendations();
                    break;
                case "recommendations":
                    this.recommendations = msg.recommendations;
                    this.loadingRecs = false;
                    break;
            }
        },

        // ── Setup ──────────────────────────────────────────────
        async setupLeague() {
            try {
                const resp = await fetch("/api/setup", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        num_teams: parseInt(this.numTeams),
                        my_team_idx: parseInt(this.myTeamIdx),
                        num_rounds: parseInt(this.numRounds),
                        snake_draft: this.snakeDraft,
                    }),
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.setupComplete = true;
                    this.showToast("League configured", "success");
                } else {
                    this.showToast(data.detail || "Setup failed", "error");
                }
            } catch (e) {
                this.showToast("Setup error: " + e.message, "error");
            }
        },

        // ── Projections ────────────────────────────────────────
        onHittersFile(event) {
            this.hittersFile = event.target.files[0];
        },

        onPitchersFile(event) {
            this.pitchersFile = event.target.files[0];
        },

        onAdpFile(event) {
            this.adpFile = event.target.files[0];
        },

        async uploadProjections() {
            if (!this.setupComplete) {
                this.showToast("Configure league settings first", "error");
                return;
            }

            const formData = new FormData();
            if (this.hittersFile) formData.append("hitters_file", this.hittersFile);
            if (this.pitchersFile) formData.append("pitchers_file", this.pitchersFile);
            if (this.adpFile) formData.append("adp_file", this.adpFile);
            formData.append("projection_system", this.projectionSystem);

            try {
                const resp = await fetch("/api/projections/upload", {
                    method: "POST",
                    body: formData,
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.projectionsLoaded = true;
                    this.playersLoaded = data.players_loaded;
                    this.showToast(`Loaded ${data.players_loaded} players`, "success");
                    this.activeTab = "draft";
                    this.refreshRecommendations();
                } else {
                    this.showToast(data.detail || "Upload failed", "error");
                }
            } catch (e) {
                this.showToast("Upload error: " + e.message, "error");
            }
        },

        // ── Recommendations ────────────────────────────────────
        async refreshRecommendations() {
            if (!this.projectionsLoaded) return;
            this.loadingRecs = true;

            try {
                const params = new URLSearchParams({
                    top_n: "30",
                    monte_carlo: this.useMonteCarlo.toString(),
                    simulations: "500",
                });
                const resp = await fetch(`/api/recommendations?${params}`);
                const data = await resp.json();
                if (resp.ok) {
                    this.recommendations = data.recommendations;
                    this.currentPick = data.pick;
                    this.currentRound = data.round;
                    this.isMyPick = data.is_my_pick;
                }
            } catch (e) {
                console.error("Failed to fetch recommendations:", e);
            }
            this.loadingRecs = false;
        },

        // ── Draft Picks ────────────────────────────────────────
        async draftPlayer(playerId, playerName) {
            try {
                const resp = await fetch("/api/pick", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        team_idx: this.isMyPick ? parseInt(this.myTeamIdx) : this.currentPickingTeam(),
                        player_id: playerId,
                        player_name: playerName,
                    }),
                });
                if (resp.ok) {
                    this.refreshDraftState();
                    this.refreshRecommendations();
                }
            } catch (e) {
                this.showToast("Pick error: " + e.message, "error");
            }
        },

        async draftByName() {
            if (!this.pickPlayerName.trim()) return;

            try {
                const teamIdx = this.isMyPick ? parseInt(this.myTeamIdx) : this.currentPickingTeam();
                const resp = await fetch("/api/pick/search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        team_idx: teamIdx,
                        player_name: this.pickPlayerName,
                    }),
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.pickPlayerName = "";
                    this.refreshDraftState();
                    this.refreshRecommendations();
                } else {
                    this.showToast(data.detail || "Player not found", "error");
                }
            } catch (e) {
                this.showToast("Pick error: " + e.message, "error");
            }
        },

        async undoPick() {
            try {
                await fetch("/api/pick/undo", { method: "POST" });
                this.refreshDraftState();
                this.refreshRecommendations();
            } catch (e) {
                this.showToast("Undo error: " + e.message, "error");
            }
        },

        currentPickingTeam() {
            // Calculate which team is on the clock based on snake draft
            const pick = this.currentPick - 1;
            const round = Math.floor(pick / this.numTeams);
            const pickInRound = pick % this.numTeams;
            if (this.snakeDraft && round % 2 === 1) {
                return this.numTeams - 1 - pickInRound;
            }
            return pickInRound;
        },

        // ── Draft State ────────────────────────────────────────
        async refreshDraftState() {
            try {
                const [summaryResp, rosterResp, picksResp] = await Promise.all([
                    fetch("/api/draft/summary"),
                    fetch("/api/draft/my-roster"),
                    fetch("/api/draft/picks"),
                ]);

                if (summaryResp.ok) {
                    const summary = await summaryResp.json();
                    this.updateDraftState(summary);
                }
                if (rosterResp.ok) {
                    const roster = await rosterResp.json();
                    this.myRoster = roster.roster;
                    this.totalSgp = roster.total_sgp;
                }
                if (picksResp.ok) {
                    const picksData = await picksResp.json();
                    this.picks = picksData.picks;
                }
            } catch (e) {
                console.error("State refresh error:", e);
            }
        },

        updateDraftState(summary) {
            this.currentPick = summary.current_pick;
            this.currentRound = summary.current_round;
            this.isMyPick = summary.is_my_pick;
            this.categoryTotals = summary.category_totals || {};
        },

        addPickToLog(msg) {
            // Will be refreshed via refreshDraftState, but add immediately for UX
        },

        // ── Streaming Analysis ─────────────────────────────────
        async fetchStreamingAnalysis() {
            try {
                const resp = await fetch("/api/streaming");
                if (resp.ok) {
                    this.streamingAnalysis = await resp.json();
                }
            } catch (e) {
                console.error("Streaming analysis error:", e);
            }
        },

        // ── ESPN ───────────────────────────────────────────────
        async connectEspn() {
            try {
                const resp = await fetch("/api/espn/connect", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        league_id: parseInt(this.espnLeagueId),
                        season: parseInt(this.espnSeason),
                        espn_s2: this.espnS2,
                        swid: this.espnSwid,
                    }),
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.espnConnected = true;
                    this.espnLeagueName = data.league_name;
                    this.showToast(`Connected to ESPN: ${data.league_name}`, "success");
                } else {
                    this.showToast(data.detail || "ESPN connection failed", "error");
                }
            } catch (e) {
                this.showToast("ESPN error: " + e.message, "error");
            }
        },

        async startEspnPolling() {
            try {
                const resp = await fetch("/api/espn/start-polling", { method: "POST" });
                if (resp.ok) {
                    this.espnPolling = true;
                    this.showToast("ESPN draft polling started", "info");
                }
            } catch (e) {
                this.showToast("Polling error: " + e.message, "error");
            }
        },

        async stopEspnPolling() {
            try {
                await fetch("/api/espn/stop-polling", { method: "POST" });
                this.espnPolling = false;
                this.showToast("ESPN polling stopped", "info");
            } catch (e) {
                this.showToast("Stop error: " + e.message, "error");
            }
        },

        // ── Player Search ──────────────────────────────────────
        async searchPlayers() {
            if (this.searchQuery.length < 2) {
                this.searchResults = [];
                return;
            }
            try {
                const resp = await fetch(`/api/players/search?q=${encodeURIComponent(this.searchQuery)}&limit=15`);
                if (resp.ok) {
                    const data = await resp.json();
                    this.searchResults = data.results;
                }
            } catch (e) {
                console.error("Search error:", e);
            }
        },

        // ── Keeper Assistant ─────────────────────────────────────
        addKeeperTier() {
            if (!this.keeperTierName.trim()) return;
            this.keeperTiers.push({
                name: this.keeperTierName,
                max_keepers: parseInt(this.keeperTierMax) || 1,
                round_cost: this.keeperTierRoundCost ? parseInt(this.keeperTierRoundCost) : null,
            });
            this.keeperTierName = "";
            this.keeperTierMax = 1;
            this.keeperTierRoundCost = "";
        },

        removeKeeperTier(idx) {
            this.keeperTiers.splice(idx, 1);
        },

        async saveKeeperTiers() {
            if (!this.setupComplete) {
                this.showToast("Configure league settings first", "error");
                return;
            }
            try {
                const resp = await fetch("/api/keeper/tiers", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ tiers: this.keeperTiers }),
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.showToast(`Saved ${this.keeperTiers.length} keeper tiers`, "success");
                } else {
                    this.showToast(data.detail || "Failed to save tiers", "error");
                }
            } catch (e) {
                this.showToast("Tier save error: " + e.message, "error");
            }
        },

        onKeeperRosterFile(event) {
            this.keeperRosterFile = event.target.files[0];
        },

        async uploadKeeperRoster() {
            if (!this.keeperRosterFile) return;
            const formData = new FormData();
            formData.append("roster_file", this.keeperRosterFile);
            formData.append("team_idx", this.keeperTeamIdx.toString());

            try {
                const resp = await fetch("/api/keeper/roster/upload", {
                    method: "POST",
                    body: formData,
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.showToast(`Loaded ${data.players_loaded} players for keeper eval`, "success");
                    this.fetchKeeperCandidates();
                } else {
                    this.showToast(data.detail || "Roster upload failed", "error");
                }
            } catch (e) {
                this.showToast("Upload error: " + e.message, "error");
            }
        },

        async fetchKeeperCandidates() {
            try {
                const resp = await fetch(`/api/keeper/candidates?team_idx=${this.keeperTeamIdx}`);
                const data = await resp.json();
                if (resp.ok) {
                    this.keeperCandidates = data.candidates;
                } else {
                    this.showToast(data.detail || "Failed to load candidates", "error");
                }
            } catch (e) {
                this.showToast("Candidates error: " + e.message, "error");
            }
        },

        async optimizeKeepers() {
            try {
                const body = {
                    team_idx: parseInt(this.keeperTeamIdx),
                };
                if (this.keeperMaxTotal) {
                    body.max_total_keepers = parseInt(this.keeperMaxTotal);
                }
                const resp = await fetch("/api/keeper/optimize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.keeperRecommendation = data;
                    this.showToast("Keeper optimization complete", "success");
                } else {
                    this.showToast(data.detail || "Optimization failed", "error");
                }
            } catch (e) {
                this.showToast("Optimize error: " + e.message, "error");
            }
        },

        async applyKeepers() {
            if (!this.keeperRecommendation) return;
            const keepers = {};
            const playerIds = this.keeperRecommendation.kept_players.map(p => p.player_id);
            keepers[this.keeperTeamIdx.toString()] = playerIds;

            try {
                const resp = await fetch("/api/keeper/apply", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ keepers }),
                });
                const data = await resp.json();
                if (resp.ok) {
                    this.keeperApplied = true;
                    this.showToast(`Applied ${data.keepers_applied} keepers to draft`, "success");
                    this.refreshDraftState();
                    this.refreshRecommendations();
                } else {
                    this.showToast(data.detail || "Apply failed", "error");
                }
            } catch (e) {
                this.showToast("Apply error: " + e.message, "error");
            }
        },

        // ── Utilities ──────────────────────────────────────────
        showToast(message, type = "info") {
            const toast = { message, type, id: Date.now() };
            this.toasts.push(toast);
            setTimeout(() => {
                this.toasts = this.toasts.filter(t => t.id !== toast.id);
            }, 4000);
        },

        urgencyClass(urgency) {
            return `urgency-${urgency}`;
        },

        posClass(position) {
            return `pos-badge pos-${position}`;
        },

        formatNum(n, decimals = 1) {
            if (n === undefined || n === null) return "-";
            return Number(n).toFixed(decimals);
        },
    }));
});
