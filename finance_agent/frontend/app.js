/* =========================================================
   Finance Agent — Frontend Logic
   ========================================================= */

const API_BASE = "/api";

// ── Utilities ──────────────────────────────────────────────

/**
 * Formats a float as a percentage string.
 * Handles both decimal (0.12 → 12.00%) and already-whole (12.5 → 12.50%) values.
 */
function formatPercent(val, alreadyWhole = false) {
    if (val == null || isNaN(val)) return 'N/A';
    const pct = alreadyWhole ? val : val * 100;
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${pct.toFixed(1)}%`;
}

function formatReturnClass(val) {
    if (val == null) return '';
    return val >= 0 ? 'pos-ret' : 'neg-ret';
}

function scoreClass(score) {
    if (score >= 7) return 'high';
    if (score >= 5) return 'mid';
    return 'low';
}

// ── Toast ───────────────────────────────────────────────────

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? 'fa-circle-check' : 'fa-circle-exclamation';
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 400);
    }, 3500);
}

// ── Navigation ──────────────────────────────────────────────

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');

        const viewId = item.getAttribute('data-view') + '-view';
        document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));
        document.getElementById(viewId).classList.add('active');

        if (viewId === 'dashboard-view') loadDashboard();
        if (viewId === 'watchlist-view') loadWatchlist();
        if (viewId === 'journal-view')   loadJournal();
    });
});

// ── Dashboard ───────────────────────────────────────────────

async function loadDashboard() {
    try {
        const [statsRes, predsRes] = await Promise.all([
            fetch(`${API_BASE}/journal/stats`),
            fetch(`${API_BASE}/journal/predictions`)
        ]);
        const stats = await statsRes.json();
        const preds = await predsRes.json();

        // Stats
        const hitRate = stats.hit_rate;
        document.getElementById('stat-hit-rate').textContent =
            hitRate != null ? `${(hitRate * 100).toFixed(0)}%` : '--';
        document.getElementById('stat-brier').textContent =
            stats.brier_score != null ? stats.brier_score.toFixed(3) : '--';
        document.getElementById('stat-total').textContent = stats.total_predictions || 0;
        document.getElementById('stat-pending').textContent = stats.pending_reviews || 0;

        // Recent predictions table
        const tbody = document.getElementById('recent-predictions-body');
        tbody.innerHTML = '';

        if (!preds.length) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:32px;">No analyses yet. Add stocks to your watchlist and run an analysis!</td></tr>`;
            return;
        }

        preds.slice(0, 8).forEach(p => {
            const tr = document.createElement('tr');
            tr.onclick = () => openJournalEntry(p.id);
            const actualStr = p.actual_return != null ? formatPercent(p.actual_return) : '—';
            const actualCls = p.actual_return != null ? formatReturnClass(p.actual_return) : '';
            const score = p.expected_return != null
                ? `<span style="font-family:var(--font-mono);font-size:12px;color:var(--text-dim)">—</span>`
                : `<span style="color:var(--text-muted)">—</span>`;

            tr.innerHTML = `
                <td><strong style="font-family:var(--font-mono)">${p.ticker}</strong></td>
                <td style="color:var(--text-muted);font-size:13px">${p.date}</td>
                <td><span class="badge ${p.rating.toLowerCase()}">${p.rating}</span></td>
                <td style="font-family:var(--font-mono);font-size:13px">${score}</td>
                <td>${p.confidence != null ? (p.confidence * 100).toFixed(0) + '%' : '—'}</td>
                <td style="font-size:12px;color:var(--text-muted)">${p.review_status || '—'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error('Dashboard load failed:', e);
    }
}

// ── Journal ─────────────────────────────────────────────────

let _allPredictions = [];

async function loadJournal() {
    try {
        const res = await fetch(`${API_BASE}/journal/predictions`);
        _allPredictions = await res.json();
        renderJournalTable(_allPredictions);
    } catch (e) {
        console.error('Journal load failed:', e);
    }
}

function renderJournalTable(preds) {
    const tbody = document.getElementById('journal-table-body');
    tbody.innerHTML = '';

    if (!preds.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:40px;">No predictions recorded yet.</td></tr>`;
        return;
    }

    preds.forEach(p => {
        const tr = document.createElement('tr');
        tr.onclick = () => openJournalEntry(p.id);
        const actualStr = p.actual_return != null ? formatPercent(p.actual_return) : '—';
        const actualCls = p.actual_return != null ? formatReturnClass(p.actual_return) : '';
        const score = p.expected_return != null
            ? `<span style="font-family:var(--font-mono)">${p.expected_return != null ? Number(p.expected_return).toFixed(2) : '—'}</span>`
            : '—';

        tr.innerHTML = `
            <td style="color:var(--text-muted);font-size:12px;">#${p.seq || '?'}</td>
            <td><strong style="font-family:var(--font-mono)">${p.ticker}</strong></td>
            <td style="color:var(--text-muted);font-size:13px">${p.date}</td>
            <td style="font-size:12px;color:var(--text-dim)">${p.horizon || '—'}</td>
            <td><span class="badge ${p.rating.toLowerCase()}">${p.rating}</span></td>
            <td>${score}</td>
            <td class="${actualCls}">${actualStr}</td>
            <td style="font-size:12px;color:var(--text-muted)">${p.review_status || '—'}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Journal search
document.getElementById('journal-search').addEventListener('input', (e) => {
    const q = e.target.value.trim().toUpperCase();
    if (!q) return renderJournalTable(_allPredictions);
    renderJournalTable(_allPredictions.filter(p => p.ticker.includes(q)));
});

// ── Watchlist ────────────────────────────────────────────────

let _lastAnalyses = {};

async function loadWatchlist() {
    try {
        const [stocksRes, lastRes] = await Promise.all([
            fetch(`${API_BASE}/watchlist`),
            fetch(`${API_BASE}/watchlist/last-analyses`)
        ]);
        const stocks = await stocksRes.json();
        _lastAnalyses = await lastRes.json();

        const grid = document.getElementById('watchlist-grid');
        grid.innerHTML = '';

        if (!stocks.length) {
            grid.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-chart-bar"></i>
                    <h3>Your watchlist is empty</h3>
                    <p>Add a stock ticker above to start tracking and analyzing companies.</p>
                </div>`;
            return;
        }

        stocks.forEach(s => renderStockCard(s, grid));
    } catch (e) {
        console.error('Watchlist load failed:', e);
        showToast('Failed to load watchlist', 'error');
    }
}

function renderStockCard(s, grid) {
    const last = _lastAnalyses[s.ticker];
    const card = document.createElement('div');
    card.className = 'stock-card glass';
    card.id = `card-${s.ticker}`;

    const lastRatingHtml = last
        ? `<span class="badge ${last.rating.toLowerCase()}">${last.rating}</span>`
        : `<span class="badge gray">Not analyzed</span>`;

    const lastScoreHtml = last && last.score != null
        ? `<span class="score-mini">Score: <span class="score-val">${Number(last.score).toFixed(1)}</span>/10</span>`
        : '';

    const lastDateHtml = last
        ? `<span class="meta-tag">${last.date}</span>`
        : '';

    card.innerHTML = `
        <div class="stock-card-top">
            <div class="stock-card-title">
                <h2>${s.ticker}</h2>
                <p>${s.name || s.ticker}</p>
            </div>
            <div class="stock-last-rating">${lastRatingHtml}</div>
        </div>
        <div class="stock-card-meta">
            <span class="meta-tag">${s.sector || 'Unknown'}</span>
            ${lastDateHtml}
            ${lastScoreHtml}
        </div>
        <div class="stock-actions">
            <button class="btn btn-analyze" id="btn-analyze-${s.ticker}" onclick="startAnalysis('${s.ticker}')">
                <i class="fa-solid fa-microchip"></i> Analyze
            </button>
            <button class="btn btn-remove" onclick="removeStock('${s.ticker}')" title="Remove from watchlist">
                <i class="fa-solid fa-trash"></i>
            </button>
        </div>
    `;
    grid.appendChild(card);
}

// Add stock
document.getElementById('add-ticker-btn').addEventListener('click', async () => {
    const input = document.getElementById('new-ticker-input');
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) return;

    const btn = document.getElementById('add-ticker-btn');
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Adding...`;

    try {
        const res = await fetch(`${API_BASE}/watchlist/${ticker}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'added') {
            showToast(`Added ${ticker} to watchlist`);
            input.value = '';
            loadWatchlist();
        } else if (data.status === 'already_exists') {
            showToast(`${ticker} is already in your watchlist`, 'error');
        }
    } catch (e) {
        showToast(`Failed to add ${ticker}`, 'error');
    }

    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-plus"></i> Add Stock`;
});

// Allow Enter key on ticker input
document.getElementById('new-ticker-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('add-ticker-btn').click();
});

// Remove stock
window.removeStock = async (ticker) => {
    if (!confirm(`Remove ${ticker} from watchlist?`)) return;
    try {
        await fetch(`${API_BASE}/watchlist/${ticker}`, { method: 'DELETE' });
        showToast(`Removed ${ticker}`);
        loadWatchlist();
    } catch (e) {
        showToast(`Failed to remove ${ticker}`, 'error');
    }
};

// ── Analysis Drawer ──────────────────────────────────────────

const drawerOverlay = document.getElementById('drawer-overlay');
const closeDrawerBtn = document.getElementById('close-drawer');

function openDrawer(title) {
    document.getElementById('drawer-title').textContent = title;
    document.getElementById('analysis-loading').classList.remove('hidden');
    document.getElementById('analysis-report').classList.add('hidden');
    drawerOverlay.classList.remove('hidden');
}

function closeDrawer() {
    drawerOverlay.classList.add('hidden');
}

closeDrawerBtn.addEventListener('click', closeDrawer);

// Close on overlay click (outside drawer)
drawerOverlay.addEventListener('click', (e) => {
    if (e.target === drawerOverlay) closeDrawer();
});

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !drawerOverlay.classList.contains('hidden')) closeDrawer();
});

// Navigate to journal from drawer
document.getElementById('btn-view-journal')?.addEventListener('click', () => {
    closeDrawer();
    document.getElementById('nav-journal').click();
});

// ── Analyze ─────────────────────────────────────────────────

window.startAnalysis = async (ticker) => {
    openDrawer(`Analyzing ${ticker}...`);

    // Disable the button on the card
    const btn = document.getElementById(`btn-analyze-${ticker}`);
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running...`;
    }

    const horizonSelect = document.getElementById('global-horizon');
    const selectedHorizon = horizonSelect ? horizonSelect.value : '12M';

    try {
        const res = await fetch(`${API_BASE}/analyze/${ticker}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizon: selectedHorizon, persist: true })
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || 'Analysis failed');
        }

        const data = await res.json();
        document.getElementById('drawer-title').textContent = `${ticker} — Analysis Complete`;
        renderReport(ticker, data);
        loadDashboard();

        // Refresh watchlist card with new score
        await loadWatchlist();

    } catch (e) {
        closeDrawer();
        showToast(`Analysis failed: ${e.message}`, 'error');
        console.error('Analysis error:', e);
    }

    if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-microchip"></i> Analyze`;
    }
};

// ── Open Journal Entry ───────────────────────────────────────

window.openJournalEntry = async (id) => {
    openDrawer('Loading entry...');

    try {
        const res = await fetch(`${API_BASE}/journal/predictions/${id}`);
        if (!res.ok) throw new Error('Prediction not found');
        const data = await res.json();
        document.getElementById('drawer-title').textContent = `${data.ticker} — ${data.date || 'Historical Entry'}`;
        renderReport(data.ticker, data);
    } catch (e) {
        closeDrawer();
        showToast(`Failed to load entry: ${e.message}`, 'error');
    }
};

// ── Render Report ─────────────────────────────────────────────

function renderReport(ticker, data) {
    document.getElementById('analysis-loading').classList.add('hidden');
    document.getElementById('analysis-report').classList.remove('hidden');

    // Hero
    const score = data.weighted_score;
    const scoreEl = document.getElementById('rep-score');
    scoreEl.textContent = score != null ? score.toFixed(1) : '--';
    scoreEl.style.color = score >= 7 ? 'var(--score-high)' : score >= 5 ? 'var(--score-mid)' : 'var(--score-low)';

    document.getElementById('rep-ticker').textContent = ticker;

    const ratingEl = document.getElementById('rep-rating');
    ratingEl.textContent = data.rating;
    ratingEl.className = `badge ${(data.rating || '').toLowerCase()}`;

    document.getElementById('rep-horizon').textContent = data.horizon || '12M';

    const dateEl = document.getElementById('rep-date-badge');
    dateEl.textContent = data.analysis_date || data.date || '';
    dateEl.className = data.analysis_date || data.date ? 'badge gray' : 'hidden';

    if (data.runtime_seconds) {
        document.getElementById('rep-runtime').textContent = `Runtime: ${data.runtime_seconds.toFixed(1)}s`;
    } else {
        document.getElementById('rep-runtime').textContent = '';
    }

    // Thesis
    const thesisEl = document.getElementById('rep-thesis');
    thesisEl.innerHTML = (data.thesis || '—').replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');

    // Bull / Bear case
    const bullText = document.getElementById('rep-bull-text');
    const bearText = document.getElementById('rep-bear-text');
    const bullBearSection = document.getElementById('bull-bear-section');

    if (data.bull_case_summary || data.bear_case_summary) {
        bullText.textContent = data.bull_case_summary || 'Not available';
        bearText.textContent = data.bear_case_summary || 'Not available';
        bullBearSection.classList.remove('hidden');
    } else {
        bullBearSection.classList.add('hidden');
    }

    // Scenario bars
    const sm = data.scenarios || {};

    function renderScenarioBar(key, barId, retId) {
        const s = sm[key] || {};
        const prob = s.probability != null ? s.probability : 0;
        const ret  = s.return != null ? s.return : (s.expected_return != null ? s.expected_return : null);

        // Probabilities are 0–1 floats
        const probPct = prob <= 1 ? prob * 100 : prob;
        document.getElementById(barId).style.width = `${Math.min(probPct, 100)}%`;

        const retEl = document.getElementById(retId);
        if (ret != null) {
            // Returns can be 0-1 floats or already percent — use formatPercent
            const retPct = Math.abs(ret) <= 1 ? ret * 100 : ret;
            const sign = retPct >= 0 ? '+' : '';
            retEl.textContent = `${sign}${retPct.toFixed(1)}% (${Math.round(probPct)}%)`;
            retEl.className = `scenario-ret ${retPct >= 0 ? 'pos' : 'neg'}`;
        } else {
            retEl.textContent = '—';
            retEl.className = 'scenario-ret';
        }
    }

    renderScenarioBar('bull', 'bar-bull', 'ret-bull');
    renderScenarioBar('base', 'bar-base', 'ret-base');
    renderScenarioBar('bear', 'bar-bear', 'ret-bear');

    const ev = sm.expected_value;
    const evEl = document.getElementById('rep-ev');
    if (ev != null) {
        const evPct = Math.abs(ev) <= 1 ? ev * 100 : ev;
        const sign = evPct >= 0 ? '+' : '';
        evEl.textContent = `${sign}${evPct.toFixed(1)}%`;
        evEl.className = `ev-val ${evPct >= 0 ? 'pos' : 'neg'}`;
    } else {
        evEl.textContent = '—';
        evEl.className = 'ev-val';
    }

    // Invalidation criteria (risks)
    const risksUl = document.getElementById('rep-risks');
    const criteria = data.invalidation_criteria || [];
    risksUl.innerHTML = criteria.length
        ? criteria.map(r => `<li>${r}</li>`).join('')
        : `<li style="color:var(--text-muted)">No specific invalidation criteria provided.</li>`;

    // Disagreement
    const disagWarn = document.getElementById('rep-disagreement');
    const disagScore = data.disagreement?.score;
    if (disagScore != null && disagScore > 3.0) {
        disagWarn.classList.remove('hidden');
    } else {
        disagWarn.classList.add('hidden');
    }

    // Agents
    const agentsDiv = document.getElementById('rep-agents');
    const agentOutputs = data.agent_outputs || {};
    agentsDiv.innerHTML = '';

    if (Object.keys(agentOutputs).length === 0) {
        agentsDiv.innerHTML = `<p style="color:var(--text-muted);font-size:13px">Agent breakdown not stored for historical entries.</p>`;
    } else {
        // Sort by score descending
        const sorted = Object.entries(agentOutputs).sort((a,b) => (b[1].score || 0) - (a[1].score || 0));

        sorted.forEach(([name, r]) => {
            const sc = r.score || 0;
            const cls = scoreClass(sc);
            const barWidth = Math.min((sc / 10) * 100, 100);
            const conf = r.confidence != null ? ` · ${(r.confidence * 100).toFixed(0)}% conf.` : '';

            const card = document.createElement('div');
            card.className = 'agent-card';
            card.innerHTML = `
                <div class="agent-card-header" onclick="this.nextElementSibling.classList.toggle('open')">
                    <div class="agent-card-header-left">
                        <span class="agent-name">${name.replace(/([A-Z])/g, ' $1').trim()}</span>
                        <span class="agent-confidence">${conf}</span>
                    </div>
                    <span class="agent-score-badge ${cls}">${sc.toFixed(1)}/10</span>
                </div>
                <div class="agent-score-bar-bg">
                    <div class="agent-score-bar-fill ${cls}" style="width:${barWidth}%"></div>
                </div>
                <div class="agent-card-body">
                    <p class="agent-summary">${r.summary || 'No summary available.'}</p>
                    ${r.key_findings && r.key_findings.length > 0
                        ? `<ul class="agent-findings">${r.key_findings.map(f => `<li>${f}</li>`).join('')}</ul>`
                        : ''}
                </div>
            `;
            agentsDiv.appendChild(card);
        });
    }

    // Peer context (if available in data)
    const peerSection = document.getElementById('peer-section');
    const peerGrid = document.getElementById('rep-peers');
    if (data.peer_context && Object.keys(data.peer_context).length > 0) {
        peerGrid.innerHTML = '';
        Object.entries(data.peer_context).forEach(([t, ret]) => {
            const retPct = Math.abs(ret) <= 1 ? ret * 100 : ret;
            const sign = retPct >= 0 ? '+' : '';
            const retCls = retPct >= 0 ? 'pos' : 'neg';
            const isSubject = t === ticker;
            const chip = document.createElement('div');
            chip.className = `peer-chip${isSubject ? ' subject' : ''}`;
            chip.innerHTML = `
                <span class="peer-ticker">${t}</span>
                <span class="peer-ret ${retCls}">${sign}${retPct.toFixed(1)}%</span>
                <span style="font-size:10px;color:var(--text-muted)">1Y</span>
            `;
            peerGrid.appendChild(chip);
        });
        peerSection.classList.remove('hidden');
    } else {
        peerSection.classList.add('hidden');
    }
}

// ── Initial Load ─────────────────────────────────────────────

loadDashboard();

// ── Screener ─────────────────────────────────────────────────

async function loadScreenerIndices() {
    try {
        const res = await fetch(`${API_BASE}/screener/indices`);
        const indices = await res.json();
        const sel = document.getElementById('screener-index');
        sel.innerHTML = '';
        indices.forEach(idx => {
            const opt = document.createElement('option');
            opt.value = idx.id;
            opt.textContent = `${idx.name} (~${idx.size} stocks)`;
            // Default to NASDAQ 100 for speed
            if (idx.id === 'nasdaq100') opt.selected = true;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load indices:', e);
    }
}

loadScreenerIndices();

document.getElementById('screener-run-btn').addEventListener('click', runScreener);

async function runScreener() {
    const index   = document.getElementById('screener-index').value;
    const horizon = document.getElementById('screener-horizon').value;
    const topN    = parseInt(document.getElementById('screener-topn').value);

    const btn = document.getElementById('screener-run-btn');
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Scanning...`;

    // Reset UI
    document.getElementById('screener-idle').classList.add('hidden');
    document.getElementById('screener-results-wrap').classList.add('hidden');
    document.getElementById('screener-progress-wrap').classList.remove('hidden');
    document.getElementById('screener-progress-fill').style.width = '0%';
    document.getElementById('screener-progress-text').textContent = 'Fetching index...';
    document.getElementById('screener-progress-count').textContent = '0 / ?';
    document.getElementById('screener-results-body').innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/screener/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index, horizon, top_n: topN })
        });

        if (!response.ok) throw new Error('Screener request failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const event = JSON.parse(line.slice(6));
                    handleScreenerEvent(event, topN);
                } catch (_) {}
            }
        }
    } catch (e) {
        showToast(`Screener failed: ${e.message}`, 'error');
        document.getElementById('screener-idle').classList.remove('hidden');
        document.getElementById('screener-progress-wrap').classList.add('hidden');
    }

    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-radar"></i> Scan Index`;
}

function handleScreenerEvent(event, topN) {
    if (event.type === 'start') {
        document.getElementById('screener-progress-count').textContent = `0 / ${event.total}`;
    }

    if (event.type === 'progress') {
        const pct = Math.round((event.done / event.total) * 100);
        document.getElementById('screener-progress-fill').style.width = pct + '%';
        document.getElementById('screener-progress-text').textContent =
            `Analyzing ${event.ticker}${event.ok ? '' : ' (skipped)'}`;
        document.getElementById('screener-progress-count').textContent =
            `${event.done} / ${event.total}`;
    }

    if (event.type === 'results') {
        document.getElementById('screener-progress-wrap').classList.add('hidden');
        document.getElementById('screener-results-wrap').classList.remove('hidden');

        const selectedIndex = document.getElementById('screener-index');
        const indexName = selectedIndex.options[selectedIndex.selectedIndex]?.text || '';

        document.getElementById('screener-results-title').textContent =
            `Top ${event.data.length} Stocks — ${indexName}`;
        document.getElementById('screener-results-meta').textContent =
            `Quantitative scan · ${new Date().toLocaleTimeString()}`;

        renderScreenerResults(event.data);
    }
}

function renderScreenerResults(stocks) {
    const tbody = document.getElementById('screener-results-body');
    tbody.innerHTML = '';

    if (!stocks.length) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;padding:32px;color:var(--text-muted)">No results — try a different index or horizon.</td></tr>`;
        return;
    }

    stocks.forEach((s, i) => {
        const sc = s.screener_score || 0;
        const cls = sc >= 7 ? 'high' : sc >= 5 ? 'mid' : 'low';

        const ev = s.expected_value;
        let evStr = '—', evCls = '';
        if (ev != null) {
            const evPct = Math.abs(ev) <= 1 ? ev * 100 : ev;
            evStr = (evPct >= 0 ? '+' : '') + evPct.toFixed(1) + '%';
            evCls = evPct >= 0 ? 'pos-ret' : 'neg-ret';
        }

        const pf = s.piotroski;
        let pfHtml = '—';
        if (pf != null) {
            const pfCls = pf >= 7 ? 'high' : pf >= 4 ? 'mid' : 'low';
            pfHtml = `<span class="piotroski-badge ${pfCls}">${pf}</span>`;
        }

        const mom = s.momentum_1y;
        let momStr = '—', momCls = '';
        if (mom != null) {
            const momPct = Math.abs(mom) <= 1 ? mom * 100 : mom;
            momStr = (momPct >= 0 ? '+' : '') + momPct.toFixed(1) + '%';
            momCls = momPct >= 0 ? 'pos-ret' : 'neg-ret';
        }

        const fpe = s.forward_pe;
        const fpeStr = fpe != null ? fpe.toFixed(1) + 'x' : '—';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="screener-rank">${i + 1}</td>
            <td><strong style="font-family:var(--font-mono);font-size:14px">${s.ticker}</strong></td>
            <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;color:var(--text-dim)">${s.name || '—'}</td>
            <td style="font-size:12px;color:var(--text-muted)">${s.sector || '—'}</td>
            <td><span class="screener-score-pill ${cls}">${sc.toFixed(1)}</span></td>
            <td class="${evCls}" style="font-family:var(--font-mono);font-size:13px">${evStr}</td>
            <td>${pfHtml}</td>
            <td class="${momCls}" style="font-family:var(--font-mono);font-size:13px">${momStr}</td>
            <td style="font-family:var(--font-mono);font-size:13px;color:var(--text-dim)">${fpeStr}</td>
            <td>
                <button class="btn btn-screener-analyze" onclick="startAnalysisFromScreener('${s.ticker}')">
                    <i class="fa-solid fa-microchip"></i> Analyze
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

window.startAnalysisFromScreener = (ticker) => {
    // Switch to Watchlist view context but open the drawer directly
    startAnalysis(ticker);
    // Also ensure the ticker gets added to watchlist if not already there
    fetch(`${API_BASE}/watchlist/${ticker}`, { method: 'POST' }).catch(() => {});
};
