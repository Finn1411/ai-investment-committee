/* =========================================================
   Finance Agent — App Logic v2
   ========================================================= */

const API = '/api';

// ── Utilities ──────────────────────────────────────────────

function fmtPct(val) {
    if (val == null || isNaN(val)) return '—';
    const p = Math.abs(val) <= 1 ? val * 100 : val;
    return (p >= 0 ? '+' : '') + p.toFixed(1) + '%';
}
function fmtRetClass(val) {
    if (val == null) return '';
    const p = Math.abs(val) <= 1 ? val * 100 : val;
    return p >= 0 ? 'pos' : 'neg';
}
function scoreClass(s) { return s >= 7 ? 'hi' : s >= 5 ? 'mid' : 'lo'; }

function fmtPrice(p, currency) {
    if (p == null) return '—';
    const sym = currency === 'EUR' ? '€' : currency === 'GBP' ? '£' : '$';
    return sym + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Toast ───────────────────────────────────────────────────

function toast(msg, type = 'success') {
    const c = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `<i class="fa-solid ${type === 'success' ? 'fa-circle-check' : 'fa-circle-exclamation'}"></i> ${msg}`;
    c.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 350); }, 3500);
}

// ── Navigation ──────────────────────────────────────────────

const views = ['watchlist', 'analyzer', 'journal', 'screener'];

function navigateTo(view) {
    views.forEach(v => {
        document.getElementById(`${v}-view`).classList.toggle('active', v === view);
        document.getElementById(`nav-${v}`).classList.toggle('active', v === view);
    });
    if (view === 'watchlist') loadWatchlist();
    if (view === 'journal')   loadJournal();
    if (view === 'analyzer')  loadAnalyzerRecents();
}

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
        e.preventDefault();
        navigateTo(item.dataset.view);
    });
});

// ── Sidebar Stats ────────────────────────────────────────────

async function loadSidebarStats() {
    try {
        const [statsRes, watchRes] = await Promise.all([
            fetch(`${API}/journal/stats`),
            fetch(`${API}/watchlist`)
        ]);
        const stats = await statsRes.json();
        const watch = await watchRes.json();

        document.getElementById('sb-total').textContent = stats.total_predictions ?? '0';
        document.getElementById('sb-watchlist').textContent = watch.length ?? '0';
        document.getElementById('sb-hit').textContent =
            stats.avg_score != null ? stats.avg_score.toFixed(1) + '/10' : '—';
    } catch (_) {}
}

// ══════════════════════════════════════════════════════════
// VIEW: WATCHLIST
// ══════════════════════════════════════════════════════════

let _stocks = [];
let _lastAnalyses = {};
let _prices = {};

async function loadWatchlist() {
    try {
        const [stocksRes, lastRes] = await Promise.all([
            fetch(`${API}/watchlist`),
            fetch(`${API}/watchlist/last-analyses`)
        ]);
        _stocks = await stocksRes.json();
        _lastAnalyses = await lastRes.json();

        renderWatchlistTable();

        // Fetch prices async — update when ready
        fetchPrices();
    } catch (e) {
        console.error('Watchlist load failed:', e);
        toast('Failed to load watchlist', 'error');
    }
}

async function fetchPrices() {
    const refreshBtn = document.getElementById('refresh-prices-btn');
    const icon = refreshBtn.querySelector('i');
    icon.classList.add('spinning');
    try {
        const res = await fetch(`${API}/watchlist/prices`);
        _prices = await res.json();
        updatePriceColumns();
    } catch (_) {}
    icon.classList.remove('spinning');
}

function renderWatchlistTable() {
    const tbody = document.getElementById('watchlist-tbody');
    const empty = document.getElementById('watchlist-empty');

    if (!_stocks.length) {
        tbody.innerHTML = '';
        empty.classList.remove('hidden');
        document.getElementById('watchlist-table').style.display = 'none';
        return;
    }

    empty.classList.add('hidden');
    document.getElementById('watchlist-table').style.display = '';

    tbody.innerHTML = _stocks.map(s => {
        const last = _lastAnalyses[s.ticker];
        const ratingHtml = last
            ? `<span class="badge ${last.rating.toLowerCase()}">${last.rating}</span>`
            : `<span style="color:var(--text-muted);font-size:12px">—</span>`;

        let scoreHtml = '<span style="color:var(--text-muted);font-size:12px">—</span>';
        if (last && last.score != null) {
            const sc = Number(last.score);
            const cls = scoreClass(sc);
            const pct = (sc / 10 * 100).toFixed(0);
            scoreHtml = `
                <div class="score-bar-cell">
                    <div class="score-bar-bg"><div class="score-bar-fill ${cls}" style="width:${pct}%"></div></div>
                    <span class="score-num-cell ${cls}">${sc.toFixed(1)}</span>
                </div>`;
        }

        const dateHtml = last
            ? `<span class="wl-date">${last.date}</span>`
            : `<span class="wl-never">Never</span>`;

        return `<tr id="wl-row-${s.ticker}">
            <td><span class="wl-ticker">${s.ticker}</span></td>
            <td><span class="wl-name">${s.name || s.ticker}</span></td>
            <td><span class="wl-sector">${s.sector || '—'}</span></td>
            <td class="col-price" id="wl-price-${s.ticker}"><span class="wl-price" style="color:var(--text-muted)">—</span></td>
            <td class="col-price" id="wl-change-${s.ticker}"><span class="wl-change" style="color:var(--text-muted)">—</span></td>
            <td>${ratingHtml}</td>
            <td class="col-score">${scoreHtml}</td>
            <td class="col-date">${dateHtml}</td>
            <td class="col-actions">
                <button class="btn btn-wl-analyze" onclick="startAnalysisFromWatchlist('${s.ticker}')">
                    <i class="fa-solid fa-microchip"></i> Analyze
                </button>
                <button class="btn btn-wl-remove" onclick="removeStock('${s.ticker}')" title="Remove from watchlist">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </td>
        </tr>`;
    }).join('');
}

function updatePriceColumns() {
    _stocks.forEach(s => {
        const pdata = _prices[s.ticker];
        if (!pdata) return;
        const priceEl  = document.getElementById(`wl-price-${s.ticker}`);
        const changeEl = document.getElementById(`wl-change-${s.ticker}`);
        if (priceEl)  priceEl.innerHTML  = `<span class="wl-price mono">${fmtPrice(pdata.price)}</span>`;
        if (changeEl) {
            const cls = fmtRetClass(pdata.change_pct);
            const pct = fmtPct(pdata.change_pct);
            const arrow = pdata.change_pct >= 0 ? '▲' : '▼';
            changeEl.innerHTML = `<span class="wl-change ${cls}">${arrow} ${pct}</span>`;
        }
    });
}

// Add stock
document.getElementById('new-ticker-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('add-ticker-btn').click();
});

document.getElementById('add-ticker-btn').addEventListener('click', async () => {
    const input = document.getElementById('new-ticker-input');
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) return;

    const btn = document.getElementById('add-ticker-btn');
    btn.disabled = true; btn.textContent = 'Adding...';

    try {
        const res = await fetch(`${API}/watchlist/${ticker}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'added') {
            toast(`Added ${ticker} — ${data.name}`);
            input.value = '';
            await loadWatchlist();
            loadSidebarStats();
        } else {
            toast(`${ticker} is already in your watchlist`, 'error');
        }
    } catch { toast(`Failed to add ${ticker}`, 'error'); }

    btn.disabled = false; btn.textContent = 'Add to Watchlist';
});

// Refresh prices button
document.getElementById('refresh-prices-btn').addEventListener('click', fetchPrices);

// Remove
window.removeStock = async (ticker) => {
    if (!confirm(`Remove ${ticker} from watchlist?`)) return;
    await fetch(`${API}/watchlist/${ticker}`, { method: 'DELETE' });
    toast(`Removed ${ticker}`);
    loadWatchlist();
    loadSidebarStats();
};

// Analyze from watchlist
window.startAnalysisFromWatchlist = (ticker) => {
    openDrawer(ticker);
    runAnalysis(ticker, document.getElementById('analyzer-horizon')?.value || '12M', true);
};


// ══════════════════════════════════════════════════════════
// VIEW: ANALYZER
// ══════════════════════════════════════════════════════════

const RECENTS_KEY = 'fa_recent_tickers';

function getRecents() {
    try { return JSON.parse(localStorage.getItem(RECENTS_KEY)) || []; } catch { return []; }
}

function pushRecent(ticker) {
    let r = getRecents().filter(t => t !== ticker);
    r.unshift(ticker);
    r = r.slice(0, 8);
    localStorage.setItem(RECENTS_KEY, JSON.stringify(r));
}

function loadAnalyzerRecents() {
    const recents = getRecents();
    const wrap = document.getElementById('analyzer-recents-wrap');
    const chips = document.getElementById('analyzer-recents');
    if (!recents.length) { wrap.classList.add('hidden'); return; }
    wrap.classList.remove('hidden');
    chips.innerHTML = recents.map(t =>
        `<span class="recent-chip" onclick="analyzeFromChip('${t}')">${t}</span>`
    ).join('');
}

window.analyzeFromChip = (ticker) => {
    document.getElementById('analyzer-ticker-input').value = ticker;
    document.getElementById('analyzer-run-btn').click();
};

document.getElementById('analyzer-ticker-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('analyzer-run-btn').click();
});

document.getElementById('analyzer-run-btn').addEventListener('click', async () => {
    const ticker = document.getElementById('analyzer-ticker-input').value.trim().toUpperCase();
    if (!ticker) { toast('Please enter a ticker symbol', 'error'); return; }

    const horizon = document.getElementById('analyzer-horizon').value;
    const persist = document.getElementById('analyzer-persist').value === 'true';

    pushRecent(ticker);
    loadAnalyzerRecents();

    openDrawer(ticker);
    await runAnalysis(ticker, horizon, persist);
});


// ══════════════════════════════════════════════════════════
// VIEW: JOURNAL
// ══════════════════════════════════════════════════════════

let _allPredictions = [];

async function loadJournal() {
    try {
        const [statsRes, predsRes] = await Promise.all([
            fetch(`${API}/journal/stats`),
            fetch(`${API}/journal/predictions`)
        ]);
        const stats = await statsRes.json();
        _allPredictions = await predsRes.json();

        // Stats bar — populated from real DB aggregates
        document.getElementById('j-total').textContent    = stats.total_predictions ?? '—';
        document.getElementById('j-avg-score').textContent = stats.avg_score != null ? stats.avg_score.toFixed(1) + '/10' : '—';
        document.getElementById('j-avg-conf').textContent  = stats.avg_confidence != null ? stats.avg_confidence.toFixed(0) + '%' : '—';
        document.getElementById('j-buy-rate').textContent  = stats.buy_rate != null ? stats.buy_rate.toFixed(0) + '%' : '—';

        renderJournal(_allPredictions);
    } catch (e) { console.error('Journal load error:', e); }
}

function renderJournal(preds) {
    const tbody = document.getElementById('journal-table-body');
    if (!preds.length) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:40px;color:var(--text-muted)">No predictions yet. Run an analysis to get started.</td></tr>`;
        return;
    }
    tbody.innerHTML = preds.map((p, i) => {
        const score    = p.expected_return != null ? Number(p.expected_return).toFixed(2) : '—';
        const wScore   = p.weighted_score != null ? Number(p.weighted_score).toFixed(1) : '—';
        const expRet   = p.expected_return != null ? fmtPct(p.expected_return) : '—';
        const expCls   = p.expected_return != null ? fmtRetClass(p.expected_return) : '';
        return `<tr onclick="openJournalEntry('${p.id}')" title="Click to view full report">
            <td style="color:var(--text-muted);font-size:11px">#${p.seq || (i+1)}</td>
            <td><strong class="mono">${p.ticker}</strong></td>
            <td style="font-size:12px;color:var(--text-muted)">${p.date}</td>
            <td style="font-size:12px;color:var(--text-dim)">${p.horizon || '—'}</td>
            <td><span class="badge ${p.rating.toLowerCase()}">${p.rating}</span></td>
            <td class="mono" style="font-size:12px">${wScore}</td>
            <td class="${expCls} mono" style="font-size:12px">${expRet}</td>
        </tr>`;
    }).join('');
}

// Search + filter
function applyJournalFilters() {
    const q   = document.getElementById('journal-search').value.trim().toUpperCase();
    const rat = document.getElementById('journal-rating-filter').value;
    let preds = _allPredictions;
    if (q)   preds = preds.filter(p => p.ticker.includes(q));
    if (rat) preds = preds.filter(p => p.rating === rat);
    renderJournal(preds);
}

document.getElementById('journal-search').addEventListener('input', applyJournalFilters);
document.getElementById('journal-rating-filter').addEventListener('change', applyJournalFilters);

// Clear journal
document.getElementById('journal-clear-btn').addEventListener('click', async () => {
    if (!confirm('Delete ALL journal entries? This cannot be undone.')) return;
    try {
        const res = await fetch(`${API}/journal/clear`, { method: 'DELETE' });
        const data = await res.json();
        toast(`Cleared ${data.deleted} journal entries`);
        _allPredictions = [];
        renderJournal([]);
        loadSidebarStats();
        // Reset journal stats
        ['j-total','j-avg-score','j-avg-conf','j-buy-rate'].forEach(id => {
            document.getElementById(id).textContent = '—';
        });
    } catch (e) { toast('Failed to clear journal', 'error'); }
});

window.openJournalEntry = async (id) => {
    openDrawer('Loading...');
    try {
        const res = await fetch(`${API}/journal/predictions/${id}`);
        if (!res.ok) throw new Error('Not found');
        const data = await res.json();
        document.getElementById('drawer-title').textContent = `${data.ticker}`;
        document.getElementById('drawer-subtitle').textContent = `${data.analysis_date || ''} · Historical Entry`;
        renderReport(data.ticker, data);
    } catch (e) {
        closeDrawer();
        toast(`Failed to load entry`, 'error');
    }
};


// ══════════════════════════════════════════════════════════
// DRAWER
// ══════════════════════════════════════════════════════════

let _currentDrawerTicker = null;

function openDrawer(ticker) {
    _currentDrawerTicker = ticker;
    document.getElementById('drawer-title').textContent = ticker;
    document.getElementById('drawer-subtitle').textContent = '';
    document.getElementById('analysis-loading').classList.remove('hidden');
    document.getElementById('analysis-report').classList.add('hidden');
    document.getElementById('drawer-overlay').classList.remove('hidden');
}

function closeDrawer() {
    document.getElementById('drawer-overlay').classList.add('hidden');
    _currentDrawerTicker = null;
}

document.getElementById('close-drawer').addEventListener('click', closeDrawer);
document.getElementById('drawer-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('drawer-overlay')) closeDrawer();
});
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeDrawer();
});

// Add to watchlist from drawer
document.getElementById('drawer-add-watchlist').addEventListener('click', async () => {
    if (!_currentDrawerTicker) return;
    const btn = document.getElementById('drawer-add-watchlist');
    btn.disabled = true;
    try {
        const res = await fetch(`${API}/watchlist/${_currentDrawerTicker}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'added') {
            toast(`Added ${_currentDrawerTicker} to watchlist`);
            loadSidebarStats();
            if (document.getElementById('watchlist-view').classList.contains('active')) loadWatchlist();
        } else {
            toast(`${_currentDrawerTicker} is already in watchlist`);
        }
    } catch { toast('Failed to add to watchlist', 'error'); }
    btn.disabled = false;
});

// View in journal
document.getElementById('btn-view-journal').addEventListener('click', () => {
    closeDrawer();
    navigateTo('journal');
});


// ══════════════════════════════════════════════════════════
// ANALYSIS
// ══════════════════════════════════════════════════════════

async function runAnalysis(ticker, horizon, persist) {
    // Disable analyze buttons
    const wlBtn = document.getElementById(`btn-analyze-wl-${ticker}`);
    if (wlBtn) { wlBtn.disabled = true; }

    try {
        const res = await fetch(`${API}/analyze/${ticker}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizon, persist })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || 'Analysis failed');
        }
        const data = await res.json();
        document.getElementById('drawer-title').textContent = ticker;
        document.getElementById('drawer-subtitle').textContent = `${horizon} · ${data.analysis_date || ''}`;
        _currentDrawerTicker = ticker;
        renderReport(ticker, data);

        // Refresh watchlist data + sidebar stats
        if (persist) {
            loadSidebarStats();
            _lastAnalyses = (await (await fetch(`${API}/watchlist/last-analyses`)).json());
            renderWatchlistTable();
            updatePriceColumns();
        }

    } catch (e) {
        closeDrawer();
        toast(`Analysis failed: ${e.message}`, 'error');
    }

    if (wlBtn) wlBtn.disabled = false;
}


// ══════════════════════════════════════════════════════════
// RENDER REPORT
// ══════════════════════════════════════════════════════════

function renderReport(ticker, data) {
    document.getElementById('analysis-loading').classList.add('hidden');
    document.getElementById('analysis-report').classList.remove('hidden');

    // Hero
    const score = data.weighted_score;
    const scoreEl = document.getElementById('rep-score');
    const hasScore = score != null && !isNaN(score);
    scoreEl.textContent = hasScore ? Number(score).toFixed(1) : '--';
    scoreEl.style.color = !hasScore ? 'var(--text-muted)' : score >= 7 ? 'var(--score-hi)' : score >= 5 ? 'var(--score-mid)' : 'var(--score-lo)';

    document.getElementById('rep-ticker').textContent = ticker;

    const rEl = document.getElementById('rep-rating');
    rEl.textContent = data.rating || '—';
    rEl.className   = `badge ${(data.rating || '').toLowerCase()}`;

    document.getElementById('rep-horizon').textContent = data.horizon || '—';

    const dateEl = document.getElementById('rep-date-badge');
    const d = data.analysis_date || data.date || '';
    dateEl.textContent = d;
    dateEl.className   = d ? 'badge gray' : 'hidden';

    document.getElementById('rep-runtime').textContent =
        data.runtime_seconds ? `Runtime: ${Number(data.runtime_seconds).toFixed(1)}s` : '';

    // Thesis
    const t = data.thesis || '—';
    document.getElementById('rep-thesis').innerHTML =
        t.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');

    // Bull / Bear
    const bullBear = document.getElementById('bull-bear-section');
    if (data.bull_case_summary || data.bear_case_summary) {
        document.getElementById('rep-bull-text').textContent = data.bull_case_summary || '—';
        document.getElementById('rep-bear-text').textContent = data.bear_case_summary || '—';
        bullBear.classList.remove('hidden');
    } else {
        bullBear.classList.add('hidden');
    }

    // Scenarios
    const sm = data.scenarios || {};
    function setBar(key, barId, retId) {
        const s    = sm[key] || {};
        const prob = s.probability ?? 0;
        const ret  = s.return ?? s.expected_return ?? null;
        const probPct = prob <= 1 ? prob * 100 : prob;
        document.getElementById(barId).style.width = Math.min(probPct, 100) + '%';
        const retEl = document.getElementById(retId);
        if (ret != null) {
            const rp = Math.abs(ret) <= 1 ? ret * 100 : ret;
            retEl.textContent = `${rp >= 0 ? '+' : ''}${rp.toFixed(1)}% (${Math.round(probPct)}%)`;
            retEl.className   = `scenario-ret ${rp >= 0 ? 'pos' : 'neg'}`;
        } else { retEl.textContent = '—'; retEl.className = 'scenario-ret'; }
    }
    setBar('bull', 'bar-bull', 'ret-bull');
    setBar('base', 'bar-base', 'ret-base');
    setBar('bear', 'bar-bear', 'ret-bear');

    const ev = sm.expected_value;
    const evEl = document.getElementById('rep-ev');
    if (ev != null) {
        const ep = Math.abs(ev) <= 1 ? ev * 100 : ev;
        evEl.textContent = `${ep >= 0 ? '+' : ''}${ep.toFixed(1)}%`;
        evEl.className   = `ev-val ${ep >= 0 ? 'pos' : 'neg'}`;
    } else { evEl.textContent = '—'; evEl.className = 'ev-val'; }

    // Risks
    const ul = document.getElementById('rep-risks');
    const risks = data.invalidation_criteria || [];
    ul.innerHTML = risks.length
        ? risks.map(r => `<li>${r}</li>`).join('')
        : `<li style="color:var(--text-muted)">No invalidation criteria specified.</li>`;

    // Disagreement
    const dis = document.getElementById('rep-disagreement');
    dis.classList.toggle('hidden', !(data.disagreement?.score > 3.0));

    // Agents
    const agDiv = document.getElementById('rep-agents');
    const ao    = data.agent_outputs || {};
    agDiv.innerHTML = '';
    if (!Object.keys(ao).length) {
        agDiv.innerHTML = `<p style="color:var(--text-muted);font-size:13px">Agent breakdown not stored for historical entries.</p>`;
    } else {
        Object.entries(ao)
            .sort((a, b) => (b[1].score || 0) - (a[1].score || 0))
            .forEach(([name, r]) => {
                const sc  = r.score || 0;
                const cls = scoreClass(sc);
                const pct = Math.min((sc / 10) * 100, 100);
                const conf = r.confidence != null ? ` · ${(r.confidence * 100).toFixed(0)}% conf.` : '';
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.innerHTML = `
                    <div class="agent-card-header" onclick="this.nextElementSibling.nextElementSibling.classList.toggle('open')">
                        <div class="agent-card-header-left">
                            <span class="agent-name">${name.replace(/([A-Z])/g,' $1').trim()}</span>
                            <span class="agent-confidence">${conf}</span>
                        </div>
                        <span class="agent-score-badge ${cls}">${sc.toFixed(1)}/10</span>
                    </div>
                    <div class="agent-score-bar-bg"><div class="agent-score-bar-fill ${cls}" style="width:${pct}%"></div></div>
                    <div class="agent-card-body">
                        <p class="agent-summary">${r.summary || 'No summary.'}</p>
                        ${r.key_findings?.length ? `<ul class="agent-findings">${r.key_findings.map(f=>`<li>${f}</li>`).join('')}</ul>` : ''}
                    </div>`;
                agDiv.appendChild(card);
            });
    }

    // Peers
    const peerSec = document.getElementById('peer-section');
    const peerGrid = document.getElementById('rep-peers');
    if (data.peer_context && Object.keys(data.peer_context).length > 0) {
        peerGrid.innerHTML = '';
        Object.entries(data.peer_context).forEach(([t, ret]) => {
            const rp = Math.abs(ret) <= 1 ? ret * 100 : ret;
            const chip = document.createElement('div');
            chip.className = `peer-chip${t === ticker ? ' subject' : ''}`;
            chip.innerHTML = `
                <span class="peer-ticker">${t}</span>
                <span class="peer-ret ${rp >= 0 ? 'pos' : 'neg'}">${rp >= 0 ? '+' : ''}${rp.toFixed(1)}%</span>
                <span style="font-size:10px;color:var(--text-muted)">1Y</span>`;
            peerGrid.appendChild(chip);
        });
        peerSec.classList.remove('hidden');
    } else { peerSec.classList.add('hidden'); }
}


// ══════════════════════════════════════════════════════════
// VIEW: SCREENER
// ══════════════════════════════════════════════════════════

async function loadScreenerIndices() {
    try {
        const res = await fetch(`${API}/screener/indices`);
        const indices = await res.json();
        const sel = document.getElementById('screener-index');
        sel.innerHTML = '';
        indices.forEach(idx => {
            const opt = document.createElement('option');
            opt.value = idx.id;
            opt.textContent = `${idx.name} (~${idx.size} stocks)`;
            if (idx.id === 'nasdaq100') opt.selected = true;
            sel.appendChild(opt);
        });
    } catch (e) { console.error('Failed to load indices:', e); }
}

document.getElementById('screener-run-btn').addEventListener('click', runScreener);

async function runScreener() {
    const index   = document.getElementById('screener-index').value;
    const horizon = document.getElementById('screener-horizon').value;
    const topN    = parseInt(document.getElementById('screener-topn').value);
    const btn     = document.getElementById('screener-run-btn');

    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Scanning...`;

    document.getElementById('screener-idle').classList.add('hidden');
    document.getElementById('screener-results-wrap').classList.add('hidden');
    document.getElementById('screener-progress-wrap').classList.remove('hidden');
    document.getElementById('screener-progress-fill').style.width = '0%';
    document.getElementById('screener-progress-text').textContent = 'Fetching index...';
    document.getElementById('screener-progress-count').textContent = '0 / ?';
    document.getElementById('screener-results-body').innerHTML = '';

    try {
        const response = await fetch(`${API}/screener/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index, horizon, top_n: topN })
        });
        if (!response.ok) throw new Error('Screener failed');

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try { handleScreenerEvent(JSON.parse(line.slice(6))); } catch (_) {}
            }
        }
    } catch (e) {
        toast(`Screener failed: ${e.message}`, 'error');
        document.getElementById('screener-idle').classList.remove('hidden');
        document.getElementById('screener-progress-wrap').classList.add('hidden');
    }

    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-radar"></i> Scan Index`;
}

function handleScreenerEvent(ev) {
    if (ev.type === 'start') {
        document.getElementById('screener-progress-count').textContent = `0 / ${ev.total}`;
    }
    if (ev.type === 'progress') {
        const pct = Math.round((ev.done / ev.total) * 100);
        document.getElementById('screener-progress-fill').style.width = pct + '%';
        document.getElementById('screener-progress-text').textContent =
            `Scoring ${ev.ticker}${ev.ok ? '' : ' (skipped)'}`;
        document.getElementById('screener-progress-count').textContent = `${ev.done} / ${ev.total}`;
    }
    if (ev.type === 'results') {
        document.getElementById('screener-progress-wrap').classList.add('hidden');
        document.getElementById('screener-results-wrap').classList.remove('hidden');
        const sel = document.getElementById('screener-index');
        document.getElementById('screener-results-title').textContent =
            `Top ${ev.data.length} — ${sel.options[sel.selectedIndex]?.text || ''}`;
        document.getElementById('screener-results-meta').textContent =
            `Quant scan · ${new Date().toLocaleTimeString()}`;
        renderScreenerResults(ev.data);
    }
}

function renderScreenerResults(stocks) {
    const tbody = document.getElementById('screener-results-body');
    if (!stocks.length) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;padding:32px;color:var(--text-muted)">No results.</td></tr>`;
        return;
    }
    tbody.innerHTML = stocks.map((s, i) => {
        const sc  = s.screener_score || 0;
        const cls = sc >= 7 ? 'hi' : sc >= 5 ? 'mid' : 'lo';

        // Earnings Yield (EBIT/EV) — better than scenario EV% for quick screening
        const eyRaw = s.earnings_yield;
        const eyStr = eyRaw != null ? (eyRaw * 100).toFixed(1) + '%' : '—';
        const eyCls = eyRaw != null ? fmtRetClass(eyRaw) : '';

        // Piotroski badge
        const pf = s.piotroski;
        const pfHtml = pf != null
            ? `<span class="piotroski-badge ${pf >= 7 ? 'hi' : pf >= 4 ? 'mid' : 'lo'}">${pf}</span>`
            : '—';

        // Momentum 12-1 (AQR standard), fallback to return_1y
        const momRaw = s.momentum_12_1 ?? s.return_1y;
        const momStr = momRaw != null ? fmtPct(momRaw) : '—';
        const momCls = momRaw != null ? fmtRetClass(momRaw) : '';

        // ROIC
        const roicRaw = s.roic;
        const roicStr = roicRaw != null ? (roicRaw * 100).toFixed(1) + '%' : '—';
        const roicCls = roicRaw != null ? fmtRetClass(roicRaw) : '';

        // Beneish M-Score fraud flag
        const bm = s.beneish_m;
        const bmFlag = bm != null && bm > -1.78
            ? `<span title="Beneish M=${bm.toFixed(2)}: earnings manipulation risk" style="color:var(--danger);font-size:11px">&#9888;</span>`
            : '';

        return `<tr>
            <td class="screener-rank">${i+1}</td>
            <td><strong class="mono" style="font-size:14px">${s.ticker}</strong>${bmFlag}</td>
            <td style="font-size:11px;color:var(--text-muted)">${s.sector || '—'}</td>
            <td><span class="screener-score-pill ${cls}">${sc.toFixed(1)}</span></td>
            <td class="${eyCls} mono" style="font-size:13px">${eyStr}</td>
            <td>${pfHtml}</td>
            <td class="${momCls} mono" style="font-size:13px">${momStr}</td>
            <td class="${roicCls} mono" style="font-size:12px">${roicStr}</td>
            <td style="display:flex;gap:6px">
                <button class="btn btn-sc-analyze" onclick="analyzeFromScreener('${s.ticker}')">
                    <i class="fa-solid fa-microchip"></i> Analyze
                </button>
                <button class="btn btn-sc-add" onclick="addToWatchlistFromScreener('${s.ticker}')" title="Add to watchlist">
                    <i class="fa-solid fa-plus"></i>
                </button>
            </td>
        </tr>`;
    }).join('');
}

window.analyzeFromScreener = (ticker) => {
    // Analyze without auto-journaling (persist=false) — user can save manually
    openDrawer(ticker);
    runAnalysis(ticker, document.getElementById('screener-horizon').value, false);
};

window.addToWatchlistFromScreener = async (ticker) => {
    try {
        const res  = await fetch(`${API}/watchlist/${ticker}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'added') {
            toast(`Added ${ticker} to watchlist`);
            loadSidebarStats();
        }
    } catch (_) {}
};


// ══════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════

loadScreenerIndices();
loadSidebarStats();
loadWatchlist();
