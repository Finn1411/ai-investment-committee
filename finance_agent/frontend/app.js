const API_BASE = "/api";

// -- Navigation --
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        
        const viewId = item.getAttribute('data-view') + '-view';
        document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));
        document.getElementById(viewId).classList.add('active');
        
        // Refresh data based on view
        if (viewId === 'dashboard-view') loadDashboard();
        if (viewId === 'watchlist-view') loadWatchlist();
        if (viewId === 'journal-view') loadJournal();
    });
});

// -- Toast --
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<i class="fa-solid ${type === 'success' ? 'fa-check-circle' : 'fa-circle-exclamation'}"></i> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// -- Formatting Utilities --
const formatPercent = (val) => val != null ? (val * 100).toFixed(2) + '%' : 'N/A';
const formatClass = (val) => val > 0 ? 'pos-ret' : (val < 0 ? 'neg-ret' : '');

// -- Dashboard --
async function loadDashboard() {
    try {
        const [statsRes, predsRes] = await Promise.all([
            fetch(`${API_BASE}/journal/stats`),
            fetch(`${API_BASE}/journal/predictions`)
        ]);
        const stats = await statsRes.json();
        const preds = await predsRes.json();

        document.getElementById('stat-hit-rate').textContent = formatPercent(stats.hit_rate);
        document.getElementById('stat-brier').textContent = stats.brier_score != null ? stats.brier_score.toFixed(4) : '--';
        document.getElementById('stat-total').textContent = stats.total_predictions || 0;
        document.getElementById('stat-pending').textContent = stats.pending_reviews || 0;

        const tbody = document.getElementById('recent-predictions-body');
        tbody.innerHTML = '';
        preds.slice(0, 5).forEach(p => {
            const tr = document.createElement('tr');
            const ratingClass = p.rating.toLowerCase();
            tr.innerHTML = `
                <td><strong>${p.ticker}</strong></td>
                <td>${p.date}</td>
                <td><span class="badge ${ratingClass}">${p.rating}</span></td>
                <td>${formatPercent(p.confidence)}</td>
                <td>${p.review_status}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Dashboard load failed:", e);
    }
}

// -- Journal --
async function loadJournal() {
    try {
        const res = await fetch(`${API_BASE}/journal/predictions`);
        const preds = await res.json();
        const tbody = document.getElementById('journal-table-body');
        tbody.innerHTML = '';
        preds.forEach(p => {
            const tr = document.createElement('tr');
            tr.style.cursor = 'pointer';
            tr.onclick = () => openJournalEntry(p.id);
            const ratingClass = p.rating.toLowerCase();
            const actualStr = p.actual_return != null ? formatPercent(p.actual_return) : 'Pending';
            const actualCls = p.actual_return != null ? formatClass(p.actual_return) : '';
            tr.innerHTML = `
                <td>#${p.seq}</td>
                <td><strong>${p.ticker}</strong></td>
                <td>${p.date}</td>
                <td><span class="badge ${ratingClass}">${p.rating}</span></td>
                <td class="${actualCls}">${actualStr}</td>
                <td>${p.review_status}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Journal load failed:", e);
    }
}

// -- Watchlist --
async function loadWatchlist() {
    try {
        const res = await fetch(`${API_BASE}/watchlist`);
        const stocks = await res.json();
        const grid = document.getElementById('watchlist-grid');
        grid.innerHTML = '';
        
        stocks.forEach(s => {
            const card = document.createElement('div');
            card.className = 'stock-card glass';
            card.innerHTML = `
                <div class="stock-card-header">
                    <div>
                        <h2>${s.ticker}</h2>
                        <p>${s.name}</p>
                        <p>${s.sector}</p>
                    </div>
                </div>
                <div class="stock-actions">
                    <button class="btn btn-analyze" onclick="startAnalysis('${s.ticker}')">
                        <i class="fa-solid fa-microchip"></i> Analyze
                    </button>
                    <button class="btn btn-remove" onclick="removeStock('${s.ticker}')" title="Remove">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        console.error("Watchlist load failed:", e);
    }
}

document.getElementById('add-ticker-btn').addEventListener('click', async () => {
    const input = document.getElementById('new-ticker-input');
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) return;
    
    input.disabled = true;
    try {
        const res = await fetch(`${API_BASE}/watchlist/${ticker}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'added') {
            showToast(`Added ${ticker} to watchlist`);
            input.value = '';
            loadWatchlist();
        } else if (data.status === 'already_exists') {
            showToast(`${ticker} is already in watchlist`, 'error');
        }
    } catch (e) {
        showToast(`Failed to add ${ticker}`, 'error');
    }
    input.disabled = false;
});

window.removeStock = async (ticker) => {
    if(!confirm(`Remove ${ticker} from watchlist?`)) return;
    try {
        await fetch(`${API_BASE}/watchlist/${ticker}`, { method: 'DELETE' });
        showToast(`Removed ${ticker}`);
        loadWatchlist();
    } catch (e) {
        showToast(`Failed to remove ${ticker}`, 'error');
    }
}

// -- Analysis --
const modal = document.getElementById('analysis-modal');
const closeBtn = document.getElementById('close-modal');

closeBtn.addEventListener('click', () => {
    modal.classList.add('hidden');
});

document.getElementById('btn-view-journal')?.addEventListener('click', () => {
    modal.classList.add('hidden');
    // Simulate click on Journal nav item
    document.querySelector('.nav-item[data-view="journal"]').click();
});

window.startAnalysis = async (ticker) => {
    modal.classList.remove('hidden');
    document.getElementById('modal-title').textContent = `Analyzing ${ticker}...`;
    document.getElementById('analysis-loading').classList.remove('hidden');
    document.getElementById('analysis-report').classList.add('hidden');
    
    // Get selected horizon
    const horizonSelect = document.getElementById('global-horizon');
    const selectedHorizon = horizonSelect ? horizonSelect.value : '12M';
    
    try {
        const res = await fetch(`${API_BASE}/analyze/${ticker}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ horizon: selectedHorizon, persist: true })
        });
        
        if (!res.ok) throw new Error("Analysis failed");
        
        const data = await res.json();
        renderReport(ticker, data);
        loadDashboard(); // update recent predictions
    } catch (e) {
        modal.classList.add('hidden');
        showToast(`Analysis for ${ticker} failed: ${e.message}`, 'error');
    }
};

window.openJournalEntry = async (id) => {
    modal.classList.remove('hidden');
    document.getElementById('modal-title').textContent = `Loading entry...`;
    document.getElementById('analysis-loading').classList.remove('hidden');
    document.getElementById('analysis-report').classList.add('hidden');
    
    try {
        const res = await fetch(`${API_BASE}/journal/predictions/${id}`);
        if (!res.ok) throw new Error("Could not load prediction");
        const data = await res.json();
        renderReport(data.ticker, data);
    } catch(e) {
        modal.classList.add('hidden');
        showToast(`Failed to load: ${e.message}`, 'error');
    }
};

function renderReport(ticker, data) {
    document.getElementById('modal-title').textContent = `Analysis Complete: ${ticker}`;
    document.getElementById('analysis-loading').classList.add('hidden');
    document.getElementById('analysis-report').classList.remove('hidden');
    
    document.getElementById('rep-ticker').textContent = ticker;
    document.getElementById('rep-score').textContent = `${data.weighted_score.toFixed(2)}/10`;
    
    const ratingEl = document.getElementById('rep-rating');
    ratingEl.textContent = data.rating;
    ratingEl.className = `badge ${data.rating.toLowerCase()}`;
    
    // Update horizon
    const horizonEl = document.getElementById('rep-horizon');
    if (horizonEl) {
        horizonEl.textContent = data.horizon || '12M';
    }
    
    document.getElementById('rep-thesis').innerHTML = data.thesis.replace(/\n\n/g, '<br><br>');
    
    // Scenarios
    const sm = data.scenarios;
    document.getElementById('rep-bull-prob').textContent = formatPercent(sm.bull.probability);
    document.getElementById('rep-bull-ret').textContent = formatPercent(sm.bull.return);
    document.getElementById('rep-base-prob').textContent = formatPercent(sm.base.probability);
    document.getElementById('rep-base-ret').textContent = formatPercent(sm.base.return);
    document.getElementById('rep-bear-prob').textContent = formatPercent(sm.bear.probability);
    document.getElementById('rep-bear-ret').textContent = formatPercent(sm.bear.return);
    document.getElementById('rep-ev').textContent = formatPercent(sm.expected_value);
    
    // Risks
    const risksUl = document.getElementById('rep-risks');
    risksUl.innerHTML = data.invalidation_criteria.map(r => `<li>${r}</li>`).join('');
    
    // Disagreement
    const diagWarn = document.getElementById('rep-disagreement');
    if (data.disagreement && data.disagreement.score > 3.0) {
        diagWarn.classList.remove('hidden');
    } else {
        diagWarn.classList.add('hidden');
    }
    
    // Agents
    const agentsDiv = document.getElementById('rep-agents');
    agentsDiv.innerHTML = Object.entries(data.agent_outputs).map(([name, r]) => `
        <div class="agent-row" style="flex-direction: column; gap: 12px;">
            <div style="display: flex; justify-content: space-between;">
                <div class="agent-row-info">
                    <h4>${name}</h4>
                    <p>Confidence: ${formatPercent(r.confidence)}</p>
                </div>
                <div class="agent-score">${r.score.toFixed(1)}</div>
            </div>
            <div class="agent-details" style="font-size: 14px; color: #cbd5e1; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px;">
                <p><strong>Summary:</strong> ${r.summary}</p>
                ${r.key_findings.length > 0 ? `<ul style="margin-top:8px; padding-left:16px;">${r.key_findings.map(k => `<li>${k}</li>`).join('')}</ul>` : ''}
            </div>
        </div>
    `).join('');
}

// Initial load
loadDashboard();
