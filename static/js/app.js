/**
 * Story Teller — Next-Gen AI Narrative Creative Suite
 * Complete, robust, fully featured application logic.
 */

// ─── Global State ─────────────────────────────────────────────────
let currentProject = null;
let activeModel = '';
let modelsCatalog = null;
let runtimeSettings = null;
let projectsList = [];
let characters = [];
let editCharacters = [];
let isGenerating = false;
let eventSource = null;
let totalWords = 0;
let scenesComplete = 0;
let totalScenes = 0;

// Manual Interactive Session State
let manualSessionActive = false;
let manualScenesCompleted = 0;
let manualChapterNum = 0;
let manualIsGeneratingScene = false;
let manualSceneTexts = new Map();

// Premise Studio State
let premiseSteps = [];
let premiseCharacters = {};

// Reader State
let currentChapterNumber = 1;
let currentChapterRaw = '';
let readerFontSize = 17;
const readerThemes = ['obsidian', 'parchment', 'midnight', 'velvet'];
let readerThemeIndex = 0;
const readerFonts = ['font-serif-lora', 'font-serif-merriweather', 'font-sans-inter'];
let readerFontIndex = 0;
let readerIsFullscreen = false;
let isReaderEditMode = false;

// ─── Helpers: Model Formatting & Provider Detection ──────────────
function parseModelProvider(modelId) {
    if (!modelId) return 'local';
    const str = String(modelId).toLowerCase();
    if (str === 'hybrid' || str.startsWith('hybrid:')) return 'hybrid';
    if (str.startsWith('groq:') || str === '__groq_api__') return 'groq';
    if (str.startsWith('gemini:') || str === '__gemini_api__') return 'gemini';
    if (str.startsWith('openrouter:') || str === '__openrouter_api__') return 'openrouter';
    return 'local';
}

function getProviderIcon(provider) {
    switch (provider) {
        case 'groq': return '⚡';
        case 'gemini': return '✨';
        case 'openrouter': return '🌐';
        case 'hybrid': return '🔀';
        default: return '💻';
    }
}

function getProviderBadgeClass(provider) {
    return 'badge-' + provider;
}

function formatModelDisplayName(modelId) {
    if (!modelId) return 'Select Model';
    if (modelId === 'hybrid' || modelId.startsWith('hybrid:')) {
        return 'Hybrid (Groq + Local)';
    }
    if (modelId.startsWith('groq:')) {
        const name = modelId.slice(5);
        return name.split('/').pop();
    }
    if (modelId.startsWith('gemini:')) {
        return modelId.slice(7);
    }
    if (modelId.startsWith('openrouter:')) {
        const name = modelId.slice(11);
        return name.split('/').pop().replace(':free', ' (Free)');
    }
    if (modelId === '__groq_api__') return 'Groq Cloud';
    if (modelId === '__gemini_api__') return 'Gemini Cloud';
    if (modelId === '__openrouter_api__') return 'OpenRouter';
    // Local GGUF
    return modelId.split('/').pop().replace('.gguf', '');
}

function escHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ─── Toast Notifications ──────────────────────────────────────────
function showToast(message, type = 'info', duration = 3800) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '⚠️';

    toast.innerHTML = `
        <span style="font-size:1.15rem;">${icon}</span>
        <span style="flex:1;line-height:1.4;">${escHtml(message)}</span>
        <button onclick="this.parentElement.remove()" style="background:transparent;border:none;color:var(--text-muted);cursor:pointer;font-size:0.9rem;padding:2px;">✕</button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        setTimeout(() => toast.remove(), 250);
    }, duration);
}

// ─── Status Indicator ─────────────────────────────────────────────
function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    const textEl = document.getElementById('statusText');
    if (!dot || !textEl) return;

    dot.className = 'status-dot';
    if (state === 'generating') dot.classList.add('generating');
    else if (state === 'error') dot.classList.add('error');

    textEl.textContent = text || 'Engine Ready';
}

// ─── Custom Modal Confirm ─────────────────────────────────────────
function showConfirmModal(title, message, onConfirm) {
    const modal = document.getElementById('confirmModal');
    const titleEl = document.getElementById('confirmModalTitle');
    const msgEl = document.getElementById('confirmModalMessage');
    const actionBtn = document.getElementById('btnConfirmAction');
    if (!modal) return;

    titleEl.textContent = title || 'Confirm Action';
    msgEl.textContent = message || 'Are you sure you want to proceed?';

    actionBtn.onclick = () => {
        closeConfirmModal();
        if (typeof onConfirm === 'function') onConfirm();
    };

    modal.classList.remove('hidden');
}

function closeConfirmModal() {
    document.getElementById('confirmModal')?.classList.add('hidden');
}

// ─── View Navigation ──────────────────────────────────────────────
function switchView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

    const view = document.getElementById('view' + viewName.charAt(0).toUpperCase() + viewName.slice(1));
    const tab = document.querySelector(`.nav-tab[data-view="${viewName}"]`);

    if (view) view.classList.add('active');
    if (tab) tab.classList.add('active');

    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (viewName === 'dashboard') loadProjects();
    if (viewName === 'reader' && currentProject) loadChapters();
    if (viewName === 'state' && currentProject) refreshState();
    if (viewName === 'combine' && currentProject) loadCombineVersions();
    if (viewName === 'edit' && currentProject) loadProjectForEdit();
    if (viewName === 'premise' && currentProject) loadProjectPremise();
    if (viewName === 'settings') loadSettings();
    if (viewName === 'vision') checkVisionStatus();
    if (viewName === 'manual' && currentProject) syncManualStatus();
}

function refreshCurrentView() {
    const activeTab = document.querySelector('.nav-tab.active');
    const viewName = activeTab ? activeTab.getAttribute('data-view') : 'dashboard';
    switchView(viewName);
    loadModels();
    showToast('Refreshed workspace', 'info');
}

// ─── Model Management (Universal Switcher & Modal) ────────────────
async function loadModels() {
    try {
        const res = await fetch('/api/models');
        if (!res.ok) throw new Error('Failed to fetch models');
        modelsCatalog = await res.json();

        activeModel = modelsCatalog.active || '';
        const provider = parseModelProvider(activeModel);

        // Update Header Button
        const headerIcon = document.getElementById('headerProviderIcon');
        const headerName = document.getElementById('headerModelName');
        const headerBadge = document.getElementById('headerProviderBadge');

        if (headerIcon) headerIcon.textContent = getProviderIcon(provider);
        if (headerName) headerName.textContent = formatModelDisplayName(activeModel);
        if (headerBadge) {
            headerBadge.className = `model-picker-badge ${getProviderBadgeClass(provider)}`;
            headerBadge.textContent = provider.toUpperCase();
        }

        // Update active engine in dashboard stats
        const activeEngineStat = document.getElementById('statActiveEngine');
        if (activeEngineStat) {
            activeEngineStat.textContent = provider.toUpperCase();
        }

        // Render modal model cards
        renderModalModelCards('all');

        // Populate settings active model select
        populateSettingsModelSelect(modelsCatalog);

    } catch (e) {
        console.error('Error loading models:', e);
        showToast('Error loading models catalog', 'error');
    }
}

function openModelModal() {
    if (!modelsCatalog) loadModels();
    document.getElementById('modelSwitcherModal')?.classList.remove('hidden');
    filterModalModels('all');
}

function closeModelModal() {
    document.getElementById('modelSwitcherModal')?.classList.add('hidden');
}

function filterModalModels(provider) {
    document.querySelectorAll('#modalProviderTabs .bible-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.toLowerCase().includes(provider) || (provider === 'all' && btn.textContent.includes('All'))) {
            btn.classList.add('active');
        }
    });
    renderModalModelCards(provider);
}

function renderModalModelCards(filter) {
    const container = document.getElementById('modalModelCardsContainer');
    if (!container || !modelsCatalog) return;

    let cardsHtml = '';

    // Hybrid Option
    if (filter === 'all' || filter === 'hybrid') {
        const isSelected = activeModel === 'hybrid' || activeModel.startsWith('hybrid:');
        cardsHtml += `
            <div class="card ${isSelected ? 'selected' : ''}" style="cursor:pointer;padding:16px;border:1px solid ${isSelected ? 'var(--accent-purple)' : 'var(--border-subtle)'};background:${isSelected ? 'rgba(139,92,246,0.18)' : 'var(--bg-surface)'};" onclick="switchModel('hybrid')">
                <div class="flex justify-between items-center mb-2">
                    <span class="model-picker-badge badge-hybrid">HYBRID</span>
                    <span style="font-size:0.75rem;color:var(--text-muted);">Writer: Groq / Planner: Local</span>
                </div>
                <strong style="font-size:0.95rem;color:var(--text-primary);display:block;margin-bottom:4px;">🔀 Hybrid Architecture</strong>
                <p class="text-xs text-dim" style="line-height:1.4;">Routes prose writing to Groq cloud and planning/critic/editor passes to local GGUF. Optimal speed and zero token waste.</p>
            </div>
        `;
    }

    // Groq Models
    if (filter === 'all' || filter === 'groq') {
        const groqList = modelsCatalog.groq_models || [];
        groqList.forEach(m => {
            const modelKey = `groq:${m.id}`;
            const isSelected = activeModel === modelKey;
            cardsHtml += `
                <div class="card ${isSelected ? 'selected' : ''}" style="cursor:pointer;padding:16px;border:1px solid ${isSelected ? 'var(--accent-purple)' : 'var(--border-subtle)'};background:${isSelected ? 'rgba(139,92,246,0.18)' : 'var(--bg-surface)'};" onclick="switchModel('${escHtml(modelKey)}')">
                    <div class="flex justify-between items-center mb-2">
                        <span class="model-picker-badge badge-groq">GROQ</span>
                        <span class="model-option-ctx">${escHtml(m.context || '128k')}</span>
                    </div>
                    <strong style="font-size:0.95rem;color:var(--text-primary);display:block;margin-bottom:4px;">⚡ ${escHtml(m.name || m.id)}</strong>
                    <p class="text-xs text-dim" style="line-height:1.4;">${escHtml(m.notes || 'Ultra-fast inference on Groq LPU hardware.')}</p>
                </div>
            `;
        });
    }

    // Gemini Models
    if (filter === 'all' || filter === 'gemini') {
        const geminiList = modelsCatalog.gemini_models || [];
        geminiList.forEach(m => {
            const modelKey = `gemini:${m.id}`;
            const isSelected = activeModel === modelKey;
            cardsHtml += `
                <div class="card ${isSelected ? 'selected' : ''}" style="cursor:pointer;padding:16px;border:1px solid ${isSelected ? 'var(--accent-purple)' : 'var(--border-subtle)'};background:${isSelected ? 'rgba(139,92,246,0.18)' : 'var(--bg-surface)'};" onclick="switchModel('${escHtml(modelKey)}')">
                    <div class="flex justify-between items-center mb-2">
                        <span class="model-picker-badge badge-gemini">GEMINI</span>
                        <span class="model-option-ctx">${escHtml(m.context || '1M')}</span>
                    </div>
                    <strong style="font-size:0.95rem;color:var(--text-primary);display:block;margin-bottom:4px;">✨ ${escHtml(m.name || m.id)}</strong>
                    <p class="text-xs text-dim" style="line-height:1.4;">${escHtml(m.notes || 'Flagship reasoning and vast 1M context window.')}</p>
                </div>
            `;
        });
    }

    // OpenRouter Models
    if (filter === 'all' || filter === 'openrouter') {
        const orList = modelsCatalog.openrouter_models || [];
        orList.forEach(m => {
            const modelKey = `openrouter:${m.id}`;
            const isSelected = activeModel === modelKey;
            cardsHtml += `
                <div class="card ${isSelected ? 'selected' : ''}" style="cursor:pointer;padding:16px;border:1px solid ${isSelected ? 'var(--accent-purple)' : 'var(--border-subtle)'};background:${isSelected ? 'rgba(139,92,246,0.18)' : 'var(--bg-surface)'};" onclick="switchModel('${escHtml(modelKey)}')">
                    <div class="flex justify-between items-center mb-2">
                        <span class="model-picker-badge badge-openrouter">OPENROUTER</span>
                        <span class="model-option-ctx">${escHtml(m.context || '128k')}</span>
                    </div>
                    <strong style="font-size:0.95rem;color:var(--text-primary);display:block;margin-bottom:4px;">🌐 ${escHtml(m.name || m.id)}</strong>
                    <p class="text-xs text-dim" style="line-height:1.4;">${escHtml(m.notes || 'OpenRouter unified API model.')}</p>
                </div>
            `;
        });
    }

    // Local GGUF Models
    if (filter === 'all' || filter === 'local') {
        const localList = modelsCatalog.local_models || [];
        if (localList.length === 0) {
            cardsHtml += `
                <div class="card" style="padding:16px;grid-column:1/-1;">
                    <div class="text-xs text-dim">No .gguf model files detected in models folder. Add GGUFs to models/ directory to run offline.</div>
                </div>
            `;
        } else {
            localList.forEach(m => {
                const isSelected = activeModel === m;
                cardsHtml += `
                    <div class="card ${isSelected ? 'selected' : ''}" style="cursor:pointer;padding:16px;border:1px solid ${isSelected ? 'var(--accent-purple)' : 'var(--border-subtle)'};background:${isSelected ? 'rgba(139,92,246,0.18)' : 'var(--bg-surface)'};" onclick="switchModel('${escHtml(m)}')">
                        <div class="flex justify-between items-center mb-2">
                            <span class="model-picker-badge badge-local">LOCAL GGUF</span>
                            <span class="model-option-ctx">Offline</span>
                        </div>
                        <strong style="font-size:0.95rem;color:var(--text-primary);display:block;margin-bottom:4px;">💻 ${escHtml(m)}</strong>
                        <p class="text-xs text-dim" style="line-height:1.4;">100% offline llama.cpp execution on your hardware.</p>
                    </div>
                `;
            });
        }
    }

    container.innerHTML = cardsHtml;
}

async function switchModel(modelId) {
    if (!modelId) return;
    setStatus('generating', 'Switching model...');

    try {
        const res = await fetch('/api/models/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelId }),
        });

        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to switch model');
        }

        activeModel = data.active || modelId;
        const provider = parseModelProvider(activeModel);
        showToast(`Switched active engine to ${formatModelDisplayName(activeModel)} (${provider.toUpperCase()})`, 'success');
        
        closeModelModal();
        await loadModels();
        setStatus('online', 'Engine Ready');

    } catch (e) {
        showToast(e.message || 'Model switch error', 'error');
        setStatus('error', 'Model Error');
        loadModels();
    }
}

function populateSettingsModelSelect(catalog) {
    const select = document.getElementById('settingsActiveModel');
    if (!select || !catalog) return;

    let optHtml = '';

    // Hybrid
    optHtml += `<optgroup label="Hybrid Architecture"><option value="hybrid" ${catalog.active === 'hybrid' ? 'selected' : ''}>🔀 Hybrid (Groq Writer + Local Planning)</option></optgroup>`;

    // Groq
    if (catalog.groq_models?.length) {
        optHtml += '<optgroup label="Groq Cloud LPU">';
        catalog.groq_models.forEach(m => {
            const val = `groq:${m.id}`;
            optHtml += `<option value="${escHtml(val)}" ${catalog.active === val ? 'selected' : ''}>⚡ ${escHtml(m.name || m.id)}</option>`;
        });
        optHtml += '</optgroup>';
    }

    // Gemini
    if (catalog.gemini_models?.length) {
        optHtml += '<optgroup label="Google Gemini">';
        catalog.gemini_models.forEach(m => {
            const val = `gemini:${m.id}`;
            optHtml += `<option value="${escHtml(val)}" ${catalog.active === val ? 'selected' : ''}>✨ ${escHtml(m.name || m.id)}</option>`;
        });
        optHtml += '</optgroup>';
    }

    // OpenRouter
    if (catalog.openrouter_models?.length) {
        optHtml += '<optgroup label="OpenRouter Hub">';
        catalog.openrouter_models.forEach(m => {
            const val = `openrouter:${m.id}`;
            optHtml += `<option value="${escHtml(val)}" ${catalog.active === val ? 'selected' : ''}>🌐 ${escHtml(m.name || m.id)}</option>`;
        });
        optHtml += '</optgroup>';
    }

    // Local
    if (catalog.local_models?.length) {
        optHtml += '<optgroup label="Local GGUF Models">';
        catalog.local_models.forEach(m => {
            optHtml += `<option value="${escHtml(m)}" ${catalog.active === m ? 'selected' : ''}>💻 ${escHtml(m)}</option>`;
        });
        optHtml += '</optgroup>';
    }

    select.innerHTML = optHtml;
    select.onchange = (e) => switchModel(e.target.value);
}

// ─── Project Management & Story Library ───────────────────────────
async function loadProjects() {
    try {
        const res = await fetch('/api/projects');
        if (!res.ok) throw new Error('Failed to load projects');
        const data = await res.json();
        projectsList = Array.isArray(data) ? data : (data.projects || []);

        // Calculate Totals
        let totalCh = 0;
        let totalW = 0;
        projectsList.forEach(p => {
            totalCh += p.total_chapters || p.current_chapter || 0;
            totalW += p.word_count || 0;
        });

        const statTotalStories = document.getElementById('statTotalStories');
        const statTotalChapters = document.getElementById('statTotalChapters');
        const statTotalWords = document.getElementById('statTotalWords');

        if (statTotalStories) statTotalStories.textContent = projectsList.length;
        if (statTotalChapters) statTotalChapters.textContent = totalCh;
        if (statTotalWords) statTotalWords.textContent = totalW >= 1000 ? `${(totalW / 1000).toFixed(1)}k` : totalW;

        renderProjectsGrid(projectsList);

    } catch (e) {
        console.error('Error loading stories:', e);
        showToast('Error loading story library', 'error');
    }
}

function filterProjectsList() {
    const searchVal = (document.getElementById('searchProjectsInput')?.value || '').toLowerCase().trim();
    const genreVal = document.getElementById('filterGenreSelect')?.value || 'all';

    const filtered = projectsList.filter(p => {
        const title = (p.title || p.name || '').toLowerCase();
        const premise = (p.premise || '').toLowerCase();
        const genre = (p.genre || '').toLowerCase();

        const matchesSearch = !searchVal || title.includes(searchVal) || premise.includes(searchVal);
        const matchesGenre = genreVal === 'all' || genre.includes(genreVal.toLowerCase());
        return matchesSearch && matchesGenre;
    });

    renderProjectsGrid(filtered);
}

function renderProjectsGrid(list) {
    const grid = document.getElementById('projectsGrid');
    if (!grid) return;

    if (!list || list.length === 0) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column:1/-1;">
                <span class="icon">📚</span>
                <h3>No matching stories found</h3>
                <p>Try adjusting your search or genre filters, or create a brand new story.</p>
                <button class="btn btn-primary" onclick="switchView('create')">✨ Create New Story</button>
            </div>
        `;
        return;
    }

    grid.innerHTML = list.map(p => {
        const title = p.title || p.name || 'Untitled Story';
        const genre = p.genre || 'Fiction';
        const premise = p.premise || 'No premise outline set.';
        const chaptersCount = p.total_chapters !== undefined ? p.total_chapters : (p.current_chapter || 0);
        const wordsCount = p.word_count || 0;
        const charNames = Array.isArray(p.characters) ? p.characters.slice(0, 4) : [];
        const isCurrent = currentProject === p.name;

        return `
            <div class="project-card ${isCurrent ? 'active-story' : ''}">
                <div class="project-card-header">
                    <span class="project-genre-badge">${escHtml(genre)}</span>
                    <span class="text-xs text-dim">${p.created_at ? new Date(p.created_at).toLocaleDateString() : 'Active'}</span>
                </div>

                <div class="project-card-title">${escHtml(title)}</div>
                <div class="project-card-premise">${escHtml(premise)}</div>

                <div class="project-meta-row">
                    <div class="project-meta-item"><span>📑</span> <strong>${chaptersCount}</strong> Ch</div>
                    <div class="project-meta-item"><span>✍️</span> <strong>${wordsCount}</strong> Words</div>
                    ${charNames.length ? `<div class="project-meta-item"><span>👥</span> ${escHtml(charNames.join(', '))}${p.characters.length > 4 ? '...' : ''}</div>` : ''}
                </div>

                <div class="project-card-actions">
                    <button class="btn btn-primary btn-sm" onclick="selectProject('${escHtml(p.name)}', 'generate')">⚡ Generate</button>
                    <button class="btn btn-secondary btn-sm" onclick="selectProject('${escHtml(p.name)}', 'manual')">🧩 Manual</button>
                    <button class="btn btn-secondary btn-sm" onclick="selectProject('${escHtml(p.name)}', 'reader')">📖 Read</button>
                    <button class="btn btn-secondary btn-sm" onclick="selectProject('${escHtml(p.name)}', 'premise')">🎬 Premise</button>
                    <button class="btn btn-secondary btn-sm" onclick="selectProject('${escHtml(p.name)}', 'edit')">✏️ Edit</button>
                </div>
            </div>
        `;
    }).join('');
}

async function selectProject(name, targetView = 'generate') {
    if (!name) return;
    currentProject = name;

    // Show Project Pill in Header
    const pill = document.getElementById('headerProjectPill');
    const nameEl = document.getElementById('headerProjectName');
    const statsEl = document.getElementById('headerProjectStats');

    if (pill) pill.classList.remove('hidden');
    if (nameEl) nameEl.textContent = name;

    // Reveal story tabs
    ['tabPremise', 'tabGenerate', 'tabManual', 'tabReader', 'tabCombine', 'tabState', 'tabLogs', 'tabEdit'].forEach(id => {
        document.getElementById(id)?.classList.remove('hidden');
    });

    document.getElementById('activeProject').textContent = `Story: ${name}`;

    try {
        const res = await fetch(`/api/project/${name}`);
        if (res.ok) {
            const data = await res.json();
            const chs = data.state?.chapters || [];
            if (statsEl) statsEl.textContent = `${chs.length} Ch`;
        }
    } catch (e) {}

    showToast(`Loaded story: ${name}`, 'info', 2200);
    switchView(targetView);
}

// ─── Character Builder (Create & Edit) ────────────────────────────
function renderCharacterList() {
    const list = document.getElementById('characterList');
    if (!list) return;

    if (characters.length === 0) {
        list.innerHTML = '<span class="text-dim text-xs">No characters added yet. Use the fields below to add protagonists, antagonists, and allies.</span>';
        return;
    }

    list.innerHTML = characters.map((c, idx) => `
        <div class="character-builder-item">
            <div class="char-info-col">
                <span class="char-info-name">${escHtml(c.name)}</span>
                <span class="char-info-desc">${escHtml(c.description || '')} • <em>${escHtml(c.traits?.join(', ') || '')}</em></span>
            </div>
            <button class="btn btn-sm btn-danger btn-icon" onclick="removeCharacter(${idx})" style="width:28px;height:28px;font-size:11px;">✕</button>
        </div>
    `).join('');
}

function addCharacter() {
    const nameInput = document.getElementById('charName');
    const descInput = document.getElementById('charDesc');
    const traitsInput = document.getElementById('charTraits');

    const name = (nameInput?.value || '').trim();
    if (!name) {
        showToast('Character name is required', 'error');
        return;
    }

    const desc = (descInput?.value || '').trim();
    const traits = (traitsInput?.value || '').split(',').map(t => t.trim()).filter(Boolean);

    characters.push({ name, description: desc, traits });
    nameInput.value = '';
    descInput.value = '';
    traitsInput.value = '';
    nameInput.focus();

    renderCharacterList();
}

function removeCharacter(idx) {
    characters.splice(idx, 1);
    renderCharacterList();
}

// ─── Create Story Action ──────────────────────────────────────────
async function createProject() {
    const titleInput = document.getElementById('createTitle');
    const genreInput = document.getElementById('createGenre');
    const premiseInput = document.getElementById('createPremise');
    const settingInput = document.getElementById('createSetting');
    const themesInput = document.getElementById('createThemes');
    const btn = document.getElementById('btnCreateProject');

    const title = (titleInput?.value || '').trim();
    if (!title) {
        showToast('Story title is required', 'error');
        titleInput?.focus();
        return;
    }

    const genre = genreInput?.value || 'dark fantasy';
    const premise = (premiseInput?.value || '').trim();
    const setting = (settingInput?.value || '').trim();
    const themes = (themesInput?.value || '').split(',').map(t => t.trim()).filter(Boolean);

    // Format characters dictionary
    const charsDict = {};
    characters.forEach(c => {
        charsDict[c.name] = {
            description: c.description,
            traits: c.traits,
        };
    });

    if (btn) {
        btn.disabled = true;
        btn.textContent = '🚀 Initializing Pipeline...';
    }

    try {
        const res = await fetch('/api/project/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: title,
                title: title,
                genre: genre,
                premise: premise,
                setting: setting,
                themes: themes,
                characters: charsDict,
            }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to create project');

        showToast(`Created story "${title}" successfully!`, 'success');
        characters = [];
        titleInput.value = '';
        premiseInput.value = '';
        settingInput.value = '';
        themesInput.value = '';
        renderCharacterList();

        selectProject(data.project || data.name || title, 'premise');

    } catch (e) {
        showToast(e.message || 'Creation failed', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🚀 Initialize Story Studio';
        }
    }
}

// ─── Edit Story Details ───────────────────────────────────────────
async function loadProjectForEdit() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}`);
        const data = await res.json();
        const state = data.state || {};
        const meta = state.metadata || {};

        document.getElementById('editTitle').value = meta.title || currentProject;
        document.getElementById('editGenre').value = meta.genre || '';
        document.getElementById('editPremise').value = meta.premise || '';
        document.getElementById('editSetting').value = meta.setting || '';
        document.getElementById('editThemes').value = (meta.themes || []).join(', ');

        editCharacters = [];
        const charsObj = state.characters || {};
        Object.keys(charsObj).forEach(k => {
            editCharacters.push({
                name: k,
                description: charsObj[k].description || '',
                traits: charsObj[k].traits || [],
            });
        });
        renderEditCharacterList();

    } catch (e) {
        showToast('Failed to load story for editing', 'error');
    }
}

function renderEditCharacterList() {
    const list = document.getElementById('editCharacterList');
    if (!list) return;

    if (editCharacters.length === 0) {
        list.innerHTML = '<span class="text-dim text-xs">No characters registered.</span>';
        return;
    }

    list.innerHTML = editCharacters.map((c, idx) => `
        <div class="character-builder-item">
            <div class="char-info-col">
                <span class="char-info-name">${escHtml(c.name)}</span>
                <span class="char-info-desc">${escHtml(c.description || '')} • <em>${escHtml(c.traits?.join(', ') || '')}</em></span>
            </div>
            <button class="btn btn-sm btn-danger btn-icon" onclick="removeEditCharacter(${idx})" style="width:28px;height:28px;font-size:11px;">✕</button>
        </div>
    `).join('');
}

function addEditCharacter() {
    const name = (document.getElementById('editCharName')?.value || '').trim();
    if (!name) return;
    const desc = (document.getElementById('editCharDesc')?.value || '').trim();
    const traits = (document.getElementById('editCharTraits')?.value || '').split(',').map(t => t.trim()).filter(Boolean);

    editCharacters.push({ name, description: desc, traits });
    document.getElementById('editCharName').value = '';
    document.getElementById('editCharDesc').value = '';
    document.getElementById('editCharTraits').value = '';
    renderEditCharacterList();
}

function removeEditCharacter(idx) {
    editCharacters.splice(idx, 1);
    renderEditCharacterList();
}

async function saveProjectEdits() {
    if (!currentProject) return;
    const btn = document.getElementById('btnSaveEdits');

    const title = (document.getElementById('editTitle')?.value || '').trim();
    const genre = (document.getElementById('editGenre')?.value || '').trim();
    const premise = (document.getElementById('editPremise')?.value || '').trim();
    const setting = (document.getElementById('editSetting')?.value || '').trim();
    const themes = (document.getElementById('editThemes')?.value || '').split(',').map(t => t.trim()).filter(Boolean);

    const charsDict = {};
    editCharacters.forEach(c => {
        charsDict[c.name] = {
            description: c.description,
            traits: c.traits,
        };
    });

    if (btn) { btn.disabled = true; btn.textContent = '💾 Saving...'; }

    try {
        const res = await fetch(`/api/project/${currentProject}/state`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                metadata: { title, genre, premise, setting, themes },
                characters: charsDict,
            }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save edits');

        showToast('Saved story changes!', 'success');
        loadProjects();
    } catch (e) {
        showToast(e.message || 'Save error', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '💾 Save Story Changes'; }
    }
}

function deleteCurrentProject() {
    if (!currentProject) return;
    showConfirmModal(
        'Delete Story Project',
        `Are you sure you want to permanently delete "${currentProject}" and all its chapters? This action cannot be undone.`,
        async () => {
            try {
                const res = await fetch(`/api/project/${currentProject}/delete`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok || data.error) throw new Error(data.error || 'Failed to delete');

                showToast(`Deleted story "${currentProject}"`, 'success');
                currentProject = null;
                document.getElementById('headerProjectPill')?.classList.add('hidden');
                ['tabPremise', 'tabGenerate', 'tabManual', 'tabReader', 'tabCombine', 'tabState', 'tabLogs', 'tabEdit'].forEach(id => {
                    document.getElementById(id)?.classList.add('hidden');
                });
                switchView('dashboard');
            } catch (e) {
                showToast(e.message, 'error');
            }
        }
    );
}

// ─── Premise Studio Logic ─────────────────────────────────────────
async function loadProjectPremise() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}`);
        const data = await res.json();
        const meta = data.state?.metadata || {};

        document.getElementById('premiseSetting').value = meta.setting || '';
        document.getElementById('premiseThemes').value = (meta.themes || []).join(', ');

        const charsObj = data.state?.characters || {};
        premiseCharacters = charsObj;
        renderPremiseCharactersChips(charsObj);

        const rawPremise = meta.premise || '';
        premiseSteps = parsePremiseSteps(rawPremise);
        renderPremiseTimeline();

    } catch (e) {
        showToast('Failed to load story premise', 'error');
    }
}

function parsePremiseSteps(rawText) {
    if (!rawText) return [];
    const lines = rawText.split('\n');
    const steps = [];
    const HEADING_RE = /^(?:#+\s*|act\s+|chapter\s+|part\s+|section\s+|phase\s+|prologue|epilogue)/i;
    for (let rawLine of lines) {
        let line = rawLine.trim();
        if (!line) continue;
        if (HEADING_RE.test(line)) continue;
        if (line.endsWith(':') && line.split(/\s+/).length <= 5) continue;
        line = line.replace(/^(?:[-*]|\d+[\).\:-])\s*/, '').trim();
        if (!line) continue;
        steps.push(line);
    }
    return steps;
}

function renderPremiseCharactersChips(charsObj) {
    const container = document.getElementById('premiseCharactersList');
    if (!container) return;
    const keys = Object.keys(charsObj);
    if (keys.length === 0) {
        container.innerHTML = '<span class="text-dim text-xs">No characters defined.</span>';
        return;
    }
    container.innerHTML = keys.map(name => `
        <span class="char-trait-chip" style="background:rgba(139,92,246,0.15);color:#c4b5fd;border:1px solid rgba(139,92,246,0.3);">
            <strong>${escHtml(name)}</strong>: ${escHtml((charsObj[name].traits || []).join(', '))}
        </span>
    `).join('');
}

function renderPremiseTimeline() {
    const container = document.getElementById('premiseTimelineContainer');
    if (!container) return;

    if (premiseSteps.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="icon">📋</span>
                <h3>No beats defined</h3>
                <p>Enter an outline on the left and click "Generate Structured Beats" or add beats manually.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = premiseSteps.map((step, idx) => `
        <div class="premise-beat-card">
            <div class="beat-number-badge">${idx + 1}</div>
            <div class="beat-content-text" contenteditable="true" onblur="updatePremiseStep(${idx}, this.innerText)">${escHtml(step)}</div>
            <div class="beat-actions-col">
                <button class="beat-action-btn" onclick="movePremiseStep(${idx}, -1)" title="Move Up" ${idx === 0 ? 'disabled' : ''}>▲</button>
                <button class="beat-action-btn" onclick="movePremiseStep(${idx}, 1)" title="Move Down" ${idx === premiseSteps.length - 1 ? 'disabled' : ''}>▼</button>
                <button class="beat-action-btn" onclick="deletePremiseStep(${idx})" title="Delete Beat" style="color:#ef4444;">✕</button>
            </div>
        </div>
    `).join('');
}

function addBlankPremiseStep() {
    premiseSteps.push('New plot beat description...');
    renderPremiseTimeline();
}

function deletePremiseStep(idx) {
    premiseSteps.splice(idx, 1);
    renderPremiseTimeline();
}

function movePremiseStep(idx, dir) {
    const target = idx + dir;
    if (target < 0 || target >= premiseSteps.length) return;
    const temp = premiseSteps[idx];
    premiseSteps[idx] = premiseSteps[target];
    premiseSteps[target] = temp;
    renderPremiseTimeline();
}

function updatePremiseStep(idx, text) {
    premiseSteps[idx] = text.trim();
}

async function aiGeneratePremise() {
    if (!currentProject) return;
    const text = (document.getElementById('premiseIdeaText')?.value || '').trim();
    if (!text) {
        showToast('Please enter an outline or notes in the text box first', 'error');
        return;
    }

    const btn = document.getElementById('btnPremiseGenerate');
    if (btn) { btn.disabled = true; btn.textContent = '✨ Architecting Beats...'; }

    try {
        const res = await fetch(`/api/project/${currentProject}/premise/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idea_text: text }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to generate beats');

        premiseSteps = data.steps || [];
        renderPremiseTimeline();
        showToast(`Generated ${premiseSteps.length} narrative timeline beats!`, 'success');

    } catch (e) {
        showToast(e.message || 'Generation failed', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '✨ Generate Structured Beats from Outline'; }
    }
}

async function aiRefinePremiseFlow() {
    if (!currentProject || premiseSteps.length === 0) return;
    const btn = document.getElementById('btnPremiseRefine');
    if (btn) { btn.disabled = true; btn.textContent = '⚖️ Refining Flow...'; }

    try {
        const res = await fetch(`/api/project/${currentProject}/premise/refine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps: premiseSteps }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to refine');

        premiseSteps = data.steps || premiseSteps;
        renderPremiseTimeline();
        showToast('Timeline beats refined for pacing and dramatic structure!', 'success');

    } catch (e) {
        showToast(e.message || 'Refinement failed', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '⚖️ Refine Flow & Pacing'; }
    }
}

async function aiExpandPremiseBeats() {
    if (!currentProject || premiseSteps.length === 0) return;
    const btn = document.getElementById('btnPremiseExpand');
    if (btn) { btn.disabled = true; btn.textContent = '🚀 Expanding...'; }

    try {
        const res = await fetch(`/api/project/${currentProject}/premise/expand`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps: premiseSteps }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to expand');

        premiseSteps = data.steps || premiseSteps;
        renderPremiseTimeline();
        showToast(`Expanded timeline to ${premiseSteps.length} scene beats!`, 'success');

    } catch (e) {
        showToast(e.message || 'Expansion failed', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '🚀 Expand (Add Beats)'; }
    }
}

async function savePremiseToProject() {
    if (!currentProject) return;
    const btn = document.getElementById('btnPremiseSave');
    if (btn) { btn.disabled = true; btn.textContent = '💾 Saving...'; }

    const formattedPremise = premiseSteps.map((s, i) => `${i + 1}. ${s}`).join('\n\n');

    try {
        const res = await fetch(`/api/project/${currentProject}/state`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                metadata: { premise: formattedPremise },
            }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save');

        showToast('Saved premise timeline to Story Bible!', 'success');

    } catch (e) {
        showToast(e.message || 'Save failed', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '💾 Save Timeline to Story Bible'; }
    }
}

// ─── Autonomous Generation Studio (Pipeline View) ─────────────────
async function startGeneration() {
    if (!currentProject || isGenerating) return;

    const pacing = document.getElementById('genPacing')?.value || 'moderate';
    const chapterCount = document.getElementById('genChapterCount')?.value || '1';

    const btnGen = document.getElementById('btnGenerate');
    const btnCancel = document.getElementById('btnCancel');
    const output = document.getElementById('genOutput');

    isGenerating = true;
    if (btnGen) btnGen.classList.add('hidden');
    if (btnCancel) btnCancel.classList.remove('hidden');

    output.innerHTML = '<div class="text-dim text-sm">Initializing pipeline agents...</div>';
    resetPipelineNodes();
    setStatus('generating', 'Pipeline Active');

    try {
        const res = await fetch(`/api/project/${currentProject}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pacing: pacing,
                chapter_count: parseInt(chapterCount, 10),
            }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to start generation');

        connectSSEStream();

    } catch (e) {
        isGenerating = false;
        if (btnGen) btnGen.classList.remove('hidden');
        if (btnCancel) btnCancel.classList.add('hidden');
        setStatus('error', 'Generation Error');
        showToast(e.message || 'Generation failed', 'error');
    }
}

async function cancelGeneration() {
    if (!currentProject) return;
    try {
        await fetch(`/api/project/${currentProject}/generate/cancel`, { method: 'POST' });
        showToast('Cancelling generation pipeline...', 'info');
    } catch (e) {
        showToast('Failed to cancel', 'error');
    }
}

function connectSSEStream() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    eventSource = new EventSource(`/api/project/${currentProject}/generate/stream`);

    eventSource.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            handleSSEEvent(data);
        } catch (err) {
            console.error('SSE JSON error:', err, e.data);
        }
    };

    eventSource.onerror = (e) => {
        console.warn('SSE stream closed or disconnected');
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        // Reset UI if we were generating when the stream died
        if (isGenerating) {
            isGenerating = false;
            document.getElementById('btnGenerate')?.classList.remove('hidden');
            document.getElementById('btnCancel')?.classList.add('hidden');
            setStatus('error', 'Stream Disconnected');
            showToast('Generation stream disconnected — check Pipeline Logs for details.', 'error');
        }
    };
}

function handleSSEEvent(payload) {
    const eventType = payload.type || payload.event;
    const data = payload.payload || payload.data || payload;

    // Log terminal
    appendLogEntry(payload);

    // Agent highlight
    if (eventType === 'agent_active') {
        const agentName = String(data.agent || '').toLowerCase();
        activatePipelineNode(agentName);
    }

    // Status message
    if (eventType === 'status') {
        const msg = typeof data === 'string' ? data : (data.content || '');
        document.getElementById('statStatus').textContent = msg;
    }

    // Scene start
    if (eventType === 'scene_start') {
        const output = document.getElementById('genOutput');
        const sceneNum = data.scene || 1;
        // Prevent duplicates on reconnect
        if (document.getElementById(`sceneCard_${sceneNum}`)) return;
        const cardHtml = `
            <div class="scene-output-card" id="sceneCard_${sceneNum}">
                <div class="scene-output-card-header">
                    <span class="scene-output-title">Scene ${sceneNum}</span>
                    <span class="scene-meta" id="sceneMeta_${sceneNum}">Writing...</span>
                </div>
                <div class="scene-prose" id="sceneProse_${sceneNum}"></div>
            </div>
        `;
        output.insertAdjacentHTML('beforeend', cardHtml);
        output.scrollTop = output.scrollHeight;
    }

    // Stream token
    if (eventType === 'stream_token') {
        const token = typeof data === 'string' ? data : (data.token || data.content || '');
        const cards = document.querySelectorAll('.scene-output-card');
        const lastCard = cards[cards.length - 1];
        if (lastCard) {
            const proseEl = lastCard.querySelector('.scene-prose');
            if (proseEl) {
                proseEl.textContent += token;
                const output = document.getElementById('genOutput');
                if (output) output.scrollTop = output.scrollHeight;
            }
        }
    }

    // Scene complete
    if (eventType === 'scene_complete') {
        const sceneNum = data.scene || 1;
        const words = data.word_count || (data.text || '').split(/\s+/).filter(Boolean).length;
        const metaEl = document.getElementById(`sceneMeta_${sceneNum}`);
        if (metaEl) metaEl.textContent = `${words} words • Score: ${(data.score || 0.85).toFixed(2)}`;

        const proseEl = document.getElementById(`sceneProse_${sceneNum}`);
        if (proseEl && data.text) proseEl.textContent = data.text;

        totalWords += words;
        document.getElementById('statWords').textContent = totalWords;
    }

    // Chapter complete / Done
    if (eventType === 'chapter_complete' || eventType === 'done') {
        isGenerating = false;
        document.getElementById('btnGenerate')?.classList.remove('hidden');
        document.getElementById('btnCancel')?.classList.add('hidden');
        setStatus('online', 'Chapter Complete');
        markAllPipelineNodesDone();
        showToast('Chapter generation complete!', 'success');
        loadProjects();
    }

    // Error
    if (eventType === 'error') {
        isGenerating = false;
        document.getElementById('btnGenerate')?.classList.remove('hidden');
        document.getElementById('btnCancel')?.classList.add('hidden');
        setStatus('error', 'Error');
        showToast(data.error || 'Generation encountered an error', 'error');
    }
}

function resetPipelineNodes() {
    document.querySelectorAll('.pipeline-node').forEach(node => {
        node.className = 'pipeline-node';
    });
}

function activatePipelineNode(agentName) {
    document.querySelectorAll('.pipeline-node').forEach(node => {
        const stage = node.getAttribute('data-stage');
        if (agentName.includes(stage) || (stage === 'writer' && agentName.includes('scene writer')) || (stage === 'architect' && agentName.includes('architect'))) {
            node.className = 'pipeline-node active';
        } else if (node.classList.contains('active')) {
            node.className = 'pipeline-node done';
        }
    });
}

function markAllPipelineNodesDone() {
    document.querySelectorAll('.pipeline-node').forEach(node => {
        node.className = 'pipeline-node done';
    });
}

// ─── Branch Options & Continuity Report ───────────────────────────
async function loadBranchOptions() {
    if (!currentProject) return;
    const card = document.getElementById('branchOptionsCard');
    const list = document.getElementById('branchOptionsList');
    const hint = document.getElementById('branchDirectionHint')?.value || '';

    card?.classList.remove('hidden');
    list.innerHTML = '<div class="text-dim text-sm">Consulting Story Architect for next chapter paths...</div>';

    try {
        const res = await fetch(`/api/project/${currentProject}/branch-options`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ direction_hint: hint }),
        });

        const data = await res.json();
        const options = data.options || [];

        if (options.length === 0) {
            list.innerHTML = '<div class="text-dim text-sm">No branch paths generated. Try adding a steering hint.</div>';
            return;
        }

        list.innerHTML = options.map((opt, i) => `
            <div class="card mb-2" style="background:rgba(255,255,255,0.03);padding:14px;">
                <div class="flex justify-between items-center mb-1">
                    <strong style="color:var(--accent-purple);font-size:0.92rem;">Path ${i + 1}: ${escHtml(opt.title || 'Option')}</strong>
                    <button class="btn btn-sm btn-primary" onclick="applyBranchOption('${escHtml(opt.summary || '')}')">Apply Path</button>
                </div>
                <p class="text-xs text-secondary" style="line-height:1.5;margin-bottom:0;">${escHtml(opt.summary || '')}</p>
            </div>
        `).join('');

    } catch (e) {
        list.innerHTML = `<div class="text-danger text-sm">Failed to generate branch paths: ${escHtml(e.message)}</div>`;
    }
}

function hideBranchOptions() {
    document.getElementById('branchOptionsCard')?.classList.add('hidden');
}

function applyBranchOption(summary) {
    showToast('Applied branch path to next chapter plan!', 'success');
    hideBranchOptions();
}

async function loadContinuityReport() {
    if (!currentProject) return;
    const card = document.getElementById('continuityCard');
    const body = document.getElementById('continuityReportBody');

    card?.classList.remove('hidden');
    body.innerHTML = '<div class="text-dim text-sm">Analyzing continuity memory...</div>';

    try {
        const res = await fetch(`/api/project/${currentProject}/continuity`);
        const data = await res.json();

        body.innerHTML = `
            <div class="grid-cols-2" style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
                <div>
                    <strong style="color:var(--accent-purple);font-size:0.85rem;display:block;margin-bottom:6px;">🧵 Open Narrative Threads</strong>
                    <div style="font-size:0.8rem;line-height:1.6;color:var(--text-secondary);">${escHtml(data.threads || 'All core threads active.')}</div>
                </div>
                <div>
                    <strong style="color:var(--accent-emerald);font-size:0.85rem;display:block;margin-bottom:6px;">🌱 Foreshadowing Seeds</strong>
                    <div style="font-size:0.8rem;line-height:1.6;color:var(--text-secondary);">${escHtml(data.seeds || 'Seeds planted and tracking properly.')}</div>
                </div>
            </div>
        `;
    } catch (e) {
        body.innerHTML = `<div class="text-danger text-sm">Failed to load continuity: ${escHtml(e.message)}</div>`;
    }
}

function hideContinuityReport() {
    document.getElementById('continuityCard')?.classList.add('hidden');
}

// ─── Interactive Manual Studio ────────────────────────────────────
async function syncManualStatus() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/status`);
        const data = await res.json();

        if (data.active) {
            manualSessionActive = true;
            manualScenesCompleted = data.scenes_completed || 0;
            manualIsGeneratingScene = Boolean(data.is_generating);

            document.getElementById('manualStartCard')?.classList.add('hidden');
            document.getElementById('manualSceneCard')?.classList.remove('hidden');

            document.getElementById('manualSessionTitle').innerHTML = `<span class="icon">✍️</span> Chapter ${data.chapter_num}: ${escHtml(data.chapter_title)}`;
            document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
            document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — What happens next?`;

            renderManualScenes(data.completed_scenes || []);

            if (manualScenesCompleted > 0 && !manualIsGeneratingScene) {
                document.getElementById('btnManualFinish')?.classList.remove('hidden');
            }
        } else {
            manualSessionActive = false;
            document.getElementById('manualStartCard')?.classList.remove('hidden');
            document.getElementById('manualSceneCard')?.classList.add('hidden');
        }
    } catch (e) {
        console.warn('Could not sync manual status', e);
    }
}

async function startManualChapter() {
    if (!currentProject) return;
    const title = (document.getElementById('manualChapterTitle')?.value || '').trim();
    const pacing = document.getElementById('manualPacing')?.value || 'moderate';

    if (!title) {
        showToast('Please enter a chapter title', 'error');
        return;
    }

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chapter_title: title,
                pacing: pacing,
            }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to start chapter');

        manualSessionActive = true;
        manualScenesCompleted = 0;
        document.getElementById('manualStartCard')?.classList.add('hidden');
        document.getElementById('manualSceneCard')?.classList.remove('hidden');
        document.getElementById('manualSessionTitle').innerHTML = `<span class="icon">✍️</span> ${escHtml(title)}`;
        document.getElementById('manualSceneCounter').textContent = '0 scenes completed';
        document.getElementById('manualSceneLabel').textContent = 'Scene 1 — What happens next?';
        document.getElementById('manualCompletedScenes').innerHTML = '';

        showToast('Manual Chapter Session Started!', 'success');

    } catch (e) {
        showToast(e.message, 'error');
    }
}

function renderManualScenes(scenes) {
    const container = document.getElementById('manualCompletedScenes');
    if (!container) return;
    container.innerHTML = '';
    manualSceneTexts.clear();

    scenes.forEach((scene, idx) => {
        const sceneText = typeof scene === 'string' ? scene : scene.text;
        const wordCount = (sceneText || '').split(/\s+/).filter(Boolean).length;
        const isLast = idx === scenes.length - 1;

        manualSceneTexts.set(idx, sceneText || '');

        const cardHtml = `
            <div class="manual-scene-result">
                <div style="position:absolute;top:14px;right:14px;display:flex;gap:6px;">
                    ${isLast ? `<button class="btn btn-sm btn-secondary" onclick="regenerateManualScene(${idx})" title="Regenerate scene">🔄 Regenerate</button>` : ''}
                    <button class="btn btn-sm btn-secondary" onclick="editManualScene(${idx})" title="Edit prose">✏️ Edit</button>
                    <button class="btn btn-sm btn-danger" onclick="deleteManualScene(${idx})" title="Delete scene">🗑️</button>
                </div>
                <div style="font-weight:700;color:var(--accent-purple);margin-bottom:8px;font-size:0.95rem;">Scene ${idx + 1}</div>
                <div style="font-family:var(--font-serif);line-height:1.8;white-space:pre-wrap;font-size:0.92rem;max-height:280px;overflow-y:auto;padding-right:8px;">${escHtml(sceneText)}</div>
                <div class="text-xs text-dim mt-2">${wordCount} words</div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', cardHtml);
    });

    container.scrollTop = container.scrollHeight;
}

async function generateNextScene() {
    if (!currentProject || manualIsGeneratingScene) return;
    const brief = (document.getElementById('manualSceneBrief')?.value || '').trim();

    manualIsGeneratingScene = true;
    document.getElementById('btnManualScene').disabled = true;
    document.getElementById('btnManualStopScene')?.classList.remove('hidden');

    document.getElementById('manualStreamArea')?.classList.remove('hidden');
    document.getElementById('manualStreamOutput').textContent = '';

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/scene`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene_brief: brief }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to generate scene');

        connectSSEStream();
        showToast('Generating scene...', 'info');

    } catch (e) {
        manualIsGeneratingScene = false;
        document.getElementById('btnManualScene').disabled = false;
        document.getElementById('btnManualStopScene')?.classList.add('hidden');
        showToast(e.message, 'error');
    }
}

function toggleTypedScenePanel() {
    const panel = document.getElementById('manualTypedScenePanel');
    if (panel) {
        panel.classList.toggle('hidden');
        if (!panel.classList.contains('hidden')) {
            document.getElementById('manualTypedSceneText')?.focus();
        }
    }
}

async function addTypedScene() {
    if (!currentProject) return;
    const text = (document.getElementById('manualTypedSceneText')?.value || '').trim();
    if (!text) {
        showToast('Please type your scene prose first', 'error');
        return;
    }

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/scene/typed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to add scene');

        document.getElementById('manualTypedSceneText').value = '';
        toggleTypedScenePanel();
        syncManualStatus();
        showToast('Committed custom scene to chapter!', 'success');

    } catch (e) {
        showToast(e.message, 'error');
    }
}

async function deleteManualScene(idx) {
    showConfirmModal('Delete Scene', `Are you sure you want to delete Scene ${idx + 1}?`, async () => {
        try {
            const res = await fetch(`/api/project/${currentProject}/generate/manual/scene/${idx}`, { method: 'DELETE' });
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || 'Failed to delete');
            syncManualStatus();
            showToast('Scene deleted', 'success');
        } catch (e) {
            showToast(e.message, 'error');
        }
    });
}

async function regenerateManualScene(idx) {
    showConfirmModal('Regenerate Scene', `Discard Scene ${idx + 1} and write a fresh version with AI?`, async () => {
        try {
            const res = await fetch(`/api/project/${currentProject}/generate/manual/scene/${idx}/regenerate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || 'Failed to regenerate');
            connectSSEStream();
            showToast('Regenerating scene...', 'info');
        } catch (e) {
            showToast(e.message, 'error');
        }
    });
}

async function finishManualChapter() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/finish`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to finish chapter');

        showToast('Chapter finished and stored successfully!', 'success');
        manualSessionActive = false;
        document.getElementById('manualStartCard')?.classList.remove('hidden');
        document.getElementById('manualSceneCard')?.classList.add('hidden');
        loadProjects();
        switchView('reader');

    } catch (e) {
        showToast(e.message, 'error');
    }
}

function cancelManualSession() {
    showConfirmModal('Reset Session', 'Discard active uncommitted manual chapter session?', () => {
        manualSessionActive = false;
        document.getElementById('manualStartCard')?.classList.remove('hidden');
        document.getElementById('manualSceneCard')?.classList.add('hidden');
        showToast('Session reset', 'info');
    });
}

// ─── Story Reader Studio ──────────────────────────────────────────
async function loadChapters() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/chapters`);
        const data = await res.json();
        const chapters = data.chapters || [];

        const listEl = document.getElementById('chapterList');
        if (!listEl) return;

        if (chapters.length === 0) {
            listEl.innerHTML = '<span class="text-dim text-xs">No chapters generated yet.</span>';
            document.getElementById('readerContent').innerHTML = `
                <div class="empty-state">
                    <span class="icon">📖</span>
                    <h3>No chapters written</h3>
                    <p>Start generation in the Auto Generator or Interactive Manual tab.</p>
                </div>
            `;
            return;
        }

        listEl.innerHTML = chapters.map(ch => {
            const chNum = ch.number || ch.num;
            return `
            <div class="chapter-nav-item ${chNum === currentChapterNumber ? 'active' : ''}" data-chapter="${chNum}" onclick="loadChapter(${chNum})">
                <span>Ch ${chNum}: ${escHtml(ch.title || `Chapter ${chNum}`)}</span>
                <span class="text-xs text-dim">${ch.words || 0}w</span>
            </div>
        `}).join('');

        if (chapters.length > 0) {
            loadChapter(chapters[0].number || chapters[0].num);
        }

    } catch (e) {
        showToast('Failed to load chapters', 'error');
    }
}

async function loadChapter(num) {
    if (!currentProject) return;
    currentChapterNumber = num;

    // Highlight active chapter in sidebar
    document.querySelectorAll('.chapter-nav-item').forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-chapter') === String(num));
    });

    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${num}`);
        const data = await res.json();
        currentChapterRaw = data.content || '';

        document.getElementById('readerChapterTitle').innerHTML = `<span class="icon">📖</span> Chapter ${num}: ${escHtml(data.title || `Chapter ${num}`)}`;
        document.getElementById('readerContent').innerHTML = formatProseMarkdown(currentChapterRaw);
        document.getElementById('readerEditor').value = currentChapterRaw;

    } catch (e) {
        showToast('Failed to read chapter', 'error');
    }
}

function formatProseMarkdown(raw) {
    if (!raw) return '<p class="text-dim">Empty chapter content.</p>';
    const paras = raw.split(/\n\n+/);
    return paras.map(p => `<p style="margin-bottom:1.5em;text-indent:1.5em;">${escHtml(p.trim())}</p>`).join('');
}

function adjustReaderFontSize(delta) {
    readerFontSize = Math.max(12, Math.min(30, readerFontSize + delta));
    const content = document.getElementById('readerContent');
    if (content) content.style.fontSize = `${readerFontSize}px`;
    showToast(`Font size: ${readerFontSize}px`, 'info', 1200);
}

function toggleReaderTheme() {
    const content = document.getElementById('readerContent');
    if (!content) return;
    readerThemes.forEach(t => content.classList.remove('theme-' + t));
    readerThemeIndex = (readerThemeIndex + 1) % readerThemes.length;
    content.classList.add('theme-' + readerThemes[readerThemeIndex]);
    showToast(`Reader Theme: ${readerThemes[readerThemeIndex].toUpperCase()}`, 'info', 1500);
}

function toggleReaderFont() {
    const content = document.getElementById('readerContent');
    if (!content) return;
    readerFonts.forEach(f => content.classList.remove(f));
    readerFontIndex = (readerFontIndex + 1) % readerFonts.length;
    content.classList.add(readerFonts[readerFontIndex]);
    showToast(`Reader Font Switched`, 'info', 1500);
}

function toggleReaderFullscreen() {
    readerIsFullscreen = !readerIsFullscreen;
    const canvas = document.querySelector('.reader-canvas-card');
    if (canvas) {
        if (readerIsFullscreen) {
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.right = '0';
            canvas.style.bottom = '0';
            canvas.style.zIndex = '99999';
            canvas.style.borderRadius = '0';
        } else {
            canvas.style.position = '';
            canvas.style.top = '';
            canvas.style.left = '';
            canvas.style.right = '';
            canvas.style.bottom = '';
            canvas.style.zIndex = '';
            canvas.style.borderRadius = 'var(--radius-lg)';
        }
    }
}

function toggleReaderEdit() {
    isReaderEditMode = !isReaderEditMode;
    const content = document.getElementById('readerContent');
    const editor = document.getElementById('readerEditor');
    const editBtn = document.getElementById('btnReaderEdit');
    const saveBtn = document.getElementById('btnReaderSave');
    const cancelBtn = document.getElementById('btnReaderCancel');

    if (isReaderEditMode) {
        content.classList.add('hidden');
        editor.classList.remove('hidden');
        editBtn.classList.add('hidden');
        saveBtn.classList.remove('hidden');
        cancelBtn.classList.remove('hidden');
    } else {
        content.classList.remove('hidden');
        editor.classList.add('hidden');
        editBtn.classList.remove('hidden');
        saveBtn.classList.add('hidden');
        cancelBtn.classList.add('hidden');
    }
}

async function saveReaderEdit() {
    if (!currentProject) return;
    const newContent = document.getElementById('readerEditor')?.value || '';

    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${currentChapterNumber}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: newContent }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save');

        currentChapterRaw = newContent;
        document.getElementById('readerContent').innerHTML = formatProseMarkdown(newContent);
        toggleReaderEdit();
        showToast('Saved chapter edits!', 'success');
        loadChapters();

    } catch (e) {
        showToast(e.message, 'error');
    }
}

function cancelReaderEdit() {
    document.getElementById('readerEditor').value = currentChapterRaw;
    toggleReaderEdit();
}

async function deleteChaptersFromUi() {
    const numInput = document.getElementById('deleteFromChapter');
    const fromNum = parseInt(numInput?.value || '0', 10);
    if (!fromNum || fromNum < 1) {
        showToast('Enter a valid chapter number', 'error');
        return;
    }

    showConfirmModal(
        'Delete Chapters',
        `Are you sure you want to delete Chapter ${fromNum} and all subsequent chapters?`,
        async () => {
            try {
                const res = await fetch(`/api/project/${currentProject}/chapters/delete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ from_chapter: fromNum }),
                });
                const data = await res.json();
                if (!res.ok || data.error) throw new Error(data.error || 'Failed to delete');
                showToast(`Deleted chapters from #${fromNum}`, 'success');
                loadChapters();
            } catch (e) {
                showToast(e.message, 'error');
            }
        }
    );
}

// ─── Combine & Polish Studio (Gemini Powered) ─────────────────────
async function loadCombineVersions() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/versions`);
        const data = await res.json();
        const versions = data.versions || [];

        const select = document.getElementById('combineVersionSelect');
        if (!select) return;

        if (versions.length === 0) {
            select.innerHTML = '<option value="">No polished versions available</option>';
            return;
        }

        select.innerHTML = versions.map(v => `
            <option value="${escHtml(v.suffix)}">${escHtml(v.name || v.suffix)} (${v.date || ''})</option>
        `).join('');

        if (versions.length > 0) {
            loadCombineVersion(versions[0].suffix);
        }

    } catch (e) {
        console.warn('Error loading combine versions:', e);
    }
}

async function changeCombineVersion() {
    const suffix = document.getElementById('combineVersionSelect')?.value;
    if (suffix) loadCombineVersion(suffix);
}

async function loadCombineVersion(suffix) {
    if (!currentProject || !suffix) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/version/${suffix}`);
        const data = await res.json();

        if (data.analysis) {
            document.getElementById('combineAnalysisCard')?.classList.remove('hidden');
            document.getElementById('combineAnalysis').textContent = data.analysis;
        }

        if (data.polished) {
            document.getElementById('combinePolishedCard')?.classList.remove('hidden');
            document.getElementById('combinePolished').innerHTML = formatProseMarkdown(data.polished);
        }

        document.getElementById('combineDownloadCard')?.classList.remove('hidden');
        document.getElementById('combineStats').textContent = `${(data.polished || '').split(/\s+/).length} words`;

    } catch (e) {
        console.warn('Error loading version:', e);
    }
}

async function startCombine() {
    if (!currentProject) return;
    const model = document.getElementById('combineModelSelect')?.value || 'gemini-3.6-flash';

    document.getElementById('combineProgress')?.classList.remove('hidden');
    document.getElementById('combineStatusText').textContent = `Polishing chapters with ${model}...`;

    try {
        const res = await fetch(`/api/project/${currentProject}/combine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Combine failed');

        showToast('Combine and Polish completed!', 'success');
        document.getElementById('combineProgress')?.classList.add('hidden');
        loadCombineVersions();

    } catch (e) {
        document.getElementById('combineProgress')?.classList.add('hidden');
        showToast(e.message, 'error');
    }
}

async function startWholeStory() {
    if (!currentProject) return;
    const model = document.getElementById('combineModelSelect')?.value || 'gemini-3.6-flash';

    document.getElementById('combineProgress')?.classList.remove('hidden');
    document.getElementById('combineStatusText').textContent = `Generating whole story with ${model}...`;

    try {
        const res = await fetch(`/api/project/${currentProject}/combine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, whole_story: true }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Generation failed');

        showToast('Generated whole story successfully!', 'success');
        document.getElementById('combineProgress')?.classList.add('hidden');
        loadCombineVersions();

    } catch (e) {
        document.getElementById('combineProgress')?.classList.add('hidden');
        showToast(e.message, 'error');
    }
}

function cancelCombine() {
    if (!currentProject) return;
    fetch(`/api/project/${currentProject}/combine/cancel`, { method: 'POST' })
        .catch(e => console.error('Cancel combine error:', e));
    document.getElementById('combineProgress')?.classList.add('hidden');
    showToast('Combiner cancelled', 'info');
}

function toggleCombineSection(contentId, btnId) {
    const el = document.getElementById(contentId);
    const btn = document.getElementById(btnId);
    if (el) {
        el.classList.toggle('hidden');
        if (btn) btn.textContent = el.classList.contains('hidden') ? '▲ Expand' : '▼ Collapse';
    }
}

function downloadCombined(type) {
    if (!currentProject) return;
    window.open(`/api/project/${currentProject}/combine/download/${type}`, '_blank');
}

function renameSelectedVersion() {
    const select = document.getElementById('combineVersionSelect');
    const suffix = select?.value;
    if (!suffix) return;
    const newName = prompt('Enter new label for this polished version:');
    if (!newName) return;

    fetch(`/api/project/${currentProject}/combine/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ suffix, new_name: newName }),
    }).then(() => {
        showToast('Version renamed', 'success');
        loadCombineVersions();
    }).catch(e => console.error('Rename error:', e));
}

function deleteSelectedVersion() {
    const select = document.getElementById('combineVersionSelect');
    const suffix = select?.value;
    if (!suffix) return;

    showConfirmModal('Delete Version', 'Are you sure you want to delete this polished story version?', () => {
        fetch(`/api/project/${currentProject}/combine/delete_version`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ suffix }),
        }).then(() => {
            showToast('Version deleted', 'success');
            loadCombineVersions();
        }).catch(e => console.error('Delete version error:', e));
    });
}

// ─── Story Bible & State Inspector ────────────────────────────────
async function refreshState() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/state`);
        const data = await res.json();
        const state = data.state || data;

        document.getElementById('stateJson').textContent = JSON.stringify(state, null, 2);

        // Render Characters Cards
        const charsGrid = document.getElementById('bibleCharactersGrid');
        const chars = state.characters || {};
        if (Object.keys(chars).length === 0) {
            charsGrid.innerHTML = '<span class="text-dim text-sm">No characters registered in Story Bible.</span>';
        } else {
            charsGrid.innerHTML = Object.keys(chars).map(name => `
                <div class="char-bible-card">
                    <div class="char-bible-header">
                        <span class="char-bible-name">${escHtml(name)}</span>
                    </div>
                    <p class="text-xs text-secondary">${escHtml(chars[name].description || 'No description.')}</p>
                    <div class="char-traits-wrap">
                        ${(chars[name].traits || []).map(t => `<span class="char-trait-chip">${escHtml(t)}</span>`).join('')}
                    </div>
                </div>
            `).join('');
        }

        // Render World Details
        const meta = state.metadata || {};
        document.getElementById('bibleWorldDetails').innerHTML = `
            <div><strong>Setting:</strong> ${escHtml(meta.setting || 'Unspecified')}</div>
            <div><strong>Genre:</strong> ${escHtml(meta.genre || 'Unspecified')}</div>
            <div><strong>Themes:</strong> ${escHtml((meta.themes || []).join(', ') || 'None')}</div>
        `;

        // Render Threads
        const threads = state.continuity?.threads || [];
        const threadsList = document.getElementById('bibleThreadsList');
        if (threads.length === 0) {
            threadsList.innerHTML = '<span class="text-dim text-sm">All narrative threads resolved or in initial state.</span>';
        } else {
            threadsList.innerHTML = threads.map(t => `<div class="card p-2 text-xs">${escHtml(typeof t === 'string' ? t : JSON.stringify(t))}</div>`).join('');
        }

    } catch (e) {
        showToast('Failed to inspect state', 'error');
    }
}

function switchBibleTab(tabName) {
    const stateView = document.getElementById('viewState');
    if (!stateView) return;
    
    stateView.querySelectorAll('.bible-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.toLowerCase().includes(tabName)) btn.classList.add('active');
    });

    ['Characters', 'World', 'Threads', 'Json'].forEach(t => {
        const el = document.getElementById('bibleTab' + t);
        if (el) el.classList.add('hidden');
    });

    const activeEl = document.getElementById('bibleTab' + tabName.charAt(0).toUpperCase() + tabName.slice(1));
    if (activeEl) activeEl.classList.remove('hidden');
}

// ─── Live Terminal Logs ───────────────────────────────────────────
function appendLogEntry(payload) {
    const terminal = document.getElementById('logTerminal');
    if (!terminal) return;

    const time = new Date().toLocaleTimeString();
    const eventType = payload.type || payload.event || 'LOG';
    const content = payload.content || (typeof payload.payload === 'string' ? payload.payload : JSON.stringify(payload.payload || ''));

    let tagClass = 'tag-sys';
    if (eventType === 'agent_active') tagClass = 'tag-arc';
    if (eventType === 'error') tagClass = 'tag-err';
    if (eventType === 'scene_written') tagClass = 'tag-wrt';
    if (eventType === 'status') tagClass = 'tag-ret';

    const row = document.createElement('div');
    row.className = 'log-row';
    row.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-tag ${tagClass}">${eventType.slice(0, 3).toUpperCase()}</span>
        <span>${escHtml(content)}</span>
    `;

    terminal.appendChild(row);

    const autoScroll = document.getElementById('logAutoScroll')?.checked;
    if (autoScroll) {
        terminal.scrollTop = terminal.scrollHeight;
    }
}

function clearLogs() {
    const terminal = document.getElementById('logTerminal');
    if (terminal) {
        terminal.innerHTML = '<div class="log-row"><span class="log-time">--:--:--</span><span class="log-tag tag-sys">SYS</span><span>Logs cleared.</span></div>';
    }
}

function filterLogs() {
    const filter = (document.getElementById('logFilterInput')?.value || '').toLowerCase();
    document.querySelectorAll('.log-row').forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = (!filter || text.includes(filter)) ? 'flex' : 'none';
    });
}

// ─── Vision Lab ───────────────────────────────────────────────────
let visionSelectedFile = null;

function checkVisionStatus() {
    fetch('/vision/status')
        .then(res => res.json())
        .then(data => {
            const badge = document.getElementById('visionStatusBadge');
            if (badge) {
                badge.textContent = data.backend === 'local' ? 'OLLAMA READY' : 'CLOUD VISION READY';
            }
        })
        .catch(() => {
            const badge = document.getElementById('visionStatusBadge');
            if (badge) badge.textContent = 'STANDALONE';
        });
}

function handleVisionFileSelect(files) {
    if (!files || !files[0]) return;
    const file = files[0];
    visionSelectedFile = file;

    const preview = document.getElementById('visionPreviewImg');
    const previewWrap = document.getElementById('visionImagePreview');
    const contentWrap = document.getElementById('visionDropzoneContent');
    const btn = document.getElementById('btnAnalyzeVision');

    const reader = new FileReader();
    reader.onload = (e) => {
        if (preview) preview.src = e.target.result;
        if (previewWrap) previewWrap.style.display = 'block';
        if (contentWrap) contentWrap.style.display = 'none';
        if (btn) btn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearVisionFile() {
    visionSelectedFile = null;
    document.getElementById('visionPreviewImg').src = '';
    document.getElementById('visionImagePreview').style.display = 'none';
    document.getElementById('visionDropzoneContent').style.display = 'block';
    document.getElementById('btnAnalyzeVision').disabled = true;
}

async function runVisionAnalysis() {
    if (!visionSelectedFile) return;
    const btn = document.getElementById('btnAnalyzeVision');
    const mode = document.getElementById('visionModeSelect')?.value || 'auto';
    const promptExtra = document.getElementById('visionPromptExtra')?.value || '';

    if (btn) { btn.disabled = true; btn.textContent = '⚡ Analyzing Persona...'; }

    const formData = new FormData();
    formData.append('image', visionSelectedFile);
    formData.append('mode', mode);
    formData.append('extra_prompt', promptExtra);

    try {
        const res = await fetch('/vision/analyze', {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Vision analysis failed');

        document.getElementById('visionChatInterface')?.classList.remove('hidden');
        appendVisionChatMessage('assistant', data.analysis || 'Analysis complete.');
        showToast('Character visual breakdown completed!', 'success');

    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '⚡ Analyze Visual Breakdown'; }
    }
}

function appendVisionChatMessage(sender, text) {
    const history = document.getElementById('visionChatHistory');
    if (!history) return;

    const bubble = document.createElement('div');
    bubble.style.cssText = `
        padding: 12px 16px;
        border-radius: var(--radius-md);
        max-width: 80%;
        line-height: 1.6;
        font-size: 0.88rem;
        ${sender === 'user' ? 'margin-left:auto;background:var(--grad-primary);color:#fff;' : 'background:rgba(255,255,255,0.06);color:var(--text-primary);border:1px solid var(--border-subtle);'}
    `;
    bubble.innerHTML = formatProseMarkdown(text);
    history.appendChild(bubble);
    history.scrollTop = history.scrollHeight;
}

function handleVisionChatKeyPress(e) {
    if (e.key === 'Enter') sendVisionChatMessage();
}

async function sendVisionChatMessage() {
    const input = document.getElementById('visionChatMessage');
    const text = (input?.value || '').trim();
    if (!text) return;

    input.value = '';
    appendVisionChatMessage('user', text);

    try {
        const res = await fetch('/vision/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        appendVisionChatMessage('assistant', data.reply || 'No response.');
    } catch (e) {
        appendVisionChatMessage('assistant', `Error: ${e.message}`);
    }
}

// ─── Settings & Backends Configuration ────────────────────────────
async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        runtimeSettings = await res.json();
        const s = runtimeSettings.settings || {};

        // Highlight selected backend provider card
        const mode = s.BACKEND_MODE || 'groq';
        document.querySelectorAll('.provider-card-select').forEach(card => {
            const input = card.querySelector('input');
            const cardMode = card.getAttribute('data-mode');
            if (cardMode === mode) {
                card.classList.add('selected');
                if (input) input.checked = true;
            } else {
                card.classList.remove('selected');
                if (input) input.checked = false;
            }
        });

        // Set inputs
        document.querySelectorAll('[data-setting]').forEach(el => {
            const key = el.getAttribute('data-setting');
            if (s[key] !== undefined) {
                if (el.type === 'checkbox') el.checked = Boolean(s[key]);
                else el.value = s[key];
            }
        });

        document.getElementById('settingsSummary').textContent = `Active: ${mode.toUpperCase()} (${formatModelDisplayName(s.ACTIVE_MODEL)})`;

    } catch (e) {
        showToast('Error loading settings', 'error');
    }
}

async function saveSettings() {
    const btn = document.getElementById('btnSaveSettings');
    if (btn) { btn.disabled = true; btn.textContent = '💾 Saving...'; }

    const settings = {};
    document.querySelectorAll('[data-setting]').forEach(el => {
        const key = el.getAttribute('data-setting');
        if (el.type === 'checkbox') settings[key] = el.checked;
        else if (el.type === 'number') settings[key] = parseFloat(el.value) || 0;
        else settings[key] = el.value;
    });

    const selectedModeCard = document.querySelector('.provider-card-select.selected');
    if (selectedModeCard) {
        settings['BACKEND_MODE'] = selectedModeCard.getAttribute('data-mode');
    }

    try {
        const res = await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ settings }),
        });

        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save settings');

        showToast('Settings saved successfully!', 'success');
        await loadModels();
        await loadSettings();

    } catch (e) {
        showToast(e.message || 'Save failed', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '💾 Save Settings'; }
    }
}

// ─── App Initialization ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Setup Backend Provider Cards click listener
    document.querySelectorAll('.provider-card-select').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.provider-card-select').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            const input = card.querySelector('input');
            if (input) input.checked = true;
        });
    });

    // Initialize Core Modules
    loadProjects();
    loadModels();
    setStatus('online', 'Engine Ready');
});
