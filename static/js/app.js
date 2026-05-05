/**
 * Story Teller — Web UI Application Logic
 * Vanilla JS with SSE for live generation updates
 */

// ─── State ───────────────────────────────────────────────────────
let currentProject = null;
let characters = [];
let isGenerating = false;
let eventSource = null;
let totalWords = 0;
let scenesComplete = 0;
let totalScenes = 0;
let streamSceneNodes = new Map();
let plannedChapters = 1;
let completedChapters = 0;

// ─── Navigation ──────────────────────────────────────────────────
function switchView(viewName) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

    const view = document.getElementById('view' + viewName.charAt(0).toUpperCase() + viewName.slice(1));
    const tab = document.querySelector(`.nav-tab[data-view="${viewName}"]`);

    if (view) view.classList.add('active');
    if (tab) tab.classList.add('active');

    if (viewName === 'reader' && currentProject) loadChapters();
    if (viewName === 'state' && currentProject) refreshState();
    if (viewName === 'dashboard') loadProjects();
}

function showProjectTabs() {
    document.getElementById('tabGenerate').classList.remove('hidden');
    document.getElementById('tabReader').classList.remove('hidden');
    document.getElementById('tabLogs').classList.remove('hidden');
    document.getElementById('tabState').classList.remove('hidden');
}

// ─── Projects ────────────────────────────────────────────────────
async function loadProjects() {
    try {
        const res = await fetch('/api/projects');
        const data = await res.json();
        renderProjects(data.projects);
    } catch (e) {
        showToast('Failed to load projects', 'error');
    }
}

function renderProjects(projects) {
    const grid = document.getElementById('projectsGrid');
    if (!projects.length) {
        grid.innerHTML = `
            <div class="empty-state">
                <span class="icon">📚</span>
                <h3>No stories yet</h3>
                <p>Create your first story project to get started.</p>
                <button class="btn btn-primary" onclick="switchView('create')">✨ Create Story</button>
            </div>`;
        return;
    }

    grid.innerHTML = projects.map(p => `
        <div class="card project-card">
            <div onclick="selectProject('${p.name}')" style="cursor:pointer">
                <div class="project-title">${escHtml(p.title)}</div>
                <div class="project-genre">${escHtml(p.genre)}</div>
                <div class="project-stats">
                    <span>📖 ${p.current_chapter} chapters</span>
                    <span>🎬 ${p.total_scenes} scenes</span>
                    <span>👥 ${p.characters.length} characters</span>
                </div>
            </div>
            <div class="flex gap-2 mt-2" style="justify-content:flex-end">
                <button class="btn btn-sm" onclick="event.stopPropagation();exportStory('${p.name}')" title="Export full story">📥 Export</button>
                <button class="btn btn-sm" onclick="event.stopPropagation();resetProject('${p.name}','${escHtml(p.title)}')" title="Reset generated content" style="background:var(--warning);color:#000">🔄 Reset</button>
                <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();deleteProject('${p.name}','${escHtml(p.title)}')" title="Delete project">🗑️ Delete</button>
            </div>
        </div>
    `).join('');
}

async function selectProject(name) {
    currentProject = name;
    document.getElementById('activeProject').textContent = name;
    showProjectTabs();

    try {
        const res = await fetch(`/api/project/${name}`);
        const data = await res.json();
        document.getElementById('genTitle').innerHTML =
            `<span class="icon">✍️</span> ${escHtml(data.info.title)}`;
        switchView('generate');
    } catch (e) {
        showToast('Failed to load project', 'error');
    }
}

// ─── Create Project ──────────────────────────────────────────────
function addCharacter() {
    const name = document.getElementById('charName').value.trim();
    const desc = document.getElementById('charDesc').value.trim();
    const traits = document.getElementById('charTraits').value.trim();

    if (!name) { showToast('Enter a character name', 'error'); return; }

    characters.push({
        name,
        description: desc,
        traits: traits ? traits.split(',').map(t => t.trim()) : [],
    });

    renderCharacterList();
    document.getElementById('charName').value = '';
    document.getElementById('charDesc').value = '';
    document.getElementById('charTraits').value = '';
    document.getElementById('charName').focus();
}

function removeCharacter(index) {
    characters.splice(index, 1);
    renderCharacterList();
}

function renderCharacterList() {
    const list = document.getElementById('characterList');
    list.innerHTML = characters.map((c, i) => `
        <div class="character-entry">
            <span class="char-name">${escHtml(c.name)}</span>
            <span>${escHtml(c.description)}</span>
            <span class="text-dim">${c.traits.join(', ')}</span>
            <span class="char-remove" onclick="removeCharacter(${i})">✕</span>
        </div>
    `).join('');
}

async function createProject() {
    const title = document.getElementById('createTitle').value.trim();
    const genre = document.getElementById('createGenre').value;
    const premise = document.getElementById('createPremise').value.trim();
    const setting = document.getElementById('createSetting').value.trim();
    const themes = document.getElementById('createThemes').value.trim();

    if (!title) { showToast('Enter a title', 'error'); return; }
    if (!premise) { showToast('Enter a premise', 'error'); return; }

    const charObj = {};
    characters.forEach(c => {
        charObj[c.name] = { description: c.description, traits: c.traits };
    });

    const btn = document.getElementById('btnCreateProject');
    btn.disabled = true;
    btn.textContent = '⏳ Creating...';

    try {
        const res = await fetch('/api/project/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title, genre, premise, setting,
                themes: themes ? themes.split(',').map(t => t.trim()) : [],
                characters: charObj,
            }),
        });

        const data = await res.json();
        if (data.status === 'ok') {
            showToast(`Project "${title}" created!`, 'success');
            characters = [];
            renderCharacterList();
            currentProject = data.project;
            document.getElementById('activeProject').textContent = data.project;
            showProjectTabs();
            switchView('generate');
            document.getElementById('genTitle').innerHTML =
                `<span class="icon">✍️</span> ${escHtml(title)}`;
        } else {
            showToast(data.error || 'Creation failed', 'error');
        }
    } catch (e) {
        showToast('Failed to create project', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Create Project';
    }
}

// ─── Generation ──────────────────────────────────────────────────
async function startGeneration() {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    if (isGenerating) return;

    isGenerating = true;
    totalWords = 0;
    scenesComplete = 0;
    totalScenes = 0;
    streamSceneNodes = new Map();
    plannedChapters = 1;
    completedChapters = 0;

    document.getElementById('btnGenerate').classList.add('hidden');
    document.getElementById('btnCancel').classList.remove('hidden');
    document.getElementById('genOutput').innerHTML = '';
    setStatus('working', 'Generating...');
    resetPipelineStages();

    const pacing = document.getElementById('genPacing').value;
    const chapterCount = parseInt(document.getElementById('genChapterCount').value) || 1;
    const selectedModel = document.getElementById('modelSelect')?.value || '';

    try {
        const res = await fetch(`/api/project/${currentProject}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pacing, chapter_count: chapterCount, model: selectedModel }),
        });

        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.error || 'Failed to start generation');
        }
        const startData = await res.json().catch(() => ({}));
        plannedChapters = startData.chapters || chapterCount || 1;

        // Connect SSE
        eventSource = new EventSource(`/api/project/${currentProject}/generate/stream`);
        eventSource.onmessage = handleSSE;
        eventSource.onerror = () => {
            stopGeneration();
            showToast('Connection lost', 'error');
        };
    } catch (e) {
        stopGeneration();
        showToast(e.message, 'error');
    }
}

function handleSSE(event) {
    const msg = JSON.parse(event.data);
    const evt = msg.event || msg.type;
    const data = msg.data ?? msg.payload ?? {};

    switch (evt) {
        case 'chapter_start':
            scenesComplete = 0;
            totalScenes = 0;
            streamSceneNodes = new Map();
            resetPipelineStages();
            updateStats({ scenes: '0/0', status: `Starting chapter ${data.chapter}` });
            if (data.chapter > 1) {
                appendOutput('<hr style="border:none;border-top:2px solid var(--accent-primary);margin:32px 0">');
            }
            break;

        case 'agent_active':
            updatePipelineStage(data.agent);
            updateStats({ status: `${data.agent}: ${data.step}` });
            break;

        case 'chapter_planned':
            appendOutput(`<h2 style="color:var(--accent-primary);font-family:'Playfair Display',serif;margin:16px 0 8px">Chapter: ${escHtml(data.plan.chapter_title)}</h2>`);
            appendOutput(`<p class="text-dim text-sm" style="margin-bottom:20px"><em>Tone: ${escHtml(data.plan.tone)} | Events: ${data.plan.key_events.join(', ')}</em></p>`);
            break;

        case 'scenes_planned':
            totalScenes = data.count;
            renderSceneProgress();
            break;

        case 'scene_start':
            updateStats({ status: `Writing scene ${data.scene}/${data.total}` });
            break;

        case 'token':
            appendToken(data.scene || 'current', data.content || '');
            if (data.provider) {
                updateStats({
                    provider: data.provider === 'groq' ? '☁️ Groq' : '💻 llama.cpp',
                });
            }
            break;

        case 'scene_written':
            totalWords += data.words;
            updateStats({
                words: totalWords,
                provider: data.provider === 'groq' ? '☁️ Groq' : '💻 llama.cpp',
            });
            break;

        case 'scene_complete':
            scenesComplete++;
            updateStats({ scenes: `${scenesComplete}/${totalScenes}` });
            updateSceneDot(scenesComplete - 1, 'done');
            if (scenesComplete < totalScenes) updateSceneDot(scenesComplete, 'active');
            // Show scene text progressively
            if (data.text && !streamSceneNodes.has(data.scene)) {
                if (scenesComplete > 1) {
                    appendOutput('<hr style="border:none;border-top:1px solid var(--border);margin:24px 0">');
                }
                appendOutput(`<p>${escHtml(data.text).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`);
            }
            break;

        case 'chapter_complete':
            completedChapters += 1;
            showToast(
                `Chapter ${data.chapter_number} complete! ${completedChapters}/${plannedChapters}`,
                'success',
            );
            updateStats({ status: `Completed chapter ${completedChapters}/${plannedChapters}` });
            break;

        case 'log':
            addLogEntry(data);
            break;

        case 'trace':
            addLogEntry({
                timestamp: new Date().toLocaleTimeString(),
                level: 'debug',
                message: `Trace ${data.agent} (${Math.round(data.latency_ms || 0)}ms)`,
                details: { step: data.step, next_action: data.next_action },
            });
            break;

        case 'status':
            updateStats({ status: data });
            break;

        case 'error':
            showToast(data.error || 'Generation failed', 'error');
            addLogEntry({ timestamp: new Date().toLocaleTimeString(), level: 'error', message: data.error || 'Generation failed' });
            stopGeneration();
            break;

        case 'done':
            stopGeneration();
            break;

        case 'heartbeat':
            break;
    }
}

async function loadChapterContent(num) {
    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${num}`);
        const data = await res.json();
        if (data.content) {
            const html = markdownToHtml(data.content);
            document.getElementById('genOutput').innerHTML = html;
        }
    } catch (e) { /* ignore */ }
}

async function cancelGeneration() {
    if (!currentProject) return;
    try {
        await fetch(`/api/project/${currentProject}/generate/cancel`, { method: 'POST' });
        showToast('Cancellation requested', 'info');
    } catch (e) { /* ignore */ }
    stopGeneration();
}

function stopGeneration() {
    isGenerating = false;
    document.getElementById('btnGenerate').classList.remove('hidden');
    document.getElementById('btnCancel').classList.add('hidden');
    setStatus('online', 'Ready');
    if (eventSource) { eventSource.close(); eventSource = null; }
}

// ─── Pipeline Stages ─────────────────────────────────────────────
const agentToStage = {
    'Retriever': 'retriever',
    'Story Architect': 'architect',
    'Scene Planner': 'planner',
    'Scene Writer': 'writer',
    'Consistency Engine': 'consistency',
    'Editor': 'editor',
};

function resetPipelineStages() {
    document.querySelectorAll('.pipeline-stage').forEach(s => {
        s.className = 'pipeline-stage idle';
    });
    document.getElementById('sceneProgress').innerHTML = '';
}

function updatePipelineStage(agentName) {
    const stageKey = agentToStage[agentName];
    if (!stageKey) return;

    // Mark previous active as done
    document.querySelectorAll('.pipeline-stage.active').forEach(s => {
        s.classList.remove('active');
        s.classList.add('done');
    });

    const stage = document.querySelector(`[data-stage="${stageKey}"]`);
    if (stage) {
        stage.classList.remove('idle');
        stage.classList.add('active');
    }
}

function renderSceneProgress() {
    const container = document.getElementById('sceneProgress');
    container.innerHTML = Array.from({ length: totalScenes }, (_, i) =>
        `<div class="scene-dot${i === 0 ? ' active' : ''}" data-scene="${i}"></div>`
    ).join('');
}

function updateSceneDot(index, state) {
    const dot = document.querySelector(`.scene-dot[data-scene="${index}"]`);
    if (dot) {
        dot.className = `scene-dot ${state}`;
    }
}

// ─── Chapter Reader ──────────────────────────────────────────────
async function loadChapters() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/chapters`);
        const data = await res.json();
        renderChapterList(data.chapters);
    } catch (e) {
        showToast('Failed to load chapters', 'error');
    }
}

function renderChapterList(chapters) {
    const list = document.getElementById('chapterList');
    if (!chapters.length) {
        list.innerHTML = '<p class="text-dim text-sm">No chapters yet.</p>';
        return;
    }
    list.innerHTML = chapters.map(ch => `
        <div class="chapter-item" onclick="readChapter(${ch.number})">
            <div><span class="ch-num">Chapter ${ch.number}</span></div>
            <div class="ch-words">${ch.words} words</div>
        </div>
    `).join('');
}

async function readChapter(num) {
    // Highlight active
    document.querySelectorAll('.chapter-item').forEach(i => i.classList.remove('active'));
    event.currentTarget?.classList.add('active');

    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${num}`);
        const data = await res.json();
        if (data.content) {
            document.getElementById('readerContent').innerHTML = markdownToHtml(data.content);
        }
    } catch (e) {
        showToast('Failed to load chapter', 'error');
    }
}

// ─── State Inspector ─────────────────────────────────────────────
async function refreshState() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/state`);
        const data = await res.json();
        document.getElementById('stateJson').innerHTML = syntaxHighlightJson(
            JSON.stringify(data.state, null, 2)
        );
    } catch (e) {
        showToast('Failed to load state', 'error');
    }
}

// ─── Utilities ───────────────────────────────────────────────────
function appendOutput(html) {
    const output = document.getElementById('genOutput');
    output.innerHTML += html;
    output.scrollTop = output.scrollHeight;
}

function appendToken(scene, text) {
    if (!text) return;
    const output = document.getElementById('genOutput');
    let node = streamSceneNodes.get(scene);
    if (!node) {
        if (streamSceneNodes.size > 0 || output.textContent.trim()) {
            output.insertAdjacentHTML('beforeend', '<hr style="border:none;border-top:1px solid var(--border);margin:24px 0">');
        }
        const wrapper = document.createElement('div');
        wrapper.className = 'stream-scene';
        const paragraph = document.createElement('p');
        paragraph.style.whiteSpace = 'pre-wrap';
        wrapper.appendChild(paragraph);
        output.appendChild(wrapper);
        node = paragraph;
        streamSceneNodes.set(scene, node);
    }
    node.appendChild(document.createTextNode(text));
    output.scrollTop = output.scrollHeight;
}

function updateStats(updates) {
    if (updates.words !== undefined) document.getElementById('statWords').textContent = updates.words;
    if (updates.scenes !== undefined) document.getElementById('statScenes').textContent = updates.scenes;
    if (updates.provider !== undefined) document.getElementById('statProvider').textContent = updates.provider;
    if (updates.status !== undefined) document.getElementById('statStatus').textContent = updates.status;
}

function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    dot.className = 'status-dot' + (state === 'offline' ? ' offline' : state === 'working' ? ' working' : '');
    document.getElementById('statusText').textContent = text;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastOut 300ms ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function markdownToHtml(md) {
    return md
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^\* \* \*$/gm, '<hr style="border:none;border-top:1px solid var(--border);margin:24px 0">')
        .replace(/\n\n/g, '</p><p>')
        .replace(/^(?!<[h|p|hr])/gm, '')
        .replace(/^(.+)$/gm, (match) => {
            if (match.startsWith('<')) return match;
            return `<p>${match}</p>`;
        });
}

function syntaxHighlightJson(json) {
    return json
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"([^"]+)"(?=\s*:)/g, '<span class="json-key">"$1"</span>')
        .replace(/:\s*"([^"]*?)"/g, ': <span class="json-string">"$1"</span>')
        .replace(/:\s*(\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
        .replace(/:\s*(true|false)/g, ': <span class="json-boolean">$1</span>')
        .replace(/:\s*(null)/g, ': <span class="json-null">$1</span>');
}

// ─── Live Logs ───────────────────────────────────────────────────
function addLogEntry(entry) {
    const terminal = document.getElementById('logTerminal');
    if (!terminal) return;

    const level = entry.level || 'info';
    const badgeMap = {
        info: 'INFO', success: 'OK', warn: 'WARN',
        error: 'ERR', header: '>>>',
    };

    let html = `<div class="log-entry log-${level}">`;
    html += `<span class="log-time">${escHtml(entry.timestamp || '--:--:--.---')}</span>`;
    html += `<span class="log-badge log-badge-${level}">${badgeMap[level] || 'LOG'}</span>`;
    html += `<span class="log-msg">${escHtml(entry.message)}`;

    // Render details inline
    if (entry.details && typeof entry.details === 'object') {
        html += `<div class="log-details">`;
        for (const [key, val] of Object.entries(entry.details)) {
            const display = Array.isArray(val) ? val.join(', ') :
                           (typeof val === 'object' ? JSON.stringify(val) : val);
            html += `<span><span class="detail-key">${escHtml(key)}</span>=<span class="detail-val">${escHtml(String(display))}</span></span>`;
        }
        html += `</div>`;
    }

    html += `</span></div>`;
    terminal.insertAdjacentHTML('beforeend', html);

    // Auto-scroll
    const autoScroll = document.getElementById('logAutoScroll');
    if (autoScroll && autoScroll.checked) {
        terminal.scrollTop = terminal.scrollHeight;
    }
}

function clearLogs() {
    const terminal = document.getElementById('logTerminal');
    if (terminal) {
        terminal.innerHTML = `<div class="log-entry log-info">
            <span class="log-time">--:--:--.---</span>
            <span class="log-badge log-badge-info">SYS</span>
            <span class="log-msg">Logs cleared. Waiting for generation...</span>
        </div>`;
    }
}

// ─── Project Management ──────────────────────────────────────────
async function deleteProject(name, title) {
    if (!confirm(`Delete "${title || name}"? This cannot be undone.`)) return;

    try {
        const res = await fetch(`/api/project/${name}/delete`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            showToast(`"${title || name}" deleted`, 'success');
            if (currentProject === name) {
                currentProject = null;
                document.getElementById('activeProject').textContent = 'No project selected';
            }
            loadProjects();
        } else {
            showToast(data.error || 'Delete failed', 'error');
        }
    } catch (e) {
        showToast('Failed to delete project', 'error');
    }
}

async function resetProject(name, title) {
    if (!confirm(`Reset "${title || name}"?\n\nThis will delete all generated chapters, scenes, and story memory.\nYour premise, characters, and settings will be kept.`)) return;

    try {
        const res = await fetch(`/api/project/${name}/reset`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            showToast(`"${title || name}" reset — ready to regenerate`, 'success');
            loadProjects();
        } else {
            showToast(data.error || 'Reset failed', 'error');
        }
    } catch (e) {
        showToast('Failed to reset project', 'error');
    }
}

function exportStory(name) {
    window.open(`/api/project/${name}/export`, '_blank');
    showToast('Exporting story...', 'info');
}

// ─── Keyboard Shortcuts ──────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Enter in character inputs → add character
    if (e.key === 'Enter' && (e.target.id === 'charName' || e.target.id === 'charDesc' || e.target.id === 'charTraits')) {
        e.preventDefault();
        addCharacter();
    }
});

// ─── Model Management ────────────────────────────────────────────
async function loadModels() {
    const select = document.getElementById('modelSelect');
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        if (!data.models.length) {
            select.innerHTML = '<option value="">No GGUF models found</option>';
            return;
        }
        select.innerHTML = data.models.map(m =>
            `<option value="${escHtml(m)}" ${m === data.active ? 'selected' : ''}>${escHtml(m)}</option>`
        ).join('');
    } catch (e) {
        select.innerHTML = '<option>Error loading models</option>';
    }
}

async function switchModel(model) {
    try {
        const res = await fetch('/api/models/switch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model}),
        });
        const data = await res.json();
        if (data.status === 'ok') {
            showToast(`Switched to ${data.active || model}`, 'success');
        } else {
            showToast(data.error || 'Switch failed', 'error');
            loadModels();
        }
    } catch (e) {
        showToast('Failed to switch model', 'error');
    }
}

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    loadModels();
    setStatus('online', 'Ready');
});
