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
const GROQ_MODEL_OPTION = '__groq_api__';

// Manual interactive session state
let manualSessionActive = false;
let manualScenesCompleted = 0;
let manualChapterNum = 0;
let manualIsGeneratingScene = false;

function isGroqProvider(provider) {
    return String(provider || '').toLowerCase().includes('groq');
}

function providerBadge(provider) {
    return isGroqProvider(provider) ? '☁️ Groq' : '💻 llama.cpp';
}

function modelOptionLabel(modelId, groqModelName) {
    if (modelId === GROQ_MODEL_OPTION) {
        return `☁️ Groq API (${groqModelName || 'default'})`;
    }
    return `💻 ${modelId}`;
}

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
    if (viewName === 'manual' && currentProject) {
        // Fetch the current manual session status to restore UI state
        fetch(`/api/project/${currentProject}/generate/manual/status`)
            .then(res => res.json())
            .then(data => {
                if (data.active) {
                    manualSessionActive = true;
                    manualScenesCompleted = data.scenes_completed;
                    document.getElementById('manualStartCard')?.classList.add('hidden');
                    document.getElementById('manualSceneCard')?.classList.remove('hidden');
                    document.getElementById('manualSessionTitle').innerHTML = `<span class="icon">✍️</span> Chapter ${data.chapter_num}: ${escHtml(data.chapter_title)}`;
                    document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
                    document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen`;
                    if (data.completed_scenes) {
                        renderManualScenes(data.completed_scenes);
                    }
                    if (manualScenesCompleted > 0) {
                        document.getElementById('btnManualFinish').classList.remove('hidden');
                    } else {
                        document.getElementById('btnManualFinish').classList.add('hidden');
                    }
                } else {
                    manualSessionActive = false;
                    document.getElementById('manualStartCard')?.classList.remove('hidden');
                    document.getElementById('manualSceneCard')?.classList.add('hidden');
                }
            })
            .catch(() => showToast('Failed to sync manual session state', 'error'));
    }
}

function renderManualScenes(scenes) {
    const completedDiv = document.getElementById('manualCompletedScenes');
    completedDiv.innerHTML = '';
    if (scenes && scenes.length > 0) {
        scenes.forEach((scene, idx) => {
            // Handle both simple strings (from DB fallback) and dicts (from generate_manual_scene)
            const sceneText = typeof scene === 'string' ? scene : scene.text;
            const enhancedSummary = typeof scene === 'string' ? '' : (scene.enhanced_summary || '');
            const words = typeof scene === 'string' ? '' : (scene.words || '');
            
            const sceneHtml = `
                <div class="manual-scene-result" style="position:relative; margin-bottom:16px;padding:16px;background:var(--surface-2);border-radius:8px;border-left:3px solid var(--accent-primary)">
                    <button class="btn btn-sm btn-danger" style="position:absolute;top:12px;right:12px;z-index:10;font-size:0.8rem;padding:4px 8px" onclick="deleteManualScene(${idx})" title="Delete this scene">🗑️</button>
                    <div style="font-weight:600;margin-bottom:8px;color:var(--accent-primary);padding-right:40px">Scene ${idx + 1}</div>
                    ${enhancedSummary ? `<div class="text-dim text-sm" style="margin-bottom:6px">Enhanced: ${escHtml(enhancedSummary)}</div>` : ''}
                    <div style="white-space:pre-wrap;line-height:1.7;font-family:'Lora',serif;font-size:0.92rem;max-height:300px;overflow-y:auto">${escHtml(sceneText || '')}</div>
                    ${words ? `<div class="text-dim text-sm" style="margin-top:8px">${words} words</div>` : ''}
                </div>
            `;
            completedDiv.insertAdjacentHTML('beforeend', sceneHtml);
        });
    }
    completedDiv.scrollTop = completedDiv.scrollHeight;
}

async function deleteManualScene(index) {
    if (!confirm('Are you sure you want to delete this scene?')) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/scene/${index}`, { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to delete scene');
        
        manualScenesCompleted = data.scenes_completed;
        document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
        document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen`;
        renderManualScenes(data.completed_scenes);
        
        if (manualScenesCompleted === 0) {
            document.getElementById('btnManualFinish').classList.add('hidden');
        }
        showToast('Scene deleted', 'success');
    } catch (e) {
        showToast(e.message, 'error');
    }
}

function showProjectTabs() {
    document.getElementById('tabGenerate').classList.remove('hidden');
    document.getElementById('tabManual').classList.remove('hidden');
    document.getElementById('tabReader').classList.remove('hidden');
    document.getElementById('tabLogs').classList.remove('hidden');
    document.getElementById('tabState').classList.remove('hidden');
    document.getElementById('tabCombine').classList.remove('hidden');
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

// Old batch manual helpers (kept for backwards compat but unused in new flow)
function syncManualSceneInputs() { /* no-op in interactive mode */ }

// ─── Interactive Manual Generation ───────────────────────────────
async function startManualChapter() {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    if (isGenerating || manualSessionActive) return;

    const chapterTitle = document.getElementById('manualChapterTitle')?.value.trim() || '';
    const pacing = document.getElementById('manualPacing')?.value || 'moderate';
    const selectedModel = document.getElementById('modelSelect')?.value || '';

    if (!chapterTitle) {
        showToast('Enter a chapter heading', 'error');
        return;
    }

    const btn = document.getElementById('btnManualStart');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Starting...'; }

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chapter_title: chapterTitle, pacing, model: selectedModel }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to start manual session');
        }

        // Session started successfully
        manualSessionActive = true;
        manualScenesCompleted = 0;
        manualChapterNum = data.chapter_num || 0;
        manualIsGeneratingScene = false;

        // Connect SSE for live streaming
        eventSource = new EventSource(`/api/project/${currentProject}/generate/stream`);
        eventSource.onmessage = handleSSE;
        eventSource.onerror = () => {
            // SSE errors in manual mode are recoverable — don't kill the session
            if (eventSource) { eventSource.close(); eventSource = null; }
        };

        // Switch UI to scene input phase
        document.getElementById('manualStartCard').classList.add('hidden');
        document.getElementById('manualSceneCard').classList.remove('hidden');
        document.getElementById('manualSessionTitle').innerHTML =
            `<span class="icon">✍️</span> Chapter ${data.chapter_num}: ${escHtml(data.chapter_title)}`;
        document.getElementById('manualSceneCounter').textContent = '0 scenes completed';
        document.getElementById('manualCompletedScenes').innerHTML = '';
        document.getElementById('manualSceneLabel').textContent = 'Scene 1 — Describe what should happen';
        document.getElementById('manualSceneBrief').value = '';
        document.getElementById('btnManualFinish').classList.add('hidden');
        document.getElementById('btnManualCancel').classList.remove('hidden');

        setStatus('online', 'Manual session active');
        showToast(`Chapter ${data.chapter_num} session started!`, 'success');

    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '🚀 Start Chapter'; }
    }
}

async function generateNextScene() {
    if (!currentProject || !manualSessionActive || manualIsGeneratingScene) return;

    const brief = document.getElementById('manualSceneBrief')?.value.trim() || '';
    if (!brief) {
        showToast('Enter a scene description', 'error');
        return;
    }

    manualIsGeneratingScene = true;
    const btn = document.getElementById('btnManualScene');
    const stopBtn = document.getElementById('btnManualStopScene');
    const finishBtn = document.getElementById('btnManualFinish');
    const statusEl = document.getElementById('manualSceneStatus');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Generating...'; }
    if (stopBtn) stopBtn.classList.remove('hidden');
    if (finishBtn) finishBtn.classList.add('hidden');
    if (statusEl) { statusEl.classList.remove('hidden'); statusEl.textContent = 'Enhancing and writing scene...'; }

    // Show streaming area
    document.getElementById('manualStreamArea').classList.remove('hidden');
    document.getElementById('manualStreamOutput').textContent = '';
    streamSceneNodes = new Map();

    // Reconnect SSE if needed
    if (!eventSource) {
        eventSource = new EventSource(`/api/project/${currentProject}/generate/stream`);
        eventSource.onmessage = handleSSE;
        eventSource.onerror = () => {
            if (eventSource) { eventSource.close(); eventSource = null; }
        };
    }

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/scene`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scene_brief: brief }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to start scene generation');
        }
        // Scene generation is now running in background — SSE will deliver tokens + completion
    } catch (e) {
        manualIsGeneratingScene = false;
        if (btn) { btn.disabled = false; btn.textContent = '▶ Generate Scene'; }
        if (statusEl) statusEl.classList.add('hidden');
        if (stopBtn) stopBtn.classList.add('hidden');
        showToast(e.message, 'error');
    }
}

async function stopManualScene() {
    if (!currentProject || !manualSessionActive || !manualIsGeneratingScene) return;

    try {
        await fetch(`/api/project/${currentProject}/generate/manual/cancel_scene`, {
            method: 'POST',
        });
        showToast('Stopping scene generation...', 'info');
        // onManualSceneDone will be called by SSE with status='cancelled'
    } catch (e) {
        showToast('Failed to stop scene', 'error');
    }
}

function onManualSceneDone(data) {
    manualIsGeneratingScene = false;
    
    const btn = document.getElementById('btnManualScene');
    const stopBtn = document.getElementById('btnManualStopScene');
    const finishBtn = document.getElementById('btnManualFinish');
    const statusEl = document.getElementById('manualSceneStatus');

    if (stopBtn) stopBtn.classList.add('hidden');

    if (data.status === 'cancelled') {
        if (btn) { btn.disabled = false; btn.textContent = '▶ Generate Scene'; }
        if (statusEl) statusEl.classList.add('hidden');
        document.getElementById('manualStreamArea').classList.add('hidden');
        if (manualScenesCompleted > 0 && finishBtn) finishBtn.classList.remove('hidden');
        showToast('Scene generation stopped.', 'info');
        return;
    }

    manualScenesCompleted = data.scene_number || (manualScenesCompleted + 1);

    // Add completed scene to the display
    const completedDiv = document.getElementById('manualCompletedScenes');
    const sceneHtml = `
        <div class="manual-scene-result" style="position:relative; margin-bottom:16px;padding:16px;background:var(--surface-2);border-radius:8px;border-left:3px solid var(--accent-primary)">
            <button class="btn btn-sm btn-danger" style="position:absolute;top:12px;right:12px;z-index:10;font-size:0.8rem;padding:4px 8px" onclick="deleteManualScene(${manualScenesCompleted - 1})" title="Delete this scene">🗑️</button>
            <div style="font-weight:600;margin-bottom:8px;color:var(--accent-primary);padding-right:40px">Scene ${manualScenesCompleted}</div>
            <div class="text-dim text-sm" style="margin-bottom:6px">Enhanced: ${escHtml(data.enhanced_summary || '')}</div>
            <div style="white-space:pre-wrap;line-height:1.7;font-family:'Lora',serif;font-size:0.92rem;max-height:300px;overflow-y:auto">${escHtml(data.scene_text || '')}</div>
            <div class="text-dim text-sm" style="margin-top:8px">${data.words || 0} words</div>
        </div>
    `;
    completedDiv.insertAdjacentHTML('beforeend', sceneHtml);
    completedDiv.scrollTop = completedDiv.scrollHeight;

    // Reset input for next scene
    document.getElementById('manualSceneBrief').value = '';
    document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen`;
    document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;

    // Show finish button after first scene
    if (finishBtn) finishBtn.classList.remove('hidden');
    document.getElementById('manualStreamArea').classList.add('hidden');

    if (btn) { btn.disabled = false; btn.textContent = '▶ Generate Scene'; }
    if (statusEl) statusEl.classList.add('hidden');

    showToast(`Scene ${manualScenesCompleted} complete! (${data.words || 0} words)`, 'success');
}

async function finishManualChapter() {
    if (!currentProject || !manualSessionActive || manualIsGeneratingScene) return;

    if (manualScenesCompleted < 1) {
        showToast('Generate at least one scene first', 'error');
        return;
    }

    if (!confirm(`Finalize this chapter with ${manualScenesCompleted} scene(s)?\nThis will save it permanently.`)) {
        return;
    }

    const btn = document.getElementById('btnManualFinish');
    const sceneBtn = document.getElementById('btnManualScene');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Finalizing...'; }
    if (sceneBtn) sceneBtn.disabled = true;

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/finish`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to finalize chapter');
        }
        // The SSE stream will send the 'done' event which calls resetManualSession
    } catch (e) {
        showToast(e.message, 'error');
        if (btn) { btn.disabled = false; btn.textContent = '✅ Finish Chapter'; }
        if (sceneBtn) sceneBtn.disabled = false;
    }
}

async function cancelManualSession() {
    if (!currentProject) return;
    if (!confirm('Cancel this manual session? Unsaved progress will be lost.')) return;

    // Cancel any active generation
    if (manualIsGeneratingScene) {
        try {
            await fetch(`/api/project/${currentProject}/generate/cancel`, { method: 'POST' });
        } catch (e) { /* ignore */ }
    }

    resetManualSession();
    showToast('Manual session cancelled', 'info');
}

function resetManualSession() {
    manualSessionActive = false;
    manualScenesCompleted = 0;
    manualChapterNum = 0;
    manualIsGeneratingScene = false;

    if (eventSource) { eventSource.close(); eventSource = null; }

    // Restore UI to start phase
    document.getElementById('manualStartCard').classList.remove('hidden');
    document.getElementById('manualSceneCard').classList.add('hidden');
    document.getElementById('manualStreamArea')?.classList.add('hidden');
    setStatus('online', 'Ready');
}

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

// Old batch manual generation (kept for API backwards compat)
async function startManualGeneration() {
    showToast('Please use the interactive scene-by-scene flow instead', 'info');
    switchView('manual');
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
            if (manualSessionActive && manualIsGeneratingScene) {
                // Stream tokens into the manual stream area
                const streamOut = document.getElementById('manualStreamOutput');
                if (streamOut && data.content) {
                    streamOut.appendChild(document.createTextNode(data.content));
                    const streamArea = document.getElementById('manualStreamArea');
                    if (streamArea) {
                        const scrollParent = streamArea.querySelector('.card') || streamArea;
                        scrollParent.scrollTop = scrollParent.scrollHeight;
                    }
                }
            } else {
                appendToken(data.scene || 'current', data.content || '');
            }
            if (data.provider) {
                updateStats({
                    provider: providerBadge(data.provider),
                });
            }
            break;

        case 'scene_written':
            totalWords += data.words;
            updateStats({
                words: totalWords,
                provider: providerBadge(data.provider),
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

        case 'manual_scene_done':
            onManualSceneDone(data);
            break;

        case 'done':
            if (manualSessionActive) {
                resetManualSession();
                showToast('Chapter finalized successfully!', 'success');
            }
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
    // Only close SSE if not in manual session (manual session manages its own SSE)
    if (!manualSessionActive && eventSource) { eventSource.close(); eventSource = null; }
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
        <div class="chapter-item" onclick="readChapter(${ch.number}, this)">
            <div><span class="ch-num">Chapter ${ch.number}</span></div>
            <div class="flex gap-2 items-center">
                <div class="ch-words">${ch.words} words</div>
                <button class="btn btn-sm btn-primary" onclick="event.stopPropagation();resumeChapter(${ch.number})" title="Edit this chapter (warning: deletes later chapters)">✎ Edit</button>
                <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();deleteChaptersFrom(${ch.number})" title="Delete this chapter and all later chapters">Delete+</button>
            </div>
        </div>
    `).join('');
}

async function readChapter(num, el = null) {
    // Highlight active
    document.querySelectorAll('.chapter-item').forEach(i => i.classList.remove('active'));
    if (el) el.classList.add('active');

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

async function resumeChapter(num) {
    if (!confirm(`Are you sure you want to edit Chapter ${num}? WARNING: This will permanently DELETE all chapters after Chapter ${num}.`)) {
        return;
    }

    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${num}/resume`, {
            method: 'POST'
        });
        const data = await res.json();
        
        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to resume chapter');
        }

        // Switch to manual tab
        switchView('manual');
        
        // Restore manual session state
        manualSessionActive = true;
        manualScenesCompleted = data.scenes_completed || data.completed_scenes.length;
        
        document.getElementById('manualStartCard').classList.add('hidden');
        document.getElementById('manualSceneCard').classList.remove('hidden');
        document.getElementById('manualSessionTitle').innerHTML = `<span class="icon">✍️</span> Editing Chapter ${num}: ${escHtml(data.chapter_title)}`;
        document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
        
        // Render completed scenes
        if (data.completed_scenes) {
            renderManualScenes(data.completed_scenes);
        }
        
        document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen next`;
        document.getElementById('manualSceneBrief').value = '';
        
        document.getElementById('btnManualFinish').classList.remove('hidden');
        document.getElementById('btnManualScene').disabled = false;
        document.getElementById('btnManualScene').textContent = '▶ Generate Scene';
        
        showToast(`Resumed Chapter ${num}`, 'success');
        
        // Reconnect SSE
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        eventSource = new EventSource(`/api/project/${currentProject}/generate/stream`);
        eventSource.onmessage = handleSSE;
        eventSource.onerror = () => {
            if (eventSource) { eventSource.close(); eventSource = null; }
        };
        
        loadProjects(); // refresh stats
        loadChapters(); // refresh chapter list
    } catch (e) {
        showToast(e.message, 'error');
    }
}

async function deleteChaptersFromUi() {
    const input = document.getElementById('deleteFromChapter');
    const from = parseInt(input?.value, 10);
    if (!from || from < 1) {
        showToast('Enter a valid chapter number (>= 1)', 'error');
        return;
    }
    await deleteChaptersFrom(from);
}

async function deleteChaptersFrom(fromChapter) {
    if (!currentProject) {
        showToast('Select a project first', 'error');
        return;
    }
    if (!confirm(`Delete chapter ${fromChapter} and all later chapters?\n\nThis will also rewind state, WIP, and vector memory.`)) {
        return;
    }

    const button = document.getElementById('btnDeleteChapters');
    if (button) {
        button.disabled = true;
        button.textContent = '⏳ Deleting...';
    }

    try {
        const res = await fetch(`/api/project/${currentProject}/chapters/delete`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ from_chapter: fromChapter }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            showToast(data.error || 'Delete failed', 'error');
            return;
        }

        if (Array.isArray(data.deleted_chapters) && data.deleted_chapters.length) {
            showToast(`Deleted chapters: ${data.deleted_chapters.join(', ')}`, 'success');
        } else {
            showToast(`No chapters found from ${fromChapter}`, 'info');
        }
        document.getElementById('readerContent').innerHTML = `<div class="empty-state">
            <span class="icon">📖</span>
            <h3>Select a chapter</h3>
            <p>Choose a chapter from the sidebar to read.</p>
        </div>`;
        await reloadCurrentProjectInfo();
        await loadChapters();
        await refreshState();
    } catch (e) {
        showToast('Failed to delete chapters', 'error');
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = '🗑️ Delete from';
        }
    }
}

async function reloadCurrentProjectInfo() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}`);
        const data = await res.json();
        if (data?.info?.title) {
            document.getElementById('genTitle').innerHTML =
                `<span class="icon">✍️</span> ${escHtml(data.info.title)}`;
        }
        document.getElementById('activeProject').textContent = currentProject;
    } catch (e) {
        // no-op; caller handles user feedback
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
    return String(md || '')
        .split(/\n{2,}/)
        .map(block => renderMarkdownBlock(block.trim()))
        .filter(Boolean)
        .join('');
}

function renderMarkdownBlock(block) {
    if (!block) return '';
    if (/^\* \* \*$/.test(block)) {
        return '<hr style="border:none;border-top:1px solid var(--border);margin:24px 0">';
    }

    const h3 = block.match(/^###\s+(.+)$/s);
    if (h3) return `<h3>${inlineMarkdown(h3[1])}</h3>`;

    const h2 = block.match(/^##\s+(.+)$/s);
    if (h2) return `<h2>${inlineMarkdown(h2[1])}</h2>`;

    const h1 = block.match(/^#\s+(.+)$/s);
    if (h1) return `<h1>${inlineMarkdown(h1[1])}</h1>`;

    return `<p>${inlineMarkdown(block).replace(/\n/g, '<br>')}</p>`;
}

function inlineMarkdown(text) {
    return escHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>');
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

// ─── Combine & Polish (Gemini) ───────────────────────────────────
let combineEventSource = null;
let isCombining = false;

async function startCombine() {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    if (isCombining) { showToast('Combine already in progress', 'info'); return; }

    isCombining = true;
    const btn = document.getElementById('btnCombine');
    const progress = document.getElementById('combineProgress');
    const analysisCard = document.getElementById('combineAnalysisCard');
    const downloadCard = document.getElementById('combineDownloadCard');

    btn.disabled = true;
    btn.innerHTML = '<span class="gemini-icon">⏳</span> Processing...';
    progress.classList.remove('hidden');
    analysisCard.classList.add('hidden');
    downloadCard.classList.add('hidden');
    document.getElementById('combineStatusText').textContent = 'Starting combine pipeline...';
    setStatus('working', 'Combining with Gemini...');

    try {
        const res = await fetch(`/api/project/${currentProject}/combine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.error || 'Failed to start combine');
        }

        // Connect SSE
        combineEventSource = new EventSource(`/api/project/${currentProject}/combine/stream`);
        combineEventSource.onmessage = handleCombineSSE;
        combineEventSource.onerror = () => {
            stopCombine();
            showToast('Connection lost during combine', 'error');
        };
    } catch (e) {
        stopCombine();
        showToast(e.message, 'error');
    }
}

function handleCombineSSE(event) {
    const msg = JSON.parse(event.data);
    const evt = msg.event || msg.type;
    const data = msg.data ?? msg.payload ?? {};

    switch (evt) {
        case 'combine_status':
            document.getElementById('combineStatusText').textContent =
                (typeof data === 'string') ? data : (data.step || 'Processing...');
            break;

        case 'combine_done': {
            const analysisCard = document.getElementById('combineAnalysisCard');
            const downloadCard = document.getElementById('combineDownloadCard');
            const analysisEl = document.getElementById('combineAnalysis');
            const statsEl = document.getElementById('combineStats');

            // Show analysis
            if (data.analysis) {
                analysisEl.innerHTML = markdownToHtml(data.analysis);
                analysisCard.classList.remove('hidden');
            }

            // Show download section
            statsEl.innerHTML = `
                <div class="combine-stats-grid">
                    <div class="combine-stat">
                        <span class="combine-stat-label">Original</span>
                        <span class="combine-stat-value">${(data.original_chars || 0).toLocaleString()} chars</span>
                    </div>
                    <div class="combine-stat">
                        <span class="combine-stat-label">Polished</span>
                        <span class="combine-stat-value">${(data.revised_chars || 0).toLocaleString()} chars</span>
                    </div>
                    <div class="combine-stat">
                        <span class="combine-stat-label">Model</span>
                        <span class="combine-stat-value">${escHtml(data.model || 'unknown')}</span>
                    </div>
                </div>
            `;
            downloadCard.classList.remove('hidden');

            stopCombine();
            showToast('Story combined and polished successfully!', 'success');
            break;
        }

        case 'combine_error':
            stopCombine();
            showToast(data.error || 'Combine failed', 'error');
            break;

        case 'heartbeat':
            break;
    }
}

function stopCombine() {
    isCombining = false;
    const btn = document.getElementById('btnCombine');
    const progress = document.getElementById('combineProgress');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span class="gemini-icon">🔮</span> Combine & Polish';
    }
    if (progress) progress.classList.add('hidden');
    setStatus('online', 'Ready');
    if (combineEventSource) { combineEventSource.close(); combineEventSource = null; }
}

function downloadCombined(type) {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    window.open(`/api/project/${currentProject}/combine/download/${type}`, '_blank');
    showToast(`Downloading ${type} file...`, 'info');
}

function toggleAnalysis() {
    const el = document.getElementById('combineAnalysis');
    const btn = document.getElementById('btnToggleAnalysis');
    if (el.style.display === 'none') {
        el.style.display = '';
        btn.textContent = '▼ Collapse';
    } else {
        el.style.display = 'none';
        btn.textContent = '▶ Expand';
    }
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
            select.innerHTML = '<option value="">No models available</option>';
            return;
        }
        select.innerHTML = data.models.map(m =>
            `<option value="${escHtml(m)}" ${m === data.active ? 'selected' : ''}>${escHtml(modelOptionLabel(m, data.groq_model))}</option>`
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
            if ((data.active || model) === GROQ_MODEL_OPTION) {
                showToast(`Switched to Groq API (${data.groq_model || 'default'})`, 'success');
            } else {
                showToast(`Switched to ${data.active || model}`, 'success');
            }
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
