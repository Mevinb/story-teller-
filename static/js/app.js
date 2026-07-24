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
const GEMINI_MODEL_OPTION = '__gemini_api__';
const OPENROUTER_MODEL_OPTION = '__openrouter_api__';

// Manual interactive session state
let manualSessionActive = false;
let manualScenesCompleted = 0;
let manualChapterNum = 0;
let manualIsGeneratingScene = false;

// Premise Generator State
let premiseSteps = [];
let premiseCharacters = {};

function isGroqProvider(provider) {
    return String(provider || '').toLowerCase().includes('groq');
}

function isGeminiProvider(provider) {
    return String(provider || '').toLowerCase().includes('gemini');
}

function isOpenRouterProvider(provider) {
    return String(provider || '').toLowerCase().includes('openrouter');
}

function providerBadge(provider) {
    if (isGeminiProvider(provider)) return '✨ Gemini';
    if (isOpenRouterProvider(provider)) return '🌐 OpenRouter';
    return isGroqProvider(provider) ? '☁️ Groq' : '💻 llama.cpp';
}

function modelOptionLabel(modelId, groqModelName, geminiModelName, openrouterModelName) {
    if (modelId === GROQ_MODEL_OPTION) {
        return `☁️ Groq API (${groqModelName || 'default'})`;
    }
    if (modelId === GEMINI_MODEL_OPTION) {
        return `✨ Gemini API (${geminiModelName || 'default'})`;
    }
    if (modelId === OPENROUTER_MODEL_OPTION) {
        return `🌐 OpenRouter API (${openrouterModelName || 'default'})`;
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
    if (viewName === 'combine' && currentProject) loadCombineVersions();
    if (viewName === 'edit' && currentProject) loadProjectForEdit();
    if (viewName === 'premise' && currentProject) loadProjectPremise();
    if (viewName === 'dashboard') loadProjects();
    if (viewName === 'settings') loadSettings();
    if (viewName === 'vision') checkVisionStatus();
    if (viewName === 'manual' && currentProject) {
        // Fetch the current manual session status to restore UI state
        fetch(`/api/project/${currentProject}/generate/manual/status`)
            .then(res => res.json())
            .then(data => {
                if (data.active) {
                    manualSessionActive = true;
                    manualScenesCompleted = data.scenes_completed;
                    manualIsGeneratingScene = Boolean(data.is_generating);
                    document.getElementById('manualStartCard')?.classList.add('hidden');
                    document.getElementById('manualSceneCard')?.classList.remove('hidden');
                    document.getElementById('manualSessionTitle').innerHTML = `<span class="icon">✍️</span> Chapter ${data.chapter_num}: ${escHtml(data.chapter_title)}`;
                    document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
                    document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen`;
                    document.getElementById('btnManualScene').disabled = manualIsGeneratingScene;
                    document.getElementById('btnManualScene').textContent = manualIsGeneratingScene ? '⏳ Generating...' : '▶ Generate Scene';
                    document.getElementById('btnManualStopScene')?.classList.toggle('hidden', !manualIsGeneratingScene);
                    if (data.completed_scenes) {
                        renderManualScenes(data.completed_scenes);
                    }
                    if (manualScenesCompleted > 0 && !manualIsGeneratingScene) {
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

// Store raw scene text for edit mode
let manualSceneTexts = new Map();

function renderManualScenes(scenes) {
    const completedDiv = document.getElementById('manualCompletedScenes');
    completedDiv.innerHTML = '';
    manualSceneTexts.clear();
    if (scenes && scenes.length > 0) {
        scenes.forEach((scene, idx) => {
            // Handle both simple strings (from DB fallback) and dicts (from generate_manual_scene)
            const sceneText = typeof scene === 'string' ? scene : scene.text;
            const enhancedSummary = typeof scene === 'string' ? '' : (scene.enhanced_summary || '');
            const wordCount = (sceneText || '').split(/\s+/).filter(Boolean).length;
            
            // Store raw text in Map for edit mode
            manualSceneTexts.set(idx, sceneText || '');
            
            const sceneHtml = `
                <div class="manual-scene-result" style="position:relative; margin-bottom:16px;padding:16px;background:var(--surface-2);border-radius:8px;border-left:3px solid var(--accent-primary)">
                    <div style="position:absolute;top:12px;right:12px;z-index:10;display:flex;gap:4px">
                        <button class="btn btn-sm" style="font-size:0.8rem;padding:4px 8px;background:var(--accent-secondary);color:#fff" onclick="editManualScene(${idx})" title="Edit this scene's text">✏️</button>
                        <button class="btn btn-sm btn-danger" style="font-size:0.8rem;padding:4px 8px" onclick="deleteManualScene(${idx})" title="Delete this scene">🗑️</button>
                    </div>
                    <div style="font-weight:600;margin-bottom:8px;color:var(--accent-primary);padding-right:80px">Scene ${idx + 1}</div>
                    ${enhancedSummary ? `<div class="text-dim text-sm" style="margin-bottom:6px">Enhanced: ${escHtml(enhancedSummary)}</div>` : ''}
                    <div style="white-space:pre-wrap;line-height:1.7;font-family:'Lora',serif;font-size:0.92rem;max-height:300px;overflow-y:auto">${escHtml(sceneText || '')}</div>
                    <div class="text-dim text-sm" style="margin-top:8px">${wordCount} words</div>
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

// ─── Typed Scene Panel (Manual Mode) ─────────────────────────────
function toggleTypedScenePanel() {
    const panel = document.getElementById('manualTypedScenePanel');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        document.getElementById('manualTypedSceneText').value = '';
        document.getElementById('typedSceneWordCount').textContent = '0 words';
        document.getElementById('manualTypedSceneText').focus();
    } else {
        panel.classList.add('hidden');
    }
}

// Live word counter for typed scene textarea
document.addEventListener('DOMContentLoaded', () => {
    const ta = document.getElementById('manualTypedSceneText');
    if (ta) {
        ta.addEventListener('input', () => {
            const words = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
            document.getElementById('typedSceneWordCount').textContent = `${words} word${words !== 1 ? 's' : ''}`;
        });
    }
});

async function addTypedScene() {
    if (!currentProject || !manualSessionActive) {
        showToast('No active manual session', 'error');
        return;
    }

    const text = document.getElementById('manualTypedSceneText').value.trim();
    if (!text) {
        showToast('Type some scene text first', 'error');
        return;
    }

    const btn = document.getElementById('btnAddTypedScene');
    btn.disabled = true;
    btn.textContent = '⏳ Adding...';

    try {
        // Use the scene API to add typed text as a "generated" scene via a special brief
        const res = await fetch(`/api/project/${currentProject}/generate/manual/scene/typed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to add typed scene');

        manualScenesCompleted = data.scenes_completed;
        document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
        document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen`;
        renderManualScenes(data.completed_scenes);

        document.getElementById('btnManualFinish').classList.remove('hidden');
        toggleTypedScenePanel();
        showToast(`Typed scene added (${data.words} words)`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '✅ Add Scene';
    }
}

// ─── Edit Scene Inline (Manual Mode) ─────────────────────────────
function editManualScene(index) {
    const container = document.getElementById('manualCompletedScenes');
    const sceneEl = container.children[index];
    if (!sceneEl) return;

    // Get the raw text from our Map
    const rawText = manualSceneTexts.get(index) || '';

    // Replace the text display with a textarea
    const oldHtml = sceneEl.innerHTML;
    sceneEl.setAttribute('data-old-html', oldHtml);
    sceneEl.innerHTML = `
        <div style="font-weight:600;margin-bottom:8px;color:var(--accent-primary)">Scene ${index + 1} — Editing</div>
        <textarea class="form-textarea" id="editSceneTextarea_${index}" rows="12"
            style="font-family:'Lora',serif;font-size:0.92rem;line-height:1.7;width:100%;white-space:pre-wrap">${escHtml(rawText)}</textarea>
        <div class="flex gap-2 items-center" style="margin-top:10px">
            <button class="btn btn-primary" onclick="saveManualSceneEdit(${index})" style="font-size:0.85rem;padding:6px 14px">💾 Save</button>
            <button class="btn" onclick="cancelManualSceneEdit(${index})" style="font-size:0.85rem;padding:6px 14px">✕ Cancel</button>
            <span class="text-dim text-sm" id="editSceneWordCount_${index}"></span>
        </div>
    `;

    // Word counter for edit textarea
    const ta = document.getElementById(`editSceneTextarea_${index}`);
    const wc = document.getElementById(`editSceneWordCount_${index}`);
    const updateWc = () => {
        const w = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
        wc.textContent = `${w} words`;
    };
    updateWc();
    ta.addEventListener('input', updateWc);
    ta.focus();
}

function cancelManualSceneEdit(index) {
    const container = document.getElementById('manualCompletedScenes');
    const sceneEl = container.children[index];
    if (!sceneEl) return;
    const oldHtml = sceneEl.getAttribute('data-old-html');
    if (oldHtml) {
        sceneEl.innerHTML = oldHtml;
    }
}

async function saveManualSceneEdit(index) {
    const ta = document.getElementById(`editSceneTextarea_${index}`);
    if (!ta) return;
    const newText = ta.value.trim();
    if (!newText) {
        showToast('Scene text cannot be empty', 'error');
        return;
    }

    try {
        const res = await fetch(`/api/project/${currentProject}/generate/manual/scene/${index}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: newText }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save scene edit');

        manualScenesCompleted = data.scenes_completed;
        document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
        renderManualScenes(data.completed_scenes);
        showToast(`Scene ${index + 1} updated (${data.words} words)`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    }
}

function showProjectTabs() {
    document.getElementById('tabGenerate').classList.remove('hidden');
    document.getElementById('tabPremise').classList.remove('hidden');
    document.getElementById('tabManual').classList.remove('hidden');
    document.getElementById('tabReader').classList.remove('hidden');
    document.getElementById('tabLogs').classList.remove('hidden');
    document.getElementById('tabState').classList.remove('hidden');
    document.getElementById('tabCombine').classList.remove('hidden');
    document.getElementById('tabEdit').classList.remove('hidden');
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
            
        // Restore Combine view state if data exists
        if (data.combine_data) {
            populateCombineUI(data.combine_data);
        } else {
            // Reset combine view
            document.getElementById('combineAnalysisCard').classList.add('hidden');
            document.getElementById('combinePolishedCard').classList.add('hidden');
            document.getElementById('combineDownloadCard').classList.add('hidden');
            document.getElementById('btnDeleteCombine').classList.add('hidden');
            document.getElementById('btnCombine').disabled = false;
            document.getElementById('btnCombine').innerHTML = '<span class="gemini-icon">🔮</span> Combine & Polish';
        }

        loadCombineVersions();

        switchView('generate');
    } catch (e) {
        showToast('Failed to load project', 'error');
    }
}

// ─── Edit Project ────────────────────────────────────────────────
let editCharacters = [];

async function loadProjectForEdit() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}`);
        const data = await res.json();
        
        const metadata = data.state.metadata || {};
        document.getElementById('editTitle').value = metadata.title || '';
        document.getElementById('editGenre').value = metadata.genre || '';
        document.getElementById('editPremise').value = metadata.premise || '';
        document.getElementById('editSetting').value = metadata.setting || '';
        document.getElementById('editThemes').value = (metadata.themes || []).join(', ');

        const charsObj = data.state.characters || {};
        editCharacters = Object.keys(charsObj).map(name => ({
            name: name,
            description: charsObj[name].description || '',
            traits: charsObj[name].traits || [],
            originalName: name
        }));
        renderEditCharacterList();
    } catch (e) {
        showToast('Failed to load project for editing', 'error');
    }
}

function renderEditCharacterList() {
    const list = document.getElementById('editCharacterList');
    list.innerHTML = editCharacters.map((c, i) => `
        <div class="character-entry">
            <span class="char-name">${escHtml(c.name)}</span>
            <span>${escHtml(c.description)}</span>
            <span class="text-dim">${c.traits.join(', ')}</span>
            <span class="char-remove" onclick="removeEditCharacter(${i})">✕</span>
        </div>
    `).join('');
}

function addEditCharacter() {
    const name = document.getElementById('editCharName').value.trim();
    const desc = document.getElementById('editCharDesc').value.trim();
    const traits = document.getElementById('editCharTraits').value.trim();

    if (!name) { showToast('Enter a character name', 'error'); return; }

    const existingIndex = editCharacters.findIndex(c => c.name.toLowerCase() === name.toLowerCase());
    if (existingIndex >= 0) {
        editCharacters[existingIndex].description = desc;
        editCharacters[existingIndex].traits = traits ? traits.split(',').map(t => t.trim()) : [];
    } else {
        editCharacters.push({
            name,
            description: desc,
            traits: traits ? traits.split(',').map(t => t.trim()) : [],
            isNew: true
        });
    }

    renderEditCharacterList();
    document.getElementById('editCharName').value = '';
    document.getElementById('editCharDesc').value = '';
    document.getElementById('editCharTraits').value = '';
    document.getElementById('editCharName').focus();
}

function removeEditCharacter(index) {
    editCharacters.splice(index, 1);
    renderEditCharacterList();
}

async function saveProjectEdits() {
    if (!currentProject) return;

    const title = document.getElementById('editTitle').value.trim();
    const genre = document.getElementById('editGenre').value.trim();
    const premise = document.getElementById('editPremise').value.trim();
    const setting = document.getElementById('editSetting').value.trim();
    const themesStr = document.getElementById('editThemes').value.trim();

    if (!title) { showToast('Title cannot be empty', 'error'); return; }

    const btn = document.getElementById('btnSaveEdits');
    btn.disabled = true;
    btn.textContent = '⏳ Saving...';

    const metadataUpdate = {
        title, genre, premise, setting,
        themes: themesStr ? themesStr.split(',').map(t => t.trim()) : []
    };

    const charactersUpdate = {};
    editCharacters.forEach(c => {
        charactersUpdate[c.name] = {
            description: c.description,
            traits: c.traits
        };
    });

    try {
        const res = await fetch(`/api/project/${currentProject}/state`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                metadata: metadataUpdate,
                characters: charactersUpdate
            }),
        });

        const data = await res.json();
        if (data.status === 'ok') {
            showToast('Story details saved successfully', 'success');
            // If title changed, we might want to update the sidebar/header
            document.getElementById('genTitle').innerHTML = `<span class="icon">✍️</span> ${escHtml(title)}`;
        } else {
            showToast(data.error || 'Failed to save edits', 'error');
        }
    } catch (e) {
        showToast('Failed to save edits', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '💾 Save Changes';
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

    if (data.status === 'error') {
        manualScenesCompleted = data.scenes_completed ?? manualScenesCompleted;
        document.getElementById('manualSceneCounter').textContent = `${manualScenesCompleted} scene${manualScenesCompleted !== 1 ? 's' : ''} completed`;
        document.getElementById('manualSceneLabel').textContent = `Scene ${manualScenesCompleted + 1} — Describe what should happen`;
        if (btn) { btn.disabled = false; btn.textContent = '▶ Generate Scene'; }
        if (statusEl) {
            statusEl.classList.remove('hidden');
            statusEl.textContent = 'Scene failed. Completed scenes are still saved.';
        }
        if (manualScenesCompleted > 0 && finishBtn) finishBtn.classList.remove('hidden');
        showToast(data.error || 'Scene generation failed', 'error');
        addLogEntry({
            timestamp: new Date().toLocaleTimeString(),
            level: 'error',
            message: data.error || 'Scene generation failed',
        });
        loadChapters();
        return;
    }

    manualScenesCompleted = data.scene_number || (manualScenesCompleted + 1);

    // Add completed scene to the display
    const completedDiv = document.getElementById('manualCompletedScenes');
    const sceneIdx = manualScenesCompleted - 1;
    manualSceneTexts.set(sceneIdx, data.scene_text || '');
    const sceneHtml = `
        <div class="manual-scene-result" style="position:relative; margin-bottom:16px;padding:16px;background:var(--surface-2);border-radius:8px;border-left:3px solid var(--accent-primary)">
            <div style="position:absolute;top:12px;right:12px;z-index:10;display:flex;gap:4px">
                <button class="btn btn-sm" style="font-size:0.8rem;padding:4px 8px;background:var(--accent-secondary);color:#fff" onclick="editManualScene(${sceneIdx})" title="Edit this scene's text">✏️</button>
                <button class="btn btn-sm btn-danger" style="font-size:0.8rem;padding:4px 8px" onclick="deleteManualScene(${sceneIdx})" title="Delete this scene">🗑️</button>
            </div>
            <div style="font-weight:600;margin-bottom:8px;color:var(--accent-primary);padding-right:80px">Scene ${manualScenesCompleted}</div>
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
    manualSceneTexts.clear();

    if (eventSource) { eventSource.close(); eventSource = null; }

    // Restore UI to start phase
    document.getElementById('manualStartCard').classList.remove('hidden');
    document.getElementById('manualSceneCard').classList.add('hidden');
    document.getElementById('manualStreamArea')?.classList.add('hidden');
    document.getElementById('manualTypedScenePanel')?.classList.add('hidden');
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
            loadChapters(); // Refresh list to show WIP chapter
            break;

        case 'agent_active':
            updatePipelineStage(data.agent);
            updateStats({ status: `${data.agent}: ${data.step}` });
            if (manualSessionActive && manualIsGeneratingScene) {
                const statusEl = document.getElementById('manualSceneStatus');
                if (statusEl && !statusEl.classList.contains('hidden')) {
                    const phaseMap = {
                        'Scene Writer':      '✍️ Writing scene...',
                        'Consistency Engine':'🔍 Reviewing consistency...',
                        'Editor':            '✂️ Polishing scene...',
                        'Retriever':         '📚 Retrieving context...',
                    };
                    statusEl.textContent = phaseMap[data.agent] || `⚙️ ${data.agent}...`;
                }
            }
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
            if (manualSessionActive && manualIsGeneratingScene) {
                const statusEl = document.getElementById('manualSceneStatus');
                if (statusEl && !statusEl.classList.contains('hidden')) {
                    statusEl.textContent = '✍️ Writing scene...';
                }
            }
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
            loadChapters(); // Update word count/status in reader list
            break;

        case 'chapter_complete':
            completedChapters += 1;
            const planDisplay = plannedChapters < 0 ? 'Entire Story' : plannedChapters;
            showToast(
                `Chapter ${data.chapter_number} complete! ${completedChapters}/${planDisplay}`,
                'success',
            );
            updateStats({ status: `Completed chapter ${completedChapters}/${planDisplay}` });
            loadChapters(); // Update status from 'writing' to 'completed'
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
            // Mirror live status into manual scene status label during generation
            if (manualSessionActive && manualIsGeneratingScene) {
                const statusEl = document.getElementById('manualSceneStatus');
                if (statusEl && !statusEl.classList.contains('hidden')) {
                    statusEl.textContent = typeof data === 'string' ? data : JSON.stringify(data);
                }
            }
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
            if (data.status === 'cancelled') {
                showToast('Generation stopped.', 'info');
            }
            if (manualSessionActive) {
                resetManualSession();
                if (data.status !== 'cancelled') {
                    showToast('Chapter finalized successfully!', 'success');
                }
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
    list.innerHTML = chapters.map(ch => {
        const isWriting = ch.status === 'writing';
        const statusBadge = isWriting ? `<span class="badge badge-working" style="font-size:0.7rem;margin-left:8px">Writing...</span>` : '';
        const progressInfo = isWriting ? `<div class="text-xs text-dim">${ch.scenes_completed}/${ch.scenes_total} scenes</div>` : '';
        
        return `
            <div class="chapter-item ${isWriting ? 'wip' : ''}" onclick="readChapter(${ch.number}, this)">
                <div class="flex flex-col">
                    <div class="flex items-center">
                        <span class="ch-num">Chapter ${ch.number}</span>
                        ${statusBadge}
                    </div>
                    ${progressInfo}
                </div>
                <div class="flex gap-2 items-center">
                    <div class="ch-words">${ch.words} words</div>
                    ${!isWriting ? `
                        <button class="btn btn-sm btn-primary" onclick="event.stopPropagation();resumeChapter(${ch.number})" title="Edit this chapter (warning: deletes later chapters)">✎ Edit</button>
                        <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();deleteChaptersFrom(${ch.number})" title="Delete this chapter and all later chapters">Delete+</button>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
}

let readerCurrentChapter = null;
let readerRawContent = '';
let readerEditMode = false;

async function readChapter(num, el = null) {
    // Highlight active
    document.querySelectorAll('.chapter-item').forEach(i => i.classList.remove('active'));
    if (el) el.classList.add('active');

    // Cancel edit mode if active
    if (readerEditMode) cancelReaderEdit();

    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${num}`);
        const data = await res.json();
        if (data.content) {
            readerCurrentChapter = num;
            readerRawContent = data.content;
            document.getElementById('readerContent').innerHTML = markdownToHtml(data.content);

            // Show card header with chapter info
            const header = document.getElementById('readerCardHeader');
            header.style.display = '';
            document.getElementById('readerChapterTitle').innerHTML = `<span class="icon">📖</span> Chapter ${num}`;
            document.getElementById('btnReaderEdit').classList.remove('hidden');
        }
    } catch (e) {
        showToast('Failed to load chapter', 'error');
    }
}

function toggleReaderEdit() {
    if (readerCurrentChapter === null) {
        showToast('Open a chapter first', 'error');
        return;
    }
    readerEditMode = true;

    // Hide rendered content, show editor textarea
    document.getElementById('readerContent').classList.add('hidden');
    const editor = document.getElementById('readerEditor');
    editor.classList.remove('hidden');
    editor.value = readerRawContent;
    editor.focus();

    // Toggle buttons
    document.getElementById('btnReaderEdit').classList.add('hidden');
    document.getElementById('btnReaderSave').classList.remove('hidden');
    document.getElementById('btnReaderCancel').classList.remove('hidden');
}

function cancelReaderEdit() {
    readerEditMode = false;
    document.getElementById('readerContent').classList.remove('hidden');
    document.getElementById('readerEditor').classList.add('hidden');
    document.getElementById('btnReaderEdit').classList.remove('hidden');
    document.getElementById('btnReaderSave').classList.add('hidden');
    document.getElementById('btnReaderCancel').classList.add('hidden');
}

async function saveReaderEdit() {
    if (readerCurrentChapter === null) return;

    const editor = document.getElementById('readerEditor');
    const content = editor.value.trim();
    if (!content) {
        showToast('Chapter content cannot be empty', 'error');
        return;
    }

    const btn = document.getElementById('btnReaderSave');
    btn.disabled = true;
    btn.textContent = '⏳ Saving...';

    try {
        const res = await fetch(`/api/project/${currentProject}/chapter/${readerCurrentChapter}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || 'Failed to save chapter');
        }

        // Update stored raw content and re-render
        readerRawContent = content;
        document.getElementById('readerContent').innerHTML = markdownToHtml(content);
        cancelReaderEdit();
        loadChapters(); // Refresh word counts in sidebar
        showToast(`Chapter ${readerCurrentChapter} saved (${data.words} words)`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '💾 Save';
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
    if (!md) return '';
    
    // Normalize newlines
    const lines = String(md).replace(/\r\n/g, '\n').split('\n');
    let html = [];
    let inList = false;
    let listType = null; // 'ul' or 'ol'
    let currentParagraph = [];

    function closeList() {
        if (inList) {
            html.push(`</${listType}>`);
            inList = false;
            listType = null;
        }
    }

    function closeParagraph() {
        if (currentParagraph.length > 0) {
            html.push(`<p>${inlineMarkdown(currentParagraph.join('<br>'))}</p>`);
            currentParagraph = [];
        }
    }

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        // Handle empty line (paragraph/list break)
        if (!trimmed) {
            closeList();
            closeParagraph();
            continue;
        }

        // Handle scene separator / horizontal rule
        if (trimmed === '* * *' || trimmed === '---' || trimmed === '***') {
            closeList();
            closeParagraph();
            html.push('<hr style="border:none;border-top:1px solid var(--border);margin:24px 0">');
            continue;
        }

        // Handle headings
        const hHeading = trimmed.match(/^(#{1,6})\s+(.+)$/);
        if (hHeading) {
            closeList();
            closeParagraph();
            const level = hHeading[1].length;
            html.push(`<h${level}>${inlineMarkdown(hHeading[2])}</h${level}>`);
            continue;
        }

        // Handle bullet lists (lines starting with -, *, or + followed by space)
        const bulletMatch = line.match(/^(\s*)([-*+])\s+(.+)$/);
        if (bulletMatch) {
            closeParagraph();
            if (!inList || listType !== 'ul') {
                closeList();
                html.push('<ul class="markdown-list">');
                inList = true;
                listType = 'ul';
            }
            html.push(`<li>${inlineMarkdown(bulletMatch[3])}</li>`);
            continue;
        }

        // Handle numbered lists (lines starting with digits followed by dot and space)
        const numberMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);
        if (numberMatch) {
            closeParagraph();
            if (!inList || listType !== 'ol') {
                closeList();
                html.push('<ol class="markdown-list">');
                inList = true;
                listType = 'ol';
            }
            html.push(`<li>${inlineMarkdown(numberMatch[3])}</li>`);
            continue;
        }

        // It's a standard text line, append to paragraph
        closeList();
        currentParagraph.push(trimmed);
    }

    closeList();
    closeParagraph();

    return html.join('\n');
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
        const modelSelect = document.getElementById('combineModelSelect');
        const selectedModel = modelSelect ? modelSelect.value : null;

        const res = await fetch(`/api/project/${currentProject}/combine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: selectedModel })
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
            stopCombine();
            showToast('Story combined and polished successfully!', 'success');
            loadCombineVersions();
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
    const btnCancel = document.getElementById('btnCancelCombine');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span class="gemini-icon">🔮</span> Combine & Polish';
    }
    if (btnCancel) {
        btnCancel.disabled = false;
        btnCancel.textContent = '🛑 Stop';
    }
    if (progress) progress.classList.add('hidden');
    setStatus('online', 'Ready');
    if (combineEventSource) { combineEventSource.close(); combineEventSource = null; }
}

async function cancelCombine() {
    if (!currentProject) return;
    const btnCancel = document.getElementById('btnCancelCombine');
    if (btnCancel) {
        btnCancel.disabled = true;
        btnCancel.textContent = '⏳ Stopping...';
    }
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/cancel`, {
            method: 'POST'
        });
        if (res.ok) {
            showToast('Cancellation requested', 'info');
        } else {
            showToast('Failed to request cancellation', 'error');
            if (btnCancel) {
                btnCancel.disabled = false;
                btnCancel.textContent = '🛑 Stop';
            }
        }
    } catch (e) {
        showToast('Error requesting cancellation', 'error');
        if (btnCancel) {
            btnCancel.disabled = false;
            btnCancel.textContent = '🛑 Stop';
        }
    }
}

function downloadCombined(type) {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    const select = document.getElementById('combineVersionSelect');
    const suffix = select ? select.value : 'latest';
    window.open(`/api/project/${currentProject}/combine/download/${type}?suffix=${suffix}`, '_blank');
    showToast(`Downloading ${type} file...`, 'info');
}

// ─── Polished Story History Management ────────────────────────────
let combineVersionsList = [];

async function loadCombineVersions(selectSuffix = null) {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/versions`);
        const data = await res.json();
        
        const select = document.getElementById('combineVersionSelect');
        if (!select) return;
        
        combineVersionsList = data.versions || [];
        
        if (combineVersionsList.length === 0) {
            select.innerHTML = '<option value="">No versions available</option>';
            document.getElementById('combineAnalysisCard').classList.add('hidden');
            document.getElementById('combinePolishedCard').classList.add('hidden');
            document.getElementById('combineDownloadCard').classList.add('hidden');
            document.getElementById('btnDeleteCombine').classList.add('hidden');
            const btn = document.getElementById('btnCombine');
            if (btn) btn.innerHTML = '<span class="gemini-icon">🔮</span> Combine & Polish';
            return;
        }
        
        select.innerHTML = combineVersionsList.map(v => 
            `<option value="${v.suffix}">${escHtml(v.label)} (${(v.revised_chars/1024).toFixed(1)} KB)</option>`
        ).join('');
        
        let targetSuffix = selectSuffix;
        if (!targetSuffix || !combineVersionsList.some(v => v.suffix === targetSuffix)) {
            targetSuffix = combineVersionsList[combineVersionsList.length - 1].suffix;
        }
        
        select.value = targetSuffix;
        fetchCombineVersion(targetSuffix);
    } catch (e) {
        showToast('Failed to load polished story history', 'error');
    }
}

async function changeCombineVersion() {
    const select = document.getElementById('combineVersionSelect');
    if (!select || !select.value) return;
    fetchCombineVersion(select.value);
}

async function fetchCombineVersion(suffix) {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/version/${suffix}`);
        const data = await res.json();
        if (res.ok) {
            populateCombineUI(data);
        } else {
            showToast(data.error || 'Failed to fetch version', 'error');
        }
    } catch (e) {
        showToast('Failed to fetch version details', 'error');
    }
}

async function renameSelectedVersion() {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    const select = document.getElementById('combineVersionSelect');
    if (!select || !select.value) { showToast('No version selected', 'error'); return; }
    
    const suffix = select.value;
    const currentLabel = select.options[select.selectedIndex].text.split(' (')[0];
    
    const newName = prompt(`Enter a new label for this polished version:`, currentLabel);
    if (!newName || newName.trim() === '') return;
    
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ suffix: suffix, new_name: newName.trim() })
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
            showToast('Polished version renamed successfully!', 'success');
            loadCombineVersions(data.new_suffix);
        } else {
            showToast(data.error || 'Rename failed', 'error');
        }
    } catch (e) {
        showToast('Failed to rename version', 'error');
    }
}

async function deleteSelectedVersion() {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    const select = document.getElementById('combineVersionSelect');
    if (!select || !select.value) { showToast('No version selected', 'error'); return; }
    
    const suffix = select.value;
    if (!confirm('Are you sure you want to delete this specific polished version and its analysis? This cannot be undone.')) return;
    
    try {
        const res = await fetch(`/api/project/${currentProject}/combine/delete_version`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ suffix: suffix })
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
            showToast('Polished version deleted.', 'success');
            loadCombineVersions();
        } else {
            showToast(data.error || 'Delete failed', 'error');
        }
    } catch (e) {
        showToast('Failed to delete version', 'error');
    }
}

function populateCombineUI(data) {
    const analysisCard = document.getElementById('combineAnalysisCard');
    const polishedCard = document.getElementById('combinePolishedCard');
    const downloadCard = document.getElementById('combineDownloadCard');
    const analysisEl = document.getElementById('combineAnalysis');
    const polishedEl = document.getElementById('combinePolished');
    const statsEl = document.getElementById('combineStats');

    if (data.analysis) {
        analysisEl.innerHTML = markdownToHtml(data.analysis);
        analysisCard.classList.remove('hidden');
    } else {
        analysisCard.classList.add('hidden');
    }

    if (data.revised) {
        polishedEl.innerHTML = markdownToHtml(data.revised);
        polishedCard.classList.remove('hidden');
    } else {
        polishedCard.classList.add('hidden');
    }

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
    
    const btn = document.getElementById('btnCombine');
    if (btn) {
        btn.innerHTML = '<span class="gemini-icon">🔮</span> Re-Combine & Polish';
    }
    const btnDel = document.getElementById('btnDeleteCombine');
    if (btnDel) {
        btnDel.classList.remove('hidden');
    }
}

function toggleCombineSection(elId, btnId) {
    const el = document.getElementById(elId);
    const btn = document.getElementById(btnId);
    if (!el || !btn) return;
    
    if (el.style.display === 'none') {
        el.style.display = '';
        btn.textContent = '▼ Collapse';
    } else {
        el.style.display = 'none';
        btn.textContent = '▶ Expand';
    }
}

async function deleteCombineData() {
    if (!currentProject) { showToast('Select a project first', 'error'); return; }
    if (!confirm('Are you sure you want to delete ALL polished versions and analyses? This cannot be undone.')) return;

    try {
        const res = await fetch(`/api/project/${currentProject}/combine/delete`, { method: 'POST' });
        const data = await res.json();
        
        if (res.ok && data.status === 'ok') {
            showToast('All polished versions deleted.', 'success');
            loadCombineVersions();
        } else {
            showToast(data.error || 'Failed to delete polished versions', 'error');
        }
    } catch (e) {
        showToast('Failed to delete polished versions', 'error');
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

// ─── Settings / Model Management ─────────────────────────────────
function settingsFieldValue(el) {
    if (!el) return '';
    if (el.type === 'checkbox') return el.checked;
    return el.value;
}

function setSettingsField(key, value) {
    const el = document.querySelector(`[data-setting="${key}"]`);
    if (!el) return;
    if (el.type === 'checkbox') {
        el.checked = Boolean(value);
    } else {
        el.value = value ?? '';
    }
}

function currentBackendMode() {
    return document.querySelector('input[name="backendMode"]:checked')?.value || 'local';
}

function renderSettingsSummary(data) {
    const summary = document.getElementById('settingsSummary');
    if (!summary) return;
    const settings = data.settings || {};
    const mode = data.backend_mode || settings.BACKEND_MODE || 'local';
    const active = data.active_model || settings.ACTIVE_MODEL || settings.LLAMA_MODEL_PATH || 'none';
    const localCount = (data.local_models || []).length;
    summary.innerHTML = `
        <div class="settings-pill"><strong>Mode</strong><span>${escHtml(mode)}</span></div>
        <div class="settings-pill"><strong>Active</strong><span>${escHtml(active)}</span></div>
        <div class="settings-pill"><strong>Local models</strong><span>${localCount}</span></div>
        <div class="settings-pill"><strong>.env</strong><span>${escHtml(data.env_path || '.env')}</span></div>
    `;
}

function populateSettings(data) {
    const settings = data.settings || {};
    Object.entries(settings).forEach(([key, value]) => setSettingsField(key, value));

    const mode = settings.BACKEND_MODE || data.backend_mode || 'local';
    const modeInput = document.querySelector(`input[name="backendMode"][value="${mode}"]`);
    if (modeInput) modeInput.checked = true;

    const activeModel = document.getElementById('settingsActiveModel');
    if (activeModel) {
        const models = data.local_models || [];
        const active = settings.ACTIVE_MODEL || data.active_model || '';
        const options = models.map(model =>
            `<option value="${escHtml(model)}" ${model === active ? 'selected' : ''}>${escHtml(model)}</option>`
        );
        if (active && active !== GROQ_MODEL_OPTION && active !== OPENROUTER_MODEL_OPTION && !models.includes(active)) {
            options.unshift(`<option value="${escHtml(active)}" selected>${escHtml(active)}</option>`);
        }
        activeModel.innerHTML = options.length
            ? options.join('')
            : '<option value="">No GGUF models found</option>';
    }

    renderSettingsSummary(data);
}

async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to load settings');
        populateSettings(data);
    } catch (e) {
        showToast(e.message || 'Failed to load settings', 'error');
    }
}

async function saveSettings() {
    const btn = document.getElementById('btnSaveSettings');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Saving...';
    }

    const settings = { BACKEND_MODE: currentBackendMode() };
    document.querySelectorAll('[data-setting]').forEach(el => {
        settings[el.dataset.setting] = settingsFieldValue(el);
    });

    const activeModel = document.getElementById('settingsActiveModel')?.value || '';
    if (activeModel) settings.ACTIVE_MODEL = activeModel;

    try {
        const res = await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ settings }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save settings');
        populateSettings(data);
        loadModels();  // refresh header model selector with new backend
        showToast('Settings saved. New generations will use them.', 'success');
    } catch (e) {
        showToast(e.message || 'Failed to save settings', 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = '💾 Save Settings';
        }
    }
}

async function loadModels() {
    const select = document.getElementById('modelSelect');
    if (!select) return;
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        if (!data.models.length) {
            select.innerHTML = '<option value="">No models available</option>';
            return;
        }
        select.innerHTML = data.models.map(m =>
            `<option value="${escHtml(m)}" ${m === data.active ? 'selected' : ''}>${escHtml(modelOptionLabel(m, data.groq_model, data.gemini_model, data.openrouter_model))}</option>`
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
            } else if ((data.active || model) === GEMINI_MODEL_OPTION) {
                showToast(`Switched to Gemini API (${data.gemini_model || 'default'})`, 'success');
            } else if ((data.active || model) === OPENROUTER_MODEL_OPTION) {
                showToast(`Switched to OpenRouter API (${data.openrouter_model || 'default'})`, 'success');
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

// ─── Polished Story Reader Toolbar ───────────────────────────────
let readerFontSize = 17;
const readerThemes = ['obsidian', 'parchment', 'midnight'];
let readerThemeIndex = 0;
let readerIsSerif = false;
let readerIsFullscreen = false;

function adjustReaderFontSize(delta) {
    const el = readerIsFullscreen
        ? document.getElementById('fullscreenReaderContent')
        : document.getElementById('combinePolished');
    if (!el) return;
    readerFontSize = Math.max(12, Math.min(28, readerFontSize + delta));
    el.style.fontSize = readerFontSize + 'px';
    showToast(`Font size: ${readerFontSize}px`, 'info');
}

function toggleReaderTheme() {
    const el = readerIsFullscreen
        ? document.getElementById('fullscreenReaderContent')
        : document.getElementById('combinePolished');
    if (!el) return;
    readerThemes.forEach(t => el.classList.remove('theme-' + t));
    readerThemeIndex = (readerThemeIndex + 1) % readerThemes.length;
    el.classList.add('theme-' + readerThemes[readerThemeIndex]);
    const themeNames = { obsidian: '🌑 Obsidian', parchment: '📜 Parchment', midnight: '🌌 Midnight' };
    showToast(`Theme: ${themeNames[readerThemes[readerThemeIndex]]}`, 'info');
}

function toggleReaderFont() {
    const el = readerIsFullscreen
        ? document.getElementById('fullscreenReaderContent')
        : document.getElementById('combinePolished');
    if (!el) return;
    readerIsSerif = !readerIsSerif;
    el.classList.toggle('font-serif', readerIsSerif);
    el.classList.toggle('font-sans', !readerIsSerif);
    showToast(readerIsSerif ? 'Font: Serif (Lora)' : 'Font: Sans-Serif (Inter)', 'info');
}

function toggleReaderFullscreen() {
    if (readerIsFullscreen) {
        exitReaderFullscreen();
        return;
    }

    const source = document.getElementById('combinePolished');
    if (!source) return;

    // Build fullscreen overlay appended to <body> (bypasses parent backdrop-filter/transform)
    const overlay = document.createElement('div');
    overlay.id = 'readerFullscreenOverlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:9999;display:flex;flex-direction:column;animation:readerFadeIn 0.3s ease;';

    // Toolbar at top
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex;align-items:center;justify-content:flex-end;gap:6px;padding:10px 20px;background:rgba(10,10,15,0.95);border-bottom:1px solid rgba(255,255,255,0.08);flex-shrink:0;backdrop-filter:blur(12px);';
    toolbar.innerHTML = `
        <button class="btn btn-sm" onclick="adjustReaderFontSize(-1)" style="padding:4px 10px;font-size:12px;">A-</button>
        <button class="btn btn-sm" onclick="adjustReaderFontSize(1)" style="padding:4px 10px;font-size:12px;">A+</button>
        <button class="btn btn-sm" onclick="toggleReaderTheme()" style="padding:4px 10px;font-size:12px;">🎨 Theme</button>
        <button class="btn btn-sm" onclick="toggleReaderFont()" style="padding:4px 10px;font-size:12px;">🔤 Font</button>
        <div style="flex:1"></div>
        <span style="color:#888;font-size:12px;font-family:Inter,sans-serif;">↑↓ Scroll · Esc to exit</span>
        <button class="btn btn-sm" onclick="exitReaderFullscreen()" style="padding:4px 14px;font-size:12px;border-color:rgba(239,68,68,0.3);color:#ef4444;">✕ Exit</button>
    `;
    overlay.appendChild(toolbar);

    // Content area — clone the reader content
    const content = document.createElement('div');
    content.id = 'fullscreenReaderContent';
    content.className = source.className; // copy all theme/font classes
    content.innerHTML = source.innerHTML;
    content.style.cssText = 'flex:1;overflow-y:auto;max-height:none;padding:60px 80px;font-size:' + readerFontSize + 'px;';
    content.tabIndex = 0; // make focusable for keyboard
    overlay.appendChild(content);

    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden'; // prevent page scroll behind overlay
    content.focus(); // focus for keyboard scrolling

    readerIsFullscreen = true;
    document.addEventListener('keydown', readerFullscreenKeyHandler);
}

function exitReaderFullscreen() {
    const overlay = document.getElementById('readerFullscreenOverlay');
    if (overlay) overlay.remove();
    document.body.style.overflow = '';
    readerIsFullscreen = false;
    document.removeEventListener('keydown', readerFullscreenKeyHandler);
}

function readerFullscreenKeyHandler(e) {
    if (e.key === 'Escape') {
        exitReaderFullscreen();
        return;
    }
    const content = document.getElementById('fullscreenReaderContent');
    if (!content) return;
    const scrollAmount = 80;
    const pageScrollAmount = content.clientHeight * 0.85;

    if (e.key === 'ArrowDown') { e.preventDefault(); content.scrollBy({ top: scrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); content.scrollBy({ top: -scrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); content.scrollBy({ top: pageScrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'PageUp') { e.preventDefault(); content.scrollBy({ top: -pageScrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'Home') { e.preventDefault(); content.scrollTo({ top: 0, behavior: 'smooth' }); }
    else if (e.key === 'End') { e.preventDefault(); content.scrollTo({ top: content.scrollHeight, behavior: 'smooth' }); }
}

// ─── Arrow Key Scrolling for Inline Reader ───────────────────────
// When the Combine view is active and polished story is visible,
// arrow keys scroll the story reader instead of the page.
document.addEventListener('keydown', function(e) {
    // Only intercept when not in fullscreen (fullscreen has its own handler)
    if (readerIsFullscreen) return;
    // Only when Combine view is active
    const combineView = document.getElementById('viewCombine');
    if (!combineView || !combineView.classList.contains('active')) return;
    // Only when polished story is visible
    const polished = document.getElementById('combinePolished');
    const polishedCard = document.getElementById('combinePolishedCard');
    if (!polished || !polishedCard || polishedCard.classList.contains('hidden')) return;
    // Don't intercept if user is in an input/textarea
    const tag = document.activeElement?.tagName?.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    const scrollAmount = 80;
    const pageScrollAmount = polished.clientHeight * 0.85;

    if (e.key === 'ArrowDown') { e.preventDefault(); polished.scrollBy({ top: scrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); polished.scrollBy({ top: -scrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'PageDown') { e.preventDefault(); polished.scrollBy({ top: pageScrollAmount, behavior: 'smooth' }); }
    else if (e.key === 'PageUp') { e.preventDefault(); polished.scrollBy({ top: -pageScrollAmount, behavior: 'smooth' }); }
});

function switchToEntireStoryGeneration() {
    // 1. Switch view to generate
    switchView('generate');
    
    // 2. Set chapter count option to -1
    const chapterCountSelect = document.getElementById('genChapterCount');
    if (chapterCountSelect) {
        chapterCountSelect.value = '-1';
    }
    
    // 3. Set the active model to Gemini API if available
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        // Try to find the __gemini_api__ option
        const geminiOption = Array.from(modelSelect.options).find(opt => opt.value === '__gemini_api__');
        if (geminiOption) {
            modelSelect.value = '__gemini_api__';
            switchModel('__gemini_api__');
        }
    }
    showToast('Switched to Auto-Generation. Select "Entire Story" and click Generate!', 'info');
}

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    loadModels();
    loadSettings();
    document.getElementById('settingsActiveModel')?.addEventListener('change', (e) => {
        const pathInput = document.getElementById('setting_LLAMA_MODEL_PATH');
        if (pathInput && e.target.value) pathInput.value = e.target.value;
    });
    setStatus('online', 'Ready');
});

// ─── Premise Generator View Logic ────────────────────────────────
async function loadProjectPremise() {
    if (!currentProject) return;
    try {
        const res = await fetch(`/api/project/${currentProject}`);
        const data = await res.json();
        
        const metadata = data.state.metadata || {};
        document.getElementById('premiseSetting').value = metadata.setting || '';
        document.getElementById('premiseThemes').value = (metadata.themes || []).join(', ');
        
        // Load characters
        const charsObj = data.state.characters || {};
        premiseCharacters = charsObj;
        renderPremiseCharacters(charsObj);

        // Load premise steps
        const rawPremise = metadata.premise || '';
        premiseSteps = parsePremiseSteps(rawPremise);
        renderPremiseTimeline();
    } catch (e) {
        showToast('Failed to load project premise data', 'error');
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

function renderPremiseCharacters(charsObj) {
    const container = document.getElementById('premiseCharactersList');
    const keys = Object.keys(charsObj);
    if (keys.length === 0) {
        container.innerHTML = '<span class="text-dim text-sm">No characters defined.</span>';
        return;
    }
    container.innerHTML = keys.map(name => {
        const traits = (charsObj[name].traits || []).join(', ');
        return `
            <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border); padding:6px 10px; border-radius:4px; font-size:12px; display:flex; justify-content:space-between;">
                <span style="font-weight:600; color:var(--accent-primary);">${escHtml(name)}</span>
                <span class="text-dim" style="max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escHtml(traits)}</span>
            </div>
        `;
    }).join('');
}

function renderPremiseTimeline() {
    const container = document.getElementById('premiseTimelineContainer');
    if (premiseSteps.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="icon">📋</span>
                <h3>No beats defined</h3>
                <p>Dump a rough outline on the left and click "Generate Structured beats" or add steps manually.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = premiseSteps.map((step, idx) => {
        return `
            <div class="premise-step-card" data-index="${idx}">
                <span class="premise-step-badge">Step ${idx + 1}</span>
                <textarea class="premise-step-input" rows="2" oninput="syncStepText(${idx}, this.value)" placeholder="Describe this beat...">${escHtml(step)}</textarea>
                <div class="premise-step-actions">
                    <button class="premise-step-btn" onclick="reorderPremiseStep(${idx}, -1)" ${idx === 0 ? 'disabled' : ''} title="Move Up">▲</button>
                    <button class="premise-step-btn" onclick="reorderPremiseStep(${idx}, 1)" ${idx === premiseSteps.length - 1 ? 'disabled' : ''} title="Move Down">▼</button>
                    <button class="premise-step-btn btn-delete" onclick="deletePremiseStep(${idx})" title="Delete Beat">🗑️</button>
                </div>
            </div>
        `;
    }).join('');
}

function syncStepText(index, val) {
    if (premiseSteps[index] !== undefined) {
        premiseSteps[index] = val;
    }
}

function addBlankPremiseStep() {
    syncAllStepsFromUI();
    premiseSteps.push("");
    renderPremiseTimeline();
    
    setTimeout(() => {
        const textareas = document.querySelectorAll('.premise-step-input');
        if (textareas.length > 0) {
            textareas[textareas.length - 1].focus();
        }
    }, 50);
}

function syncAllStepsFromUI() {
    const textareas = document.querySelectorAll('.premise-step-input');
    textareas.forEach((ta, idx) => {
        if (premiseSteps[idx] !== undefined) {
            premiseSteps[idx] = ta.value;
        }
    });
}

function reorderPremiseStep(index, direction) {
    syncAllStepsFromUI();
    const targetIdx = index + direction;
    if (targetIdx < 0 || targetIdx >= premiseSteps.length) return;
    
    const temp = premiseSteps[index];
    premiseSteps[index] = premiseSteps[targetIdx];
    premiseSteps[targetIdx] = temp;
    
    renderPremiseTimeline();
}

function deletePremiseStep(index) {
    syncAllStepsFromUI();
    premiseSteps.splice(index, 1);
    renderPremiseTimeline();
}

async function aiGeneratePremise() {
    const idea = document.getElementById('premiseIdeaText').value.trim();
    if (!idea) {
        showToast('Please enter a rough story idea or timeline dump first.', 'error');
        return;
    }

    const btn = document.getElementById('btnPremiseGenerate');
    const oldText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="stage-spinner"></span> Generating beats...';

    const selectedModel = document.getElementById('modelSelect')?.value || '';

    try {
        const res = await fetch(`/api/project/${currentProject}/premise/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                characters: premiseCharacters,
                setting: document.getElementById('premiseSetting').value.trim(),
                themes: document.getElementById('premiseThemes').value.trim(),
                idea: idea,
                model: selectedModel
            })
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to generate premise');

        premiseSteps = data.steps || [];
        renderPremiseTimeline();
        showToast(`Successfully generated ${premiseSteps.length} story beats!`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = oldText;
    }
}

async function aiRefinePremiseFlow() {
    syncAllStepsFromUI();
    if (premiseSteps.length === 0) {
        showToast('Please add or generate some beats first.', 'error');
        return;
    }

    const btn = document.getElementById('btnPremiseRefine');
    const oldText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="stage-spinner"></span> Refining pacing...';

    const selectedModel = document.getElementById('modelSelect')?.value || '';

    try {
        const res = await fetch(`/api/project/${currentProject}/premise/refine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                steps: premiseSteps,
                characters: premiseCharacters,
                setting: document.getElementById('premiseSetting').value.trim(),
                themes: document.getElementById('premiseThemes').value.trim(),
                model: selectedModel
            })
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to refine premise');

        premiseSteps = data.steps || [];
        renderPremiseTimeline();
        showToast(`Successfully refined story flow!`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = oldText;
    }
}

async function aiExpandPremiseBeats() {
    syncAllStepsFromUI();
    if (premiseSteps.length === 0) {
        showToast('Please add or generate some beats first.', 'error');
        return;
    }

    const btn = document.getElementById('btnPremiseExpand');
    const oldText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="stage-spinner"></span> Expanding scenes...';

    const selectedModel = document.getElementById('modelSelect')?.value || '';

    try {
        const res = await fetch(`/api/project/${currentProject}/premise/expand`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                steps: premiseSteps,
                characters: premiseCharacters,
                setting: document.getElementById('premiseSetting').value.trim(),
                themes: document.getElementById('premiseThemes').value.trim(),
                model: selectedModel
            })
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to expand premise');

        premiseSteps = data.steps || [];
        renderPremiseTimeline();
        showToast(`Successfully expanded timeline to ${premiseSteps.length} beats!`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = oldText;
    }
}

async function savePremiseToProject() {
    syncAllStepsFromUI();
    if (!currentProject) return;

    const btn = document.getElementById('btnPremiseSave');
    const oldText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Saving...';

    const formatted = premiseSteps.map((step, idx) => `${idx + 1}. ${step}`).join('\n');
    const themesStr = document.getElementById('premiseThemes').value.trim();

    try {
        const res = await fetch(`/api/project/${currentProject}/state`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                metadata: {
                    premise: formatted,
                    setting: document.getElementById('premiseSetting').value.trim(),
                    themes: themesStr ? themesStr.split(',').map(t => t.trim()).filter(Boolean) : []
                }
            })
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || 'Failed to save premise');

        showToast('Story premise saved to project successfully!', 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = oldText;
    }
}

// ─── Vision Lab Module ────────────────────────────────────────────────
let selectedVisionFile = null;

async function checkVisionStatus() {
    const badge = document.getElementById('visionStatusBadge');
    if (!badge) return;
    try {
        const res = await fetch('/api/vision/status');
        const json = await res.json();
        if (json.success) {
            const data = json.data;
            let statusText = `Active: ${data.active_mode}`;
            if (data.local && data.local.available) {
                statusText += ' | Local (Ollama) Ready';
            } else {
                statusText += ' | Local Offline';
            }
            if (data.cloud && data.cloud.available) {
                statusText += ' | Gemini Cloud Ready';
            }
            badge.innerText = statusText;
            badge.className = 'badge badge-info';
        }
    } catch (e) {
        badge.innerText = 'Vision API Unreachable';
        badge.className = 'badge badge-warning';
    }
}

function handleVisionFileSelect(files) {
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showToast('Please select a valid image file (PNG, JPG, WEBP).', 'error');
        return;
    }
    selectedVisionFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('visionPreviewImg').src = e.target.result;
        document.getElementById('visionDropzoneContent').style.display = 'none';
        document.getElementById('visionImagePreview').style.display = 'block';
        document.getElementById('btnAnalyzeVision').disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearVisionFile() {
    selectedVisionFile = null;
    document.getElementById('visionFileInput').value = '';
    document.getElementById('visionPreviewImg').src = '';
    document.getElementById('visionDropzoneContent').style.display = 'block';
    document.getElementById('visionImagePreview').style.display = 'none';
    document.getElementById('btnAnalyzeVision').disabled = true;
    document.getElementById('visionResults').style.display = 'none';
}

let currentVisionSessionId = null;

async function runVisionAnalysis() {
    if (!selectedVisionFile) {
        showToast('Please select or drag an image first.', 'warning');
        return;
    }

    const btn = document.getElementById('btnAnalyzeVision');
    const oldText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Scanning Image (LLaVA)...';

    const formData = new FormData();
    formData.append('image', selectedVisionFile);

    try {
        const res = await fetch('/api/vision/chat/start', {
            method: 'POST',
            body: formData
        });
        const json = await res.json();
        if (!res.ok || !json.success) {
            throw new Error(json.error || 'Vision scan failed');
        }
        
        currentVisionSessionId = json.data.session_id;
        
        // Hide dropzone, show chat interface
        document.getElementById('visionDropzone').style.display = 'none';
        document.getElementById('visionChatInterface').style.display = 'block';
        
        // Clear chat history
        const history = document.getElementById('visionChatHistory');
        history.innerHTML = '';
        
        // Add initial system message with observation summary
        addChatMessage('system', `System: ${json.data.observations_summary}`);
        
        // Add initial Grok message
        addChatMessage('assistant', json.data.initial_message);
        
        showToast('Chat started!', 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Chat Started';
    }
}

function addChatMessage(role, content) {
    const history = document.getElementById('visionChatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    const label = role === 'assistant' ? 'Grok' : (role === 'user' ? 'You' : 'System');
    
    messageDiv.innerHTML = `
        <div class="chat-label">${label}</div>
        <div class="chat-bubble">${content}</div>
    `;
    
    history.appendChild(messageDiv);
    history.scrollTop = history.scrollHeight;
}

function handleVisionChatKeyPress(e) {
    if (e.key === 'Enter') {
        sendVisionChatMessage();
    }
}

async function sendVisionChatMessage() {
    if (!currentVisionSessionId) return;
    
    const input = document.getElementById('visionChatMessage');
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message to UI
    addChatMessage('user', message);
    input.value = '';
    
    const btn = document.getElementById('btnVisionChatSend');
    btn.disabled = true;
    
    // Add typing indicator
    const history = document.getElementById('visionChatHistory');
    const typingDiv = document.createElement('div');
    typingDiv.className = `chat-message assistant`;
    typingDiv.id = 'visionChatTyping';
    typingDiv.innerHTML = `
        <div class="chat-label">Grok</div>
        <div class="chat-bubble">Typing...</div>
    `;
    history.appendChild(typingDiv);
    history.scrollTop = history.scrollHeight;
    
    try {
        const res = await fetch('/api/vision/chat/message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentVisionSessionId,
                message: message
            })
        });
        
        const json = await res.json();
        
        // Remove typing indicator
        const typingEl = document.getElementById('visionChatTyping');
        if (typingEl) typingEl.remove();
        
        if (!res.ok || !json.success) {
            throw new Error(json.error || 'Message failed');
        }
        
        // Add response to UI
        addChatMessage('assistant', json.data.response);
        
    } catch (e) {
        // Remove typing indicator on error
        const typingEl = document.getElementById('visionChatTyping');
        if (typingEl) typingEl.remove();
        
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
        input.focus();
    }
}

function setupVisionDropzone() {
    const dropzone = document.getElementById('visionDropzone');
    if (!dropzone || dropzone.dataset.initialized) return;
    dropzone.dataset.initialized = 'true';

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleVisionFileSelect(files);
    }, false);
}
