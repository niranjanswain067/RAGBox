/**
 * RAGBox - Frontend Application Logic (Vanilla JavaScript)
 * Communicates with the FastAPI backend to provide clean, responsive,
 * and transparent RAG workflows with interactive Teaching Mode telemetry.
 */

// Application State
const state = {
    selectedFile: null,
    settings: {
        topK: 3,
        temperature: 0.2,
        chunkSize: 500,
        chunkOverlap: 100
    },
    lastIngestion: null,
    lastQAResult: null
};

// DOM Element References
const elements = {
    // Header & Status
    statusBadge: document.getElementById('statusBadge'),
    statusText: document.getElementById('statusText'),
    settingsBtn: document.getElementById('settingsBtn'),
    clearDbBtn: document.getElementById('clearDbBtn'),
    
    // Upload & Ingestion
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    selectedFileInfo: document.getElementById('selectedFileInfo'),
    fileNameDisplay: document.getElementById('fileNameDisplay'),
    processBtn: document.getElementById('processBtn'),
    progressContainer: document.getElementById('progressContainer'),
    progressStatus: document.getElementById('progressStatus'),
    docStatusBadge: document.getElementById('docStatusBadge'),
    docSummaryBar: document.getElementById('docSummaryBar'),
    summaryDocName: document.getElementById('summaryDocName'),
    summaryPages: document.getElementById('summaryPages'),
    summaryChunks: document.getElementById('summaryChunks'),
    
    // Q&A
    questionForm: document.getElementById('questionForm'),
    questionInput: document.getElementById('questionInput'),
    askBtn: document.getElementById('askBtn'),
    askSpinner: document.getElementById('askSpinner'),
    teachingToggle: document.getElementById('teachingToggle'),
    suggestionsBar: document.querySelector('.suggestions-bar'),
    
    // Answer & Sources
    answerSection: document.getElementById('answerSection'),
    answerText: document.getElementById('answerText'),
    sourcesContainer: document.getElementById('sourcesContainer'),
    sourcesList: document.getElementById('sourcesList'),
    
    // Teaching Mode
    teachingModeSection: document.getElementById('teachingModeSection'),
    teachPages: document.getElementById('teachPages'),
    teachChunks: document.getElementById('teachChunks'),
    teachQuery: document.getElementById('teachQuery'),
    retrievedChunksList: document.getElementById('retrievedChunksList'),
    teachPrompt: document.getElementById('teachPrompt'),
    teachReasoning: document.getElementById('teachReasoning'),
    teachModel: document.getElementById('teachModel'),
    teachTime: document.getElementById('teachTime'),
    teachTokens: document.getElementById('teachTokens'),
    teachSpeed: document.getElementById('teachSpeed'),
    
    // Settings Modal
    settingsModal: document.getElementById('settingsModal'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    settingTopK: document.getElementById('settingTopK'),
    settingTemp: document.getElementById('settingTemp'),
    settingChunkSize: document.getElementById('settingChunkSize'),
    settingOverlap: document.getElementById('settingOverlap'),
    topKVal: document.getElementById('topKVal'),
    tempVal: document.getElementById('tempVal'),
    chunkSizeVal: document.getElementById('chunkSizeVal'),
    overlapVal: document.getElementById('overlapVal')
};


// -------------------------------------------------------------
// Initialization & Health Check
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    initEventListeners();
    checkSystemHealth();
});

async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('Health check failed');
        const data = await response.json();
        
        const dot = elements.statusBadge.querySelector('.status-dot');
        if (data.model_available) {
            dot.classList.add('online');
            elements.statusText.textContent = `Qwen Ready • ${data.indexed_chunks} Chunks`;
        } else {
            elements.statusText.textContent = `⚠️ Qwen Model Missing`;
        }

        if (data.indexed_chunks > 0) {
            elements.docStatusBadge.textContent = `${data.source_count} Document(s) Active`;
            elements.docStatusBadge.className = 'badge-success';
            
            // Show summary if documents exist
            elements.docSummaryBar.classList.remove('hidden');
            elements.summaryDocName.textContent = data.indexed_sources.join(', ');
            elements.summaryPages.textContent = `${data.source_count} file(s)`;
            elements.summaryChunks.textContent = data.indexed_chunks;
        }
    } catch (err) {
        elements.statusText.textContent = 'Backend Offline';
    }
}


// -------------------------------------------------------------
// Event Listeners
// -------------------------------------------------------------
function initEventListeners() {
    // Drag & Drop
    const { dropzone, fileInput } = elements;
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Process Document Button
    elements.processBtn.addEventListener('click', processDocument);

    // Question Submission
    elements.questionForm.addEventListener('submit', handleQuestionSubmit);

    // Quick suggestions chips
    elements.suggestionsBar.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip-btn');
        if (chip && chip.dataset.question) {
            elements.questionInput.value = chip.dataset.question;
            elements.questionForm.dispatchEvent(new Event('submit'));
        }
    });

    // Teaching Mode Toggle
    elements.teachingToggle.addEventListener('change', (e) => {
        if (e.target.checked && state.lastQAResult) {
            elements.teachingModeSection.classList.remove('hidden');
        } else {
            elements.teachingModeSection.classList.add('hidden');
        }
    });

    // Accordion expand/collapse
    document.querySelectorAll('.accordion-header').forEach(header => {
        header.addEventListener('click', () => {
            const item = header.parentElement;
            const wasActive = item.classList.contains('active');
            item.classList.toggle('active', !wasActive);
            const icon = header.querySelector('.acc-icon');
            if (icon) icon.textContent = wasActive ? '+' : '−';
        });
    });

    // Settings Modal
    elements.settingsBtn.addEventListener('click', () => elements.settingsModal.classList.remove('hidden'));
    elements.closeSettingsBtn.addEventListener('click', () => elements.settingsModal.classList.add('hidden'));
    elements.saveSettingsBtn.addEventListener('click', () => elements.settingsModal.classList.add('hidden'));

    // Range Sliders
    bindSlider(elements.settingTopK, elements.topKVal, 'topK', parseInt);
    bindSlider(elements.settingTemp, elements.tempVal, 'temperature', parseFloat);
    bindSlider(elements.settingChunkSize, elements.chunkSizeVal, 'chunkSize', parseInt);
    bindSlider(elements.settingOverlap, elements.overlapVal, 'chunkOverlap', parseInt);

    // Clear Database Button
    elements.clearDbBtn.addEventListener('click', clearDatabase);
}

function bindSlider(slider, display, key, parser) {
    slider.addEventListener('input', (e) => {
        const val = parser(e.target.value);
        display.textContent = val;
        state.settings[key] = val;
        localStorage.setItem('ragbox_settings', JSON.stringify(state.settings));
    });
}

function loadSettings() {
    const saved = localStorage.getItem('ragbox_settings');
    if (saved) {
        try {
            state.settings = { ...state.settings, ...JSON.parse(saved) };
        } catch (e) {}
    }
    // Update controls
    elements.settingTopK.value = state.settings.topK;
    elements.topKVal.textContent = state.settings.topK;
    elements.settingTemp.value = state.settings.temperature;
    elements.tempVal.textContent = state.settings.temperature;
    elements.settingChunkSize.value = state.settings.chunkSize;
    elements.chunkSizeVal.textContent = state.settings.chunkSize;
    elements.settingOverlap.value = state.settings.chunkOverlap;
    elements.overlapVal.textContent = state.settings.chunkOverlap;
}


// -------------------------------------------------------------
// File Selection & Document Ingestion
// -------------------------------------------------------------
function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Please select a valid PDF file.');
        return;
    }
    state.selectedFile = file;
    elements.fileNameDisplay.textContent = file.name;
    elements.selectedFileInfo.classList.remove('hidden');
    elements.docStatusBadge.textContent = 'File ready to process';
    elements.docStatusBadge.className = 'badge';
}

async function processDocument() {
    if (!state.selectedFile) return;

    elements.processBtn.disabled = true;
    elements.progressContainer.classList.remove('hidden');
    elements.progressStatus.textContent = 'Extracting pages from PDF...';

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    formData.append('chunk_size', state.settings.chunkSize);
    formData.append('chunk_overlap', state.settings.chunkOverlap);

    try {
        elements.progressStatus.textContent = 'Generating embeddings and storing in ChromaDB...';
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Upload failed');

        state.lastIngestion = data;

        // Update UI
        elements.progressContainer.classList.add('hidden');
        elements.docSummaryBar.classList.remove('hidden');
        elements.summaryDocName.textContent = data.source;
        elements.summaryPages.textContent = data.total_pages;
        elements.summaryChunks.textContent = data.chunks_stored || data.chunks_created;
        
        elements.docStatusBadge.textContent = 'Indexed in ChromaDB';
        elements.docStatusBadge.className = 'badge-success';

        // Update Header Status
        checkSystemHealth();

        // Update Step 1 in Teaching Mode
        elements.teachPages.textContent = data.total_pages;
        elements.teachChunks.textContent = data.chunks_created;

    } catch (err) {
        elements.progressContainer.classList.add('hidden');
        alert(`Error processing document: ${err.message}`);
    } finally {
        elements.processBtn.disabled = false;
    }
}


// -------------------------------------------------------------
// Question Answering Workflow
// -------------------------------------------------------------
async function handleQuestionSubmit(e) {
    e.preventDefault();
    const question = elements.questionInput.value.trim();
    if (!question) return;

    // UI Loading state
    elements.askBtn.disabled = true;
    elements.askSpinner.classList.remove('hidden');
    elements.answerSection.classList.remove('hidden');
    elements.answerText.innerHTML = '<span style="color:#94A3B8;">Retrieving chunks and generating answer via local Qwen 3.0 INT4...</span>';

    try {
        const payload = {
            question: question,
            top_k: state.settings.topK,
            temperature: state.settings.temperature,
            max_tokens: 512
        };

        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Failed to generate answer');

        state.lastQAResult = result;
        renderAnswerAndSources(result);

        // Update Teaching Mode
        if (elements.teachingToggle.checked) {
            elements.teachingModeSection.classList.remove('hidden');
            renderTeachingMode(result);
        }

    } catch (err) {
        elements.answerText.innerHTML = `<span style="color:#DC2626;">Error: ${err.message}</span>`;
    } finally {
        elements.askBtn.disabled = false;
        elements.askSpinner.classList.add('hidden');
    }
}

function renderAnswerAndSources(result) {
    // 1. Answer text
    elements.answerText.textContent = result.answer;

    // 2. Sources list
    elements.sourcesList.innerHTML = '';
    if (result.sources && result.sources.length > 0) {
        elements.sourcesContainer.classList.remove('hidden');
        result.sources.forEach(src => {
            const badge = document.createElement('div');
            badge.className = 'source-badge';
            badge.innerHTML = `📄 ${src.source} <span class="page-num">Page ${src.page}</span>`;
            elements.sourcesList.appendChild(badge);
        });
    } else {
        if (!result.has_answer) {
            elements.sourcesContainer.classList.remove('hidden');
            elements.sourcesList.innerHTML = '<span style="font-size:0.85rem; color:#94A3B8;">No sources cited (information absent from document).</span>';
        } else {
            elements.sourcesContainer.classList.add('hidden');
        }
    }
}


// -------------------------------------------------------------
// Teaching Mode Telemetry Rendering
// -------------------------------------------------------------
function renderTeachingMode(result) {
    // Step 1: Ingestion
    if (state.lastIngestion) {
        elements.teachPages.textContent = state.lastIngestion.total_pages;
        elements.teachChunks.textContent = state.lastIngestion.chunks_created;
    }

    // Step 2: Query
    elements.teachQuery.textContent = result.question;

    // Step 3: Retrieved Chunks
    elements.retrievedChunksList.innerHTML = '';
    const chunks = result.retrieved_chunks || [];
    chunks.forEach((chunk, idx) => {
        const card = document.createElement('div');
        card.className = 'retrieved-chunk-card';
        const simPct = (chunk.similarity * 100).toFixed(1);
        card.innerHTML = `
            <div class="chunk-meta">
                <span>Excerpt ${idx + 1} — ${chunk.source} (Page ${chunk.page})</span>
                <span>Similarity: ${chunk.similarity.toFixed(4)} (${simPct}%)</span>
            </div>
            <div class="chunk-body">${escapeHtml(chunk.text)}</div>
        `;
        elements.retrievedChunksList.appendChild(card);
    });

    // Step 4: Prompt
    elements.teachPrompt.textContent = result.prompt || 'No prompt constructed.';

    // Step 5: Model Reasoning
    if (elements.teachReasoning) {
        elements.teachReasoning.textContent = result.reasoning || 'No internal reasoning steps generated (direct answer).';
    }

    // Step 6: LLM Telemetry
    const stats = result.generation_stats || {};
    elements.teachModel.textContent = stats.model_name || 'Qwen 3.0 INT4';
    elements.teachTime.textContent = stats.generation_time_sec ? `${stats.generation_time_sec} s` : '-';
    elements.teachTokens.textContent = stats.completion_tokens || '-';
    elements.teachSpeed.textContent = stats.tokens_per_second ? `${stats.tokens_per_second} t/s` : '-';
}


// -------------------------------------------------------------
// Database Reset
// -------------------------------------------------------------
async function clearDatabase() {
    if (!confirm('Are you sure you want to clear the entire ChromaDB collection?')) {
        return;
    }

    try {
        const response = await fetch('/api/clear', { method: 'POST' });
        if (!response.ok) throw new Error('Clear failed');

        alert('ChromaDB collection cleared successfully.');
        state.selectedFile = null;
        state.lastIngestion = null;
        state.lastQAResult = null;

        elements.docSummaryBar.classList.add('hidden');
        elements.selectedFileInfo.classList.add('hidden');
        elements.answerSection.classList.add('hidden');
        elements.teachingModeSection.classList.add('hidden');
        elements.fileInput.value = '';

        checkSystemHealth();
    } catch (err) {
        alert(`Error clearing database: ${err.message}`);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
