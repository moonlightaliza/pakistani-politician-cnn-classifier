// ── Config ─────────────────────────────────────────────────────────────────
const API_BASE = ''; // Change to your server URL

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let selectedModel = 'resnet50';

// ── Elements ───────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const previewWrap    = document.getElementById('previewWrap');
const previewImg     = document.getElementById('previewImg');
const clearBtn       = document.getElementById('clearBtn');
const submitBtn      = document.getElementById('submitBtn');
const topNSelect     = document.getElementById('topNSelect');
const resultsSection = document.getElementById('resultsSection');
const resultList     = document.getElementById('resultList');
const resultsModelTag= document.getElementById('resultsModelTag');
const errorBox       = document.getElementById('errorBox');
const errorMsg       = document.getElementById('errorMsg');
const modelBtns      = document.querySelectorAll('.model-btn');

// ── Model toggle ───────────────────────────────────────────────────────────
modelBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    modelBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedModel = btn.dataset.model;
  });
});

// ── Drag & drop ────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (file.size > 10 * 1024 * 1024) {
    showError('Image is too large. Please use an image under 10 MB.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropZone.style.display = 'none';
    previewWrap.classList.add('visible');
    submitBtn.disabled = false;
    hideError();
    resetResults();
  };
  reader.readAsDataURL(file);
}

// ── Clear ──────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.classList.remove('visible');
  dropZone.style.display = '';
  submitBtn.disabled = true;
  resetResults();
  hideError();
});

// ── Submit ─────────────────────────────────────────────────────────────────
submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();
  resetResults();

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('model', selectedModel);
  formData.append('top_n', topNSelect.value);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    showError(err.message || 'Failed to connect to the server. Make sure the API is running.');
  } finally {
    setLoading(false);
  }
});

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  resultsModelTag.textContent = data.model;
  resultList.innerHTML = '';

  data.predictions.forEach((pred, i) => {
    const li = document.createElement('li');
    li.className = 'result-item' + (i === 0 ? ' top-result' : '');
    li.innerHTML = `
      <div class="result-rank">${pred.rank}</div>
      <div class="result-info">
        <div class="result-name">${escapeHtml(pred.politician)}</div>
        <div class="result-bar-wrap">
          <div class="result-bar" data-pct="${pred.confidence}"></div>
        </div>
      </div>
      <div class="result-confidence">${pred.confidence}%</div>
    `;
    resultList.appendChild(li);
  });

  resultsSection.classList.add('visible');

  // Animate bars after DOM insert
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      resultList.querySelectorAll('.result-bar').forEach(bar => {
        bar.style.width = bar.dataset.pct + '%';
      });
    });
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────
function resetResults() {
  resultsSection.classList.remove('visible');
  resultList.innerHTML = '';
}
function showError(msg) {
  errorMsg.textContent = msg;
  errorBox.classList.add('visible');
}
function hideError() {
  errorBox.classList.remove('visible');
}
function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.classList.toggle('loading', on);
}
function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}