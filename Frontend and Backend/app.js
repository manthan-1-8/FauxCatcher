// CONFIG: change this if your backend runs on another host/port
const API_BASE = (localStorage.getItem('visioncop_api_base') || 'http://127.0.0.1:8000').replace(/\/$/, '');

const els = {
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  browseBtn: document.getElementById('browseBtn'),
  preview: document.getElementById('preview'),
  previewImg: document.getElementById('previewImg'),
  previewMeta: document.getElementById('previewMeta'),
  toast: document.getElementById('toast'),
  results: document.getElementById('results'),
  resultsGrid: document.getElementById('resultsGrid'),
  rawOutput: document.getElementById('rawOutput'),
  btnSearch: document.getElementById('btnSearch'),
  btnVerify: document.getElementById('btnVerify'),
  btnCaption: document.getElementById('btnCaption'),
  btnRanked: document.getElementById('btnRanked'),
  btnAIDetect: document.getElementById('btnAIDetect'),
  deleteBtn: document.getElementById('deleteBtn'),
  // modal
  topkModal: document.getElementById('topkModal'),
  topkInput: document.getElementById('topKInput'),
  topkCancel: document.getElementById('topkCancel'),
  topkConfirm: document.getElementById('topkConfirm'),
};

let currentFile = null;

function toast(msg, duration = 2400) {
  els.toast.textContent = msg;
  els.toast.classList.add('show');
  setTimeout(() => els.toast.classList.remove('show'), duration);
}

function setLoading(button, loading) {
  if (!button) return;
  if (loading) {
    button.dataset.prevText = button.textContent;
    button.disabled = true;
    button.textContent = 'Working…';
  } else {
    button.disabled = false;
    button.textContent = button.dataset.prevText || button.textContent;
  }
}

function resetResults() {
  els.resultsGrid.innerHTML = '';
  els.rawOutput.textContent = '';
  els.rawOutput.hidden = true;
}

function clearImage() {
  currentFile = null;
  els.fileInput.value = '';
  els.previewImg.src = '';
  els.previewImg.style.display = 'none';
  els.previewMeta.textContent = '';
  resetResults();
  toast('Image removed. Upload a new one.');
}

function renderResults(items) {
  resetResults();
  if (!items || !items.length) {
    els.results.hidden = false;
    els.resultsGrid.innerHTML = '<p style="color:var(--muted)">No results found.</p>';
    return;
  }
  const frag = document.createDocumentFragment();
  items.forEach(it => {
    const card = document.createElement('div');
    card.className = 'result-card';
    const img = document.createElement('img');
    img.src = it.image_url || it.url || '';
    img.alt = 'result image';
    const meta = document.createElement('div');
    meta.className = 'result-card__meta';
    const left = document.createElement('span');
    left.textContent = (it.image_url || it.url || '').slice(0, 28) || 'Image';
    const score = document.createElement('span');
    if (typeof it.similarity === 'number') {
      score.className = 'result-card__score';
      score.textContent = `${(it.similarity * 100).toFixed(1)}%`;
    } else {
      score.textContent = '';
    }
    meta.appendChild(left);
    meta.appendChild(score);
    card.appendChild(img);
    card.appendChild(meta);
    frag.appendChild(card);
  });
  els.resultsGrid.appendChild(frag);
  els.results.hidden = false;
}

function showRaw(obj) {
  els.rawOutput.hidden = false;
  els.rawOutput.textContent = JSON.stringify(obj, null, 2);
}

function setPreview(file) {
  if (!file) return;
  currentFile = file;
  const url = URL.createObjectURL(file);
  els.previewImg.src = url;
  els.previewImg.style.display = 'block';
  els.previewMeta.textContent = `${file.name} · ${(file.size/1024).toFixed(1)} KB`;
}

function getFormData() {
  if (!currentFile) throw new Error('Please upload an image first.');
  const fd = new FormData();
  fd.append('file', currentFile);
  return fd;
}

async function callJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || res.statusText);
  }
  return res.json();
}

// Event bindings
els.browseBtn.addEventListener('click', () => els.fileInput.click());
els.dropzone.addEventListener('click', () => els.fileInput.click());
els.fileInput.addEventListener('change', e => { const f = e.target.files?.[0]; if (f) setPreview(f); });
els.dropzone.addEventListener('dragover', e => { e.preventDefault(); });
els.dropzone.addEventListener('drop', e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) setPreview(f); });
els.deleteBtn.addEventListener('click', clearImage);

// Actions
els.btnSearch.addEventListener('click', async () => {
  try { setLoading(els.btnSearch, true);
    const data = await callJSON(`${API_BASE}/images/search/`, { method:'POST', body:getFormData() });
    renderResults(data.results); showRaw(data); toast('Found similar images');
  } catch (err) { toast(`Error: ${err.message}`); } finally { setLoading(els.btnSearch, false); }
});

els.btnVerify.addEventListener('click', async () => {
  try { setLoading(els.btnVerify, true);
    const data = await callJSON(`${API_BASE}/images/verify/`, { method:'POST', body:getFormData() });
    renderResults([]); showRaw(data); toast(`Verification: ${data.verdict || 'unknown'}`);
  } catch (err) { toast(`Error: ${err.message}`); } finally { setLoading(els.btnVerify, false); }
});

els.btnCaption.addEventListener('click', async () => {
  try { setLoading(els.btnCaption, true);
    const data = await callJSON(`${API_BASE}/images/caption-check/`, { method:'POST', body:getFormData() });
    renderResults([]); showRaw(data);
    const msg = data.consistent ? `Caption matches (score ${Math.round(data.score*100)}%)` : `Caption mismatch (score ${Math.round(data.score*100)}%)`;
    toast(msg);
  } catch (err) { toast(`Error: ${err.message}`); } finally { setLoading(els.btnCaption, false); }
});

function openTopKModal() {
  els.topkInput.value = '10';
  els.topkModal.classList.add('show');
  els.topkModal.setAttribute('aria-hidden', 'false');
  els.topkInput.focus();
}
function closeTopKModal() {
  els.topkModal.classList.remove('show');
  els.topkModal.setAttribute('aria-hidden', 'true');
}
els.topkCancel.addEventListener('click', closeTopKModal);
els.topkConfirm.addEventListener('click', async () => {
  const topK = Math.min(100, Math.max(1, parseInt(els.topkInput.value || '10', 10)));
  closeTopKModal();
  try { setLoading(els.btnRanked, true);
    const url = `${API_BASE}/images/ranked-search/?top_k=${encodeURIComponent(topK)}`;
    const data = await callJSON(url, { method:'POST', body:getFormData() });
    renderResults(data.results); showRaw(data); toast(`Showing top ${data.top_k || topK}`);
  } catch (err) { toast(`Error: ${err.message}`); } finally { setLoading(els.btnRanked, false); }
});
els.btnRanked.addEventListener('click', openTopKModal);

els.btnAIDetect.addEventListener('click', async () => {
  try { setLoading(els.btnAIDetect, true);
    const data = await callJSON(`${API_BASE}/images/ai-detection/`, { method:'POST', body:getFormData() });
    renderResults([]); showRaw(data);
    const flag = data.is_ai_generated ? 'AI‑generated' : 'Likely human‑captured';
    const conf = data.confidence != null ? ` (${Math.round(data.confidence*100)}% conf.)` : '';
    toast(`${flag}${conf}`);
  } catch (err) { toast(`Error: ${err.message}`); } finally { setLoading(els.btnAIDetect, false); }
});

window.addEventListener('keydown', e => {
  if (e.key.toLowerCase() === 'u') els.fileInput.click();
  if (e.key === 'Escape') closeTopKModal();
});

console.log('VISION‑COP frontend loaded. Backend:', API_BASE);