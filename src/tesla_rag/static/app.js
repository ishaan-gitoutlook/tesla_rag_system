document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const form = $('rag-form'), input = $('query-input'), submit = $('submit-btn');
  const loading = $('loading-state'), result = $('result-card'), error = $('error-message');
  const mode = $('gen-mode-select'), keyGroup = $('api-key-group');

  const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
  const formatAnswer = (text) => {
    const safe = escapeHTML(text);
    return safe.split(/\n\s*\n/).map((block) => {
      if (/^(?:[-•] |\d+\. )/m.test(block)) {
        const items = block.split('\n').filter(Boolean).map((line) => `<li>${line.replace(/^(?:[-•]|\d+\.)\s*/, '')}</li>`).join('');
        return `<ul>${items}</ul>`;
      }
      return `<p>${block.replace(/\n/g, '<br>')}</p>`;
    }).join('');
  };
  const showError = (message) => { error.textContent = message; error.hidden = false; };
  const clearError = () => { error.hidden = true; error.textContent = ''; };

  async function loadStats() {
    try {
      const response = await fetch('/api/stats');
      if (!response.ok) throw new Error('The vector store is unavailable.');
      const data = await response.json();
      $('status-text').textContent = 'Vector store ready';
      $('stat-chunks').textContent = `${data.total_chunks} chunks`;
      $('stat-model').textContent = data.embedding_model || 'Local embeddings';
    } catch (err) {
      $('status-text').textContent = 'Vector store offline';
      $('status-dot').style.color = '#e82127';
      $('stat-chunks').textContent = 'Unavailable';
      $('stat-model').textContent = 'Unavailable';
    }
  }
  loadStats();

  input.addEventListener('input', () => { $('query-count').textContent = input.value.length; });
  mode.addEventListener('change', () => { keyGroup.hidden = mode.value !== 'gemini'; });
  input.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') form.requestSubmit(); });

  async function executeQuery(query) {
    const trimmed = query.trim();
    if (!trimmed) { showError('Enter a question before searching.'); input.focus(); return; }
    clearError(); input.value = trimmed; $('query-count').textContent = trimmed.length;
    result.hidden = true; loading.hidden = false; submit.disabled = true;
    try {
      const response = await fetch('/api/query', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: trimmed, top_k: Number($('top-k-select').value), mode: mode.value, api_key: $('gemini-key-input').value.trim() || undefined}) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The query could not be completed.');
      $('result-mode-badge').textContent = data.generator_mode || 'Local model';
      $('result-timing-badge').textContent = `${data.timing?.total_ms ?? 0} ms total · ${data.timing?.retrieval_ms ?? 0} ms retrieval`;
      $('answer-content').innerHTML = formatAnswer(data.answer);
      const sources = Array.isArray(data.sources) ? data.sources : [];
      $('source-count').textContent = sources.length;
      const list = $('sources-list'); list.replaceChildren();
      sources.forEach((source, index) => {
        const card = document.createElement('article'); card.className = 'source-card';
        const meta = document.createElement('div'); meta.className = 'source-meta';
        const page = document.createElement('span'); page.textContent = `Source ${index + 1} · Page ${source.page_number ?? '—'}`;
        const score = document.createElement('span'); score.className = 'source-score'; score.textContent = `Similarity: ${((source.score || 0) * 100).toFixed(1)}%`;
        meta.append(page, score); const snippet = document.createElement('div'); snippet.className = 'source-snippet'; snippet.textContent = source.snippet || '';
        card.append(meta, snippet); list.append(card);
      });
      result.hidden = false; result.scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (err) { showError(err.message); }
    finally { loading.hidden = true; submit.disabled = false; }
  }
  form.addEventListener('submit', (event) => { event.preventDefault(); executeQuery(input.value); });
  document.querySelectorAll('.suggestion').forEach((button) => button.addEventListener('click', () => executeQuery(button.dataset.query)));
  $('copy-answer-btn').addEventListener('click', async () => { try { await navigator.clipboard.writeText($('answer-content').innerText); $('copy-answer-btn').textContent = 'Copied'; setTimeout(() => $('copy-answer-btn').textContent = 'Copy answer', 1600); } catch { showError('Copy is not available in this browser.'); } });
});
