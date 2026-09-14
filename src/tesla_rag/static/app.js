document.addEventListener('DOMContentLoaded', () => {
    const statusText = document.getElementById('status-text');
    const statChunks = document.getElementById('stat-chunks');
    const ragForm = document.getElementById('rag-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const topKSelect = document.getElementById('top-k-select');
    const genModeSelect = document.getElementById('gen-mode-select');
    const apiKeyGroup = document.getElementById('api-key-group');
    const geminiKeyInput = document.getElementById('gemini-key-input');
    
    const loadingState = document.getElementById('loading-state');
    const resultCard = document.getElementById('result-card');
    const resultModeBadge = document.getElementById('result-mode-badge');
    const resultTimingBadge = document.getElementById('result-timing-badge');
    const answerContent = document.getElementById('answer-content');
    const copyBtn = document.getElementById('copy-answer-btn');
    
    const sourcesToggle = document.getElementById('sources-toggle');
    const sourcesList = document.getElementById('sources-list');
    const sourceCount = document.getElementById('source-count');
    const chips = document.querySelectorAll('.chip');

    // 1. Fetch initial vector store stats
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.status === 'ready') {
                statusText.textContent = 'Vector Space Ready';
                statChunks.textContent = `${data.total_chunks} Chunks`;
            }
        } catch (err) {
            statusText.textContent = 'Vector Store Offline';
            console.error('Stats load error:', err);
        }
    }
    loadStats();

    // 2. Mode change listener
    genModeSelect.addEventListener('change', () => {
        if (genModeSelect.value === 'gemini') {
            apiKeyGroup.style.display = 'block';
        } else {
            apiKeyGroup.style.display = 'none';
        }
    });

    // 3. Simple Markdown formatter
    function formatMarkdown(text) {
        if (!text) return '';
        let formatted = text
            // Escape basic HTML
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Bullet points
            .replace(/^• (.*?)$/gm, '<li>$1</li>')
            .replace(/^- (.*?)$/gm, '<li>$1</li>')
            // Numbered items
            .replace(/^(\d+)\. (.*?)$/gm, '<li style="list-style-type: decimal;">$2</li>');

        // Wrap list items
        formatted = formatted.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
        // Clean duplicate ul tags if consecutive
        formatted = formatted.replace(/<\/ul>\s*<ul>/g, '');
        
        // Paragraphs
        const paragraphs = formatted.split(/\n\n+/);
        return paragraphs.map(p => {
            if (p.startsWith('<ul>') || p.startsWith('<li') || p.startsWith('<table')) {
                return p;
            }
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');
    }

    // 4. Query Execution
    async function executeQuery(queryText) {
        if (!queryText.trim()) return;

        queryInput.value = queryText;
        resultCard.style.display = 'none';
        loadingState.style.display = 'flex';
        submitBtn.disabled = true;

        const payload = {
            query: queryText.trim(),
            top_k: parseInt(topKSelect.value, 10),
            mode: genModeSelect.value,
            api_key: geminiKeyInput.value.trim() || undefined
        };

        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (res.ok) {
                // Populate result
                resultModeBadge.textContent = data.generator_mode || 'Local Generator';
                const totalMs = data.timing?.total_ms || 0;
                resultTimingBadge.textContent = `${totalMs}ms (${data.timing?.retrieval_ms}ms search)`;
                
                answerContent.innerHTML = formatMarkdown(data.answer);

                // Render sources
                const sources = data.sources || [];
                sourceCount.textContent = sources.length;
                sourcesList.innerHTML = '';
                
                sources.forEach((src, idx) => {
                    const card = document.createElement('div');
                    card.className = 'source-card';
                    card.innerHTML = `
                        <div class="source-meta">
                            <span class="source-page">Source #${idx + 1} &bull; Page ${src.page_number}</span>
                            <span class="source-score">Similarity: ${(src.score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="source-snippet">${src.snippet}</div>
                    `;
                    sourcesList.appendChild(card);
                });

                resultCard.style.display = 'flex';
                // Smooth scroll to answer
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert(`Error: ${data.error || 'Failed to generate answer'}`);
            }
        } catch (err) {
            alert(`Query request failed: ${err.message}`);
        } finally {
            loadingState.style.display = 'none';
            submitBtn.disabled = false;
        }
    }

    // 5. Form Submit handler
    ragForm.addEventListener('submit', (e) => {
        e.preventDefault();
        executeQuery(queryInput.value);
    });

    // 6. Sample Query Chips
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            executeQuery(query);
        });
    });

    // 7. Sources Accordion Toggle
    let sourcesOpen = true;
    sourcesToggle.addEventListener('click', () => {
        sourcesOpen = !sourcesOpen;
        sourcesList.style.display = sourcesOpen ? 'flex' : 'none';
        const arrow = sourcesToggle.querySelector('.accordion-arrow');
        arrow.style.transform = sourcesOpen ? 'rotate(0deg)' : 'rotate(-90deg)';
    });

    // 8. Copy Answer Button
    copyBtn.addEventListener('click', async () => {
        const text = answerContent.innerText;
        try {
            await navigator.clipboard.writeText(text);
            const span = copyBtn.querySelector('span');
            span.textContent = 'Copied!';
            setTimeout(() => { span.textContent = 'Copy'; }, 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
    });
});
