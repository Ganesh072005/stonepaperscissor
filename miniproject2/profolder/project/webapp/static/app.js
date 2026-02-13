const listEl = document.getElementById('desc-list');
const pillEl = document.getElementById('status-pill');
const geminiNote = document.getElementById('gemini-note');
const btnWebcam = document.getElementById('btn-webcam');
const btnDroid = document.getElementById('btn-droidcam');
const fileVideo = document.getElementById('file-video');

function renderItems(items) {
  listEl.innerHTML = '';
  if (!items || items.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'desc-item';
    empty.innerHTML = '<div></div><div><p class="label">Waiting for detections…</p><p class="text">When objects are recognized, short descriptions and thumbnails will appear here.</p></div>';
    listEl.appendChild(empty);
    return;
  }
  items.forEach(({ label, text, thumb }) => {
    const li = document.createElement('li');
    li.className = 'desc-item';
    const imgHtml = thumb ? `<img class="thumb" src="${thumb}" alt="${label}">` : '<div class="thumb" style="display:flex;align-items:center;justify-content:center;color:#4c5f86">N/A</div>';
    li.innerHTML = `
      ${imgHtml}
      <div>
        <p class="label">${label}</p>
        <p class="text">${text || '(No description yet)'} </p>
      </div>
    `;
    listEl.appendChild(li);
  });
}

async function poll() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    renderItems(data.items || []);
    pillEl.textContent = data.source ? data.source.toUpperCase() : 'LIVE';
    pillEl.style.opacity = '1';
    if (data.gemini) {
      geminiNote.style.display = 'none';
    } else {
      geminiNote.style.display = 'block';
      geminiNote.textContent = data.gemini_error
        ? `Gemini unavailable: ${data.gemini_error}`
        : 'Gemini descriptions disabled. Set GEMINI_API_KEY to enable.';
    }
  } catch (e) {
    pillEl.textContent = 'OFFLINE';
    pillEl.style.opacity = '0.6';
  } finally {
    setTimeout(poll, 1000);
  }
}

async function setSource(type, path) {
  await fetch('/set_source', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, path }),
  });
  // Refresh stream image
  const img = document.getElementById('video');
  img.src = '/video_feed?_=' + Date.now();
}

btnWebcam?.addEventListener('click', () => setSource('webcam'));
btnDroid?.addEventListener('click', () => setSource('droidcam'));

fileVideo?.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  try {
    await fetch('/upload_video', { method: 'POST', body: form });
    const img = document.getElementById('video');
    img.src = '/video_feed?_=' + Date.now();
  } catch (err) {
    console.error(err);
  } finally {
    e.target.value = '';
  }
});

poll();
