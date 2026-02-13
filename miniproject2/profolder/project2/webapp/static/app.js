// ================================
// DOM REFERENCES
// ================================
const listEl = document.getElementById("desc-list");
const sceneSummaryEl = document.getElementById("scene-summary");
const fpsEl = document.getElementById("fps-pill");
const pillEl = document.getElementById("status-pill");
const geminiErrEl = document.getElementById("gemini-error");

const btnWebcam = document.getElementById("btn-webcam");
const btnDroid = document.getElementById("btn-droid");
const fileVideo = document.getElementById("file-video");


// ================================
// RENDER OBJECT LIST
// ================================
function renderItems(items) {
    listEl.innerHTML = "";

    if (!items || items.length === 0) {
        listEl.innerHTML = `<li class="desc-card">
            <div style="padding:6px;color:#667">
                No objects detected yet…
            </div>
        </li>`;
        return;
    }

    items.forEach(({ label, text, thumb }) => {
        const li = document.createElement("li");
        li.className = "desc-card";

        li.innerHTML = `
            <img class="desc-thumb" src="${thumb || ""}" alt="${label}" />
            <div class="info">
                <p class="label">${label}</p>
                <p class="text">${text || "(No description available)"}</p>
            </div>
        `;
        listEl.appendChild(li);
    });
}


// ================================
// POLL STATUS FROM SERVER
// ================================
async function pollStatus() {
    try {
        const res = await fetch("/status");
        const data = await res.json();

        // Object list
        renderItems(data.items || []);

        // Source Pill
        pillEl.textContent = data.source ? data.source.toUpperCase() : "NONE";

        // FPS indicator
        fpsEl.textContent = `FPS: ${data.fps?.toFixed(1) || "--"}`;

        // Scene summary (Gemini full-frame analysis)
        if (data.items.length > 0) {
            // Use the description of the first detected label (they all share the same Gemini summary)
            sceneSummaryEl.textContent = data.items[0].text;
        } else {
            sceneSummaryEl.textContent = "Waiting for scene analysis…";
        }

        // Gemini error message
        if (data.gemini_error) {
            geminiErrEl.style.display = "block";
            geminiErrEl.textContent = "Gemini Error: " + data.gemini_error;
        } else if (!data.gemini) {
            geminiErrEl.style.display = "block";
            geminiErrEl.textContent = "Gemini disabled — set GEMINI_API_KEY.";
        } else {
            geminiErrEl.style.display = "none";
        }

    } catch (err) {
        console.error("Polling error:", err);
        pillEl.textContent = "OFFLINE";
    }

    setTimeout(pollStatus, 900);
}


// ================================
// CHANGE SOURCE
// ================================
async function setSource(type, path = null) {
    await fetch("/set_source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, path })
    });

    // Reset video stream to avoid caching
    const img = document.getElementById("video");
    img.src = "/video_feed?_=" + Date.now();
}


// ================================
// EVENT HANDLERS
// ================================
btnWebcam?.addEventListener("click", () => setSource("webcam"));
btnDroid?.addEventListener("click", () => setSource("droidcam"));

fileVideo?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    try {
        await fetch("/upload_video", {
            method: "POST",
            body: form
        });

        // refresh stream
        const img = document.getElementById("video");
        img.src = "/video_feed?_=" + Date.now();

    } catch (err) {
        console.error("Upload error:", err);
    }

    e.target.value = ""; // reset input
});


// ================================
// START POLLING
// ================================
pollStatus();
