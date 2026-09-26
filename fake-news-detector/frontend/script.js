const newsInput = document.getElementById("newsInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const sampleRealBtn = document.getElementById("sampleRealBtn");
const sampleFakeBtn = document.getElementById("sampleFakeBtn");
const resultCard = document.getElementById("resultCard");
const errorCard = document.getElementById("errorCard");
const verdictBadge = document.getElementById("verdictBadge");
const confidenceText = document.getElementById("confidenceText");
const barFake = document.getElementById("barFake");
const flagsList = document.getElementById("flagsList");

// Same-origin by default (Flask serves the frontend). Override if hosting separately.
const API_URL = "/api/predict";

const SAMPLE_REAL =
  "The central bank kept interest rates unchanged at its policy meeting on Thursday, citing stable inflation figures. Officials said further details would be shared in a follow-up briefing.";
const SAMPLE_FAKE =
  "SHOCKING: You won't believe what the government is hiding about vaccines!!! Share this before it gets DELETED!!!";

newsInput.addEventListener("input", () => {
  charCount.textContent = `${newsInput.value.length} characters`;
});

sampleRealBtn.addEventListener("click", () => {
  newsInput.value = SAMPLE_REAL;
  charCount.textContent = `${newsInput.value.length} characters`;
});

sampleFakeBtn.addEventListener("click", () => {
  newsInput.value = SAMPLE_FAKE;
  charCount.textContent = `${newsInput.value.length} characters`;
});

analyzeBtn.addEventListener("click", analyze);

async function analyze() {
  const text = newsInput.value.trim();
  hide(errorCard);
  hide(resultCard);

  if (text.length < 15) {
    showError("Please enter at least 15 characters of text to analyze.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong. Please try again.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Could not reach the backend API. Is the Flask server running?");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
}

function renderResult(data) {
  show(resultCard);
  verdictBadge.textContent = data.label === "FAKE" ? "⚠ LIKELY FAKE" : "✓ LIKELY REAL";
  verdictBadge.className = "badge " + (data.label === "FAKE" ? "fake" : "real");
  confidenceText.textContent = `${data.confidence}% confidence`;

  barFake.style.width = `${data.probabilities.fake}%`;

  flagsList.innerHTML = "";
  if (data.flags && data.flags.length > 0) {
    data.flags.forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f;
      flagsList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "No obvious sensationalism signals detected.";
    li.className = "none";
    flagsList.appendChild(li);
  }
}

function showError(msg) {
  errorCard.textContent = msg;
  show(errorCard);
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }
