/**
 * Main Application Orchestrator for EE5130 MNIST Live Exhibition Demo.
 */

let drawingCanvas;
let currentTrueLabel = null;
let currentSamplePixels = null;
let currentActiveModel = "ENSEMBLE – FINAL ANSWER";
let broadcastChannel = null;

// Initialize BroadcastChannel for multi-tab sync with /clusters
try {
  broadcastChannel = new BroadcastChannel('mnist_demo_channel');
} catch (e) {
  console.log('BroadcastChannel not supported in this environment, using localStorage fallback');
}

document.addEventListener('DOMContentLoaded', async () => {
  drawingCanvas = new DrawingCanvas('drawCanvas');
  initFullscreen();
  await loadModelList();
  bindUIEvents();
  renderEmptyChart();
});

function initFullscreen() {
  const fsBtn = document.getElementById('fullscreenBtn');
  fsBtn.addEventListener('click', toggleFullscreen);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'F11') {
      e.preventDefault();
      toggleFullscreen();
    }
  });
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch((err) => console.log(err));
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
}

async function loadModelList() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const container = document.getElementById('modelListContainer');
    container.innerHTML = '';

    const models = data.models;
    // Always place ENSEMBLE first
    const modelKeys = Object.keys(models).sort((a, b) => {
      if (a.includes('ENSEMBLE')) return -1;
      if (b.includes('ENSEMBLE')) return 1;
      return (models[b].test_accuracy || 0) - (models[a].test_accuracy || 0);
    });

    modelKeys.forEach((key, idx) => {
      const m = models[key];
      const isEnsemble = key.includes('ENSEMBLE');
      const isChecked = isEnsemble || idx === 0;
      if (isChecked) currentActiveModel = key;

      const accStr = (m.test_accuracy !== undefined && m.test_accuracy !== null)
        ? `${(m.test_accuracy * 100).toFixed(2)}%`
        : 'N/A';

      const opt = document.createElement('label');
      opt.className = 'model-option';
      opt.innerHTML = `
        <input type="radio" name="activeModelRadio" value="${key}" ${isChecked ? 'checked' : ''}>
        <span class="model-name">${m.name || key}</span>
        <span class="model-acc">${accStr}</span>
      `;
      opt.querySelector('input').addEventListener('change', (e) => {
        currentActiveModel = e.target.value;
        document.getElementById('statusActiveModel').textContent = currentActiveModel;
        // If an image is already drawn or loaded on the canvas, run prediction immediately!
        if (drawingCanvas && (drawingCanvas.hasDrawn || currentSamplePixels)) {
          runInference();
        }
      });

      container.appendChild(opt);
    });

    document.getElementById('statusActiveModel').textContent = currentActiveModel;
  } catch (err) {
    console.error('Failed to load model registry:', err);
  }
}

function bindUIEvents() {
  // Predict Button
  document.getElementById('predictBtn').addEventListener('click', runInference);

  // Clear Button
  document.getElementById('clearBtn').addEventListener('click', () => {
    drawingCanvas.clear();
    currentTrueLabel = null;
    currentSamplePixels = null;
    resetInferenceView();
  });

  // Random Test Sample Button
  document.getElementById('randomSampleBtn').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/random_sample');
      const sample = await res.json();
      currentTrueLabel = sample.true_label;
      currentSamplePixels = sample.pixels;
      drawingCanvas.loadImageFromArray(sample.pixels);

      // Auto run prediction on test sample
      await runInference();
    } catch (err) {
      console.error('Error fetching random sample:', err);
    }
  });

  // Benchmark 10 Samples Button
  document.getElementById('benchmarkBtn').addEventListener('click', runBenchmark);

  // File Upload
  const uploadInput = document.getElementById('digitUploadInput');
  uploadInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB limit.');
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        drawingCanvas.loadImageFromDataURL(event.target.result);
        currentTrueLabel = null;
        currentSamplePixels = null;
      };
      reader.readAsDataURL(file);
    }
  });

  // Strategy Selector change
  const stratSelect = document.getElementById('combinationStrategySelect');
  if (stratSelect) {
    stratSelect.addEventListener('change', () => {
      if (drawingCanvas.hasDrawn || currentSamplePixels) {
        runInference();
      }
    });
  }

  // Close Modal
  document.getElementById('closeModalBtn').addEventListener('click', () => {
    document.getElementById('benchmarkModal').style.display = 'none';
  });
}

async function runInference() {
  if (!drawingCanvas.hasDrawn && !currentSamplePixels) {
    alert('Please draw a digit or click "RANDOM TEST SAMPLE" first.');
    return;
  }

  const payload = {
    model_name: currentActiveModel,
    true_label: currentTrueLabel,
    combination_method: document.getElementById('combinationStrategySelect').value,
  };

  if (currentSamplePixels) {
    payload.pixels = currentSamplePixels;
  } else {
    payload.image_b64 = drawingCanvas.getImageDataURL();
  }

  try {
    const t0 = performance.now();
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    const clientLatency = (performance.now() - t0).toFixed(1);

    updateInferenceUI(data);

    // Sync to external /clusters tab via BroadcastChannel & localStorage
    const syncData = {
      timestamp: Date.now(),
      pixels: currentSamplePixels,
      image_b64: data.tensor_b64,
      predicted_digit: data.digit,
      true_label: data.true_label,
      pca_2d: data.cluster_topology ? data.cluster_topology.pca_2d_coords : [0, 0],
      pca_3d: data.cluster_topology ? data.cluster_topology.pca_3d_coords : [0, 0, 0],
      cluster_id: data.cluster_topology ? data.cluster_topology.cluster_id : 0,

      distance: data.cluster_topology ? data.cluster_topology.distance_to_centroid : 0,
    };
    try {
      localStorage.setItem('active_mnist_sample', JSON.stringify(syncData));
      if (broadcastChannel) {
        broadcastChannel.postMessage(syncData);
      }
    } catch (e) {
      console.log('Sync storage note:', e);
    }
  } catch (err) {
    console.error('Inference request failed:', err);
  }
}

function updateInferenceUI(data) {
  // 1. Preprocessing Previews
  const cropImg = document.getElementById('cropBoxImg');
  const tensorImg = document.getElementById('tensorImg');
  cropImg.src = data.crop_b64;
  cropImg.style.display = 'block';
  tensorImg.src = data.tensor_b64;
  tensorImg.style.display = 'block';
  document.getElementById('tensorTitle').textContent = `2. 28x28 (Pred: ${data.digit})`;

  // 2. Status Box
  document.getElementById('statusActiveModel').textContent = data.method_name;
  document.getElementById('statusPredDigit').textContent = `${data.digit} (${data.confidence_percent})`;
  document.getElementById('statusLatency').textContent = `${data.inference_latency_ms} ms`;

  const outcomeElem = document.getElementById('statusOutcome');
  if (data.true_label !== null && data.true_label !== undefined) {
    const statusText = data.is_correct ? 'CORRECT' : 'WRONG';
    const statusClass = data.is_correct ? 'outcome-correct' : 'outcome-wrong';
    outcomeElem.innerHTML = `True: ${data.true_label} | <span class="${statusClass}">${statusText}</span>`;
  } else {
    outcomeElem.textContent = 'True: n/a (drawn digit)';
  }

  const top3Elem = document.getElementById('statusTop3');
  top3Elem.textContent = data.top3.map(t => `${t.digit}: ${t.percent}`).join(' | ');

  // 3. Render Output Probability Chart
  renderProbabilityChart(data.probabilities, data.digit, data.is_correct, data.inference_latency_ms);

  // 4. Update Model Comparison Table
  updateComparisonTable(data.model_breakdown, data.digit, data.most_confident_method, data.best_method_overall);
  document.getElementById('agreementSummaryText').textContent = data.agreement_summary || '';

  // 5. Update Unsupervised Cluster Profiler
  const topo = data.cluster_topology;
  if (topo) {
    document.getElementById('clusterAssigned').textContent = `Group #${topo.cluster_id} / ${topo.total_clusters}`;
    document.getElementById('clusterDominant').textContent = `Digit ${topo.dominant_digit} (${topo.purity_percent} pure)`;
    document.getElementById('clusterDist').textContent = `${topo.distance_to_centroid} (50-D)`;
    document.getElementById('clusterPlaceholderText').style.display = 'none';
    document.getElementById('clusterActiveDetails').style.display = 'flex';
  }
}

function renderEmptyChart() {
  const probs = Array(10).fill(0.0);
  renderProbabilityChart(probs, -1, null, 0);
}

function renderProbabilityChart(probs, predDigit, isCorrect, latencyMs) {
  const container = document.getElementById('probabilityChart');
  const digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  let barColors = digits.map(d => '#333333');
  if (predDigit >= 0) {
    let highlightColor = '#2979ff'; // blue
    if (isCorrect === true) highlightColor = '#00e676'; // green
    if (isCorrect === false) highlightColor = '#ff5252'; // red
    barColors[predDigit] = highlightColor;
  }

  if (window.Plotly) {
    const trace = {
      x: digits.map(d => `Digit ${d}`),
      y: probs,
      type: 'bar',
      marker: {
        color: barColors,
        line: { color: '#444444', width: 1 },
      },
    };

    const layout = {
      margin: { l: 30, r: 10, t: 15, b: 25 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis: {
        tickfont: { color: '#a0a0a0', size: 10, family: 'Consolas, monospace' },
        gridcolor: '#222222',
      },
      yaxis: {
        range: [0, 1.05],
        tickfont: { color: '#a0a0a0', size: 10, family: 'Consolas, monospace' },
        gridcolor: '#262626',
        dtick: 0.2,
      },
      autosize: true,
    };

    const config = { displayModeBar: false, responsive: true };
    Plotly.react(container, [trace], layout, config);
  } else {
    // Fallback simple HTML5 bars if Plotly bundle pending
    container.innerHTML = '<div style="display:flex;height:100%;align-items:flex-end;gap:4px;padding:10px;">' +
      probs.map((p, d) => `
        <div style="flex:1;text-align:center;">
          <div style="height:${p*120}px;background:${barColors[d]};border-radius:2px;"></div>
          <div style="font-size:10px;color:#aaa;margin-top:2px;">${d}</div>
        </div>
      `).join('') + '</div>';
  }

  document.getElementById('chartTitleLatency').textContent = `(${latencyMs} ms latency)`;
}

function updateComparisonTable(breakdown, finalDigit, mostConfident, bestOverall) {
  const tbody = document.getElementById('comparisonTableBody');
  tbody.innerHTML = '';

  if (!breakdown || breakdown.length === 0) return;

  breakdown.forEach((m) => {
    const tr = document.createElement('tr');
    const agrees = (m.digit === finalDigit);
    const agreeBadge = agrees
      ? `<span class="badge-agree">AGREE</span>`
      : `<span class="badge-disagree">DIFF (${m.digit})</span>`;

    let specialBadge = '';
    const isActive = (m.method_name === currentActiveModel);
    if (isActive) {
      specialBadge += ' <span style="color:#2979ff;font-weight:bold;">● [ACTIVE]</span>';
      tr.style.backgroundColor = 'rgba(41, 121, 255, 0.15)';
    }
    if (mostConfident && m.method_name === mostConfident.name) {
      specialBadge += ' ⚡ [Most Confident]';
    }
    if (bestOverall && m.method_name === bestOverall) {
      specialBadge += ' ★ [Best Overall]';
    }


    tr.innerHTML = `
      <td>${m.method_name}${specialBadge}</td>
      <td style="color:#888;">${m.family}</td>
      <td style="font-weight:bold;color:${agrees ? '#00e676' : '#fff'};">${m.digit}</td>
      <td>${(m.confidence * 100).toFixed(1)}%</td>
      <td style="color:#aaa;">${m.inference_time_ms.toFixed(2)} ms</td>
      <td>${agreeBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

function resetInferenceView() {
  document.getElementById('cropBoxImg').style.display = 'none';
  document.getElementById('tensorImg').style.display = 'none';
  document.getElementById('tensorTitle').textContent = '2. 28x28 Tensor';
  document.getElementById('statusPredDigit').textContent = '-';
  document.getElementById('statusLatency').textContent = '-';
  document.getElementById('statusOutcome').textContent = 'Draw digit or load sample';
  document.getElementById('statusTop3').textContent = '-';
  document.getElementById('clusterActiveDetails').style.display = 'none';
  document.getElementById('clusterPlaceholderText').style.display = 'block';
  document.getElementById('comparisonTableBody').innerHTML = '';
  document.getElementById('agreementSummaryText').textContent = '';
  renderEmptyChart();
}

async function runBenchmark() {
  const modal = document.getElementById('benchmarkModal');
  const grid = document.getElementById('benchmarkGrid');
  modal.style.display = 'flex';
  grid.innerHTML = '<div style="color:#fff;grid-column:1/6;padding:20px;text-align:center;">Executing live benchmark on 10 random samples...</div>';

  try {
    const payload = {
      model_name: currentActiveModel,
      combination_method: document.getElementById('combinationStrategySelect').value,
    };
    const res = await fetch('/api/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    document.getElementById('benchSummaryStats').textContent =
      `Evaluated: ${data.total_count} samples | Accuracy: ${data.accuracy_percent} (${data.correct_count}/${data.total_count}) | Avg Latency: ${data.average_latency_ms} ms/sample`;

    grid.innerHTML = '';
    data.benchmark_samples.forEach((s) => {
      const card = document.createElement('div');
      card.className = 'bench-card';
      const statusClass = s.is_correct ? 'bench-status-correct' : 'bench-status-wrong';
      const statusText = s.is_correct ? 'CORRECT' : 'WRONG';
      card.innerHTML = `
        <img src="${s.image_b64}" alt="sample">
        <div>True: <b>${s.true_label}</b></div>
        <div>Pred: <b>${s.predicted_digit}</b></div>
        <div style="color:#888;">${s.confidence_percent}</div>
        <div class="${statusClass}">${statusText}</div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<div style="color:#ff5252;grid-column:1/6;">Benchmark failed: ${err.message}</div>`;
  }
}

