/**
 * EE5130 Cluster Spread Explorer Orchestrator
 * Interactive Latent Space Visualization & Cross-Tab Live Sync
 */

let currentMethod = 'K-Means';
let currentProjection = 'PCA-3D';
let currentColorMode = 'cluster';
let clusterData = null;

let activeQuerySample = null;
let broadcastChannel = null;

// Palette for up to 15 clusters
const CLUSTER_COLORS = [
  '#2979ff', '#00e676', '#ff9100', '#ab47bc', '#00e5ff',
  '#f50057', '#ffd600', '#00b0ff', '#76ff03', '#ff3d00',
  '#8d6e63', '#90caf9', '#b39ddb', '#80cbc4', '#e0e0e0'
];

document.addEventListener('DOMContentLoaded', async () => {
  initLiveSync();
  bindControls();
  await loadClusterMethod(currentMethod);
});

function initLiveSync() {
  try {
    broadcastChannel = new BroadcastChannel('mnist_demo_channel');
    broadcastChannel.onmessage = (event) => {
      if (event.data) {
        handleLiveQuery(event.data);
      }
    };
  } catch (e) {
    console.log('BroadcastChannel unsupported, relying on storage events.');
  }

  // Cross-tab storage fallback
  window.addEventListener('storage', (e) => {
    if (e.key === 'active_mnist_sample' && e.newValue) {
      try {
        const query = JSON.parse(e.newValue);
        handleLiveQuery(query);
      } catch (err) {
        console.error('Storage parse error:', err);
      }
    }
  });

  // Check if a sample is already active from previous inference
  const existing = localStorage.getItem('active_mnist_sample');
  if (existing) {
    try {
      activeQuerySample = JSON.parse(existing);
      document.getElementById('liveQueryTag').style.display = 'inline';
    } catch (e) {}
  }
}

function handleLiveQuery(query) {
  activeQuerySample = query;
  document.getElementById('liveQueryTag').style.display = 'inline';
  document.getElementById('liveQueryTag').textContent =
    `★ Live Query Active (Pred: ${query.predicted_digit}, Cluster: #${query.cluster_id})`;
  renderScatter();
}

function bindControls() {
  document.getElementById('clusterMethodSelect').addEventListener('change', async (e) => {
    currentMethod = e.target.value;
    await loadClusterMethod(currentMethod);
  });

  document.getElementById('clusterProjectionSelect').addEventListener('change', (e) => {
    currentProjection = e.target.value;
    renderScatter();
  });

  document.getElementById('clusterColorSelect').addEventListener('change', (e) => {
    currentColorMode = e.target.value;
    renderScatter();
  });
}

async function loadClusterMethod(method) {
  try {
    const res = await fetch(`/api/clusters/${encodeURIComponent(method)}/spread`);
    clusterData = await res.json();

    updateMetrics(clusterData.cluster_data.metrics);
    renderScatter();
    renderSpreadPlot();
    renderClusterSizes();
    renderContingencyHeatmap();
    renderGallery();
  } catch (err) {
    console.error(`Failed to load cluster data for ${method}:`, err);
  }
}

function updateMetrics(metrics) {
  if (!metrics) return;

  const sil = metrics.silhouette !== undefined ? metrics.silhouette.toFixed(3) : 'N/A';
  const db = metrics.davies_bouldin !== undefined ? metrics.davies_bouldin.toFixed(3) : 'N/A';
  const ch = metrics.calinski_harabasz !== undefined ? metrics.calinski_harabasz.toFixed(1) : 'N/A';
  const ari = metrics.ari !== undefined ? metrics.ari.toFixed(3) : 'N/A';
  const nmi = metrics.nmi !== undefined ? metrics.nmi.toFixed(3) : 'N/A';
  const pur = metrics.purity !== undefined ? `${(metrics.purity * 100).toFixed(1)}%` : 'N/A';
  const timeMs = metrics.runtime_s !== undefined ? `${(metrics.runtime_s * 1000).toFixed(1)} ms` : '';

  document.getElementById('metricSilhouette').textContent = sil;
  document.getElementById('metricDaviesBouldin').textContent = db;
  document.getElementById('metricCalinski').textContent = ch;
  document.getElementById('metricARI').textContent = ari;
  document.getElementById('metricNMI').textContent = nmi;
  document.getElementById('metricPurity').textContent = pur;
  document.getElementById('metricRuntime').textContent = `Fit time: ${timeMs}`;
}

function renderScatter() {
  const container = document.getElementById('scatterContainer');
  if (!clusterData || !window.Plotly) return;

  const isDual = (currentProjection === 'Side-by-Side');
  const is3D = currentProjection.includes('3D');
  const labels = clusterData.cluster_data.labels;
  const trueLabels = clusterData.true_labels;
  const nPoints = clusterData.reference_points_count;

  // Choose color grouping
  const groupArr = (currentColorMode === 'cluster') ? labels : trueLabels;
  const colorLegendTitle = (currentColorMode === 'cluster') ? 'Cluster' : 'True Digit';

  const hoverTexts = [];
  for (let i = 0; i < nPoints; i++) {
    hoverTexts.push(`Sample #${i}<br>True Digit: ${trueLabels[i]}<br>Assigned Cluster: ${labels[i]}`);
  }

  const projList = isDual ? ['PCA-2D', 't-SNE'] : [currentProjection];
  const traces = [];

  projList.forEach((projName, pIdx) => {
    const coords = clusterData.projections[projName] || clusterData.projections['PCA-3D'];
    const xVals = coords.map(p => p[0]);
    const yVals = coords.map(p => p[1]);
    const zVals = is3D ? coords.map(p => p[2] || 0) : null;

    if (is3D) {
      // 3D Point Cloud Trace
      const pointTrace3D = {
        x: xVals,
        y: yVals,
        z: zVals,
        mode: 'markers',
        type: 'scatter3d',
        name: `${projName} Samples`,
        hoverinfo: 'text',
        text: hoverTexts,
        marker: {
          size: 3.5,
          color: groupArr,
          colorscale: 'Portland',
          opacity: 0.85,
          line: { width: 0.5, color: '#111' },
        },
      };
      traces.push(pointTrace3D);

      // Centroids in 3D
      const centroids3D = (clusterData.cluster_data.centroids_3d && clusterData.cluster_data.centroids_3d[projName])
        || (clusterData.cluster_data.centroids && clusterData.cluster_data.centroids[projName]);
      if (centroids3D && centroids3D.length > 0) {
        const cTrace3D = {
          x: centroids3D.map(c => c[0]),
          y: centroids3D.map(c => c[1]),
          z: centroids3D.map(c => c[2]),
          mode: 'markers+text',
          type: 'scatter3d',
          name: `${projName} Centroids`,
          text: centroids3D.map((_, idx) => `C${idx}`),
          textposition: 'top center',
          textfont: { color: '#ffffff', size: 10, family: 'Consolas' },
          marker: {
            size: 7,
            symbol: 'diamond',
            color: '#ffffff',
            line: { width: 1.5, color: '#000' },
          },
        };
        traces.push(cTrace3D);
      }

      // Active Query Sample Marker in 3D (Gold Star / Diamond)
      if (activeQuerySample && (activeQuerySample.pca_3d || activeQuerySample.pca_2d)) {
        const qCoords = activeQuerySample.pca_3d || [activeQuerySample.pca_2d[0], activeQuerySample.pca_2d[1], 0.0];
        const qTrace3D = {
          x: [qCoords[0]],
          y: [qCoords[1]],
          z: [qCoords[2]],
          mode: 'markers+text',
          type: 'scatter3d',
          name: 'Live Query Digit',
          text: [`★ Pred: ${activeQuerySample.predicted_digit}`],
          textposition: 'top center',
          textfont: { color: '#ffd700', size: 13, family: 'Consolas' },
          marker: {
            size: 11,
            symbol: 'diamond',
            color: '#ffd700',
            line: { width: 2.5, color: '#ffffff' },
          },
        };
        traces.push(qTrace3D);
      }
    } else {
      // 2D Scatter Trace
      const pointTrace = {
        x: xVals,
        y: yVals,
        mode: 'markers',
        type: 'scatter',
        name: `${projName} Data`,
        hoverinfo: 'text',
        text: hoverTexts,
        marker: {
          size: 5,
          color: groupArr,
          colorscale: 'Portland',
          opacity: 0.75,
          line: { width: 0.5, color: '#111' },
        },
      };

      if (isDual) {
        pointTrace.xaxis = (pIdx === 0) ? 'x' : 'x2';
        pointTrace.yaxis = (pIdx === 0) ? 'y' : 'y2';
      }
      traces.push(pointTrace);

      // Centroids in 2D
      const centroids = clusterData.cluster_data.centroids_2d ? clusterData.cluster_data.centroids_2d[projName] : null;
      if (centroids && centroids.length > 0) {
        const cTrace = {
          x: centroids.map(c => c[0]),
          y: centroids.map(c => c[1]),
          mode: 'markers+text',
          type: 'scatter',
          name: `${projName} Centroids`,
          text: centroids.map((_, idx) => `C${idx}`),
          textposition: 'top center',
          textfont: { color: '#ffffff', size: 10, family: 'Consolas' },
          marker: {
            size: 11,
            symbol: 'diamond',
            color: '#ffffff',
            line: { width: 1.5, color: '#000' },
          },
        };
        if (isDual) {
          cTrace.xaxis = (pIdx === 0) ? 'x' : 'x2';
          cTrace.yaxis = (pIdx === 0) ? 'y' : 'y2';
        }
        traces.push(cTrace);
      }

      // Active Query Sample Marker in 2D (Gold Star)
      if (activeQuerySample && activeQuerySample.pca_2d && (projName === 'PCA-2D' || isDual)) {
        const qTrace = {
          x: [activeQuerySample.pca_2d[0]],
          y: [activeQuerySample.pca_2d[1]],
          mode: 'markers+text',
          type: 'scatter',
          name: 'Live Query Digit',
          text: [`★ Pred: ${activeQuerySample.predicted_digit}`],
          textposition: 'bottom center',
          textfont: { color: '#ffd700', size: 12, family: 'Consolas' },
          marker: {
            size: 16,
            symbol: 'star',
            color: '#ffd700',
            line: { width: 2, color: '#ffffff' },
          },
        };
        if (isDual) {
          qTrace.xaxis = (pIdx === 0) ? 'x' : 'x2';
          qTrace.yaxis = (pIdx === 0) ? 'y' : 'y2';
        }
        traces.push(qTrace);
      }
    }
  });

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: is3D ? { l: 0, r: 0, t: 10, b: 10 } : { l: 40, r: 20, t: 25, b: 35 },
    showlegend: !isDual,
    legend: {
      font: { color: '#bbb', size: 10, family: 'Consolas' },
      bgcolor: 'rgba(20,20,20,0.8)',
    },
    autosize: true,
  };

  if (is3D) {
    layout.scene = {
      xaxis: {
        title: `${currentProjection} Dim 1`,
        backgroundcolor: '#141414',
        gridcolor: '#282828',
        showbackground: true,
        tickfont: { color: '#888', size: 9 },
        titlefont: { color: '#aaa', size: 10 },
      },
      yaxis: {
        title: `${currentProjection} Dim 2`,
        backgroundcolor: '#141414',
        gridcolor: '#282828',
        showbackground: true,
        tickfont: { color: '#888', size: 9 },
        titlefont: { color: '#aaa', size: 10 },
      },
      zaxis: {
        title: `${currentProjection} Dim 3`,
        backgroundcolor: '#141414',
        gridcolor: '#282828',
        showbackground: true,
        tickfont: { color: '#888', size: 9 },
        titlefont: { color: '#aaa', size: 10 },
      },
      camera: {
        eye: { x: 1.45, y: 1.45, z: 1.15 },
      },
    };
  } else {
    layout.xaxis = {
      title: isDual ? 'PCA Component 1' : `${currentProjection} Dimension 1`,
      tickfont: { color: '#888', size: 9 },
      titlefont: { color: '#aaa', size: 10 },
      gridcolor: '#222',
    };
    layout.yaxis = {
      title: isDual ? 'PCA Component 2' : `${currentProjection} Dimension 2`,
      tickfont: { color: '#888', size: 9 },
      titlefont: { color: '#aaa', size: 10 },
      gridcolor: '#222',
    };

    if (isDual) {
      layout.grid = { rows: 1, columns: 2, pattern: 'independent' };
      layout.xaxis2 = {
        title: 't-SNE Dimension 1',
        tickfont: { color: '#888', size: 9 },
        titlefont: { color: '#aaa', size: 10 },
        gridcolor: '#222',
      };
      layout.yaxis2 = {
        title: 't-SNE Dimension 2',
        tickfont: { color: '#888', size: 9 },
        titlefont: { color: '#aaa', size: 10 },
        gridcolor: '#222',
      };
    }
  }

  const config = { responsive: true, displayModeBar: true, displaylogo: false };
  Plotly.react(container, traces, layout, config);
  const dimLabel = is3D ? '3D Interactive Manifold' : '2D Manifold';
  document.getElementById('scatterTitle').textContent = `${dimLabel} (${currentProjection}) – Color: ${colorLegendTitle}`;
}


function renderSpreadPlot() {
  const container = document.getElementById('spreadPlotContainer');
  if (!clusterData || !window.Plotly) return;

  const spreads = clusterData.cluster_data.cluster_spreads;
  if (!spreads || Object.keys(spreads).length === 0) {
    container.innerHTML = '<div style="color:#888;padding:40px;text-align:center;">Centroid distance spread metric not applicable for this method.</div>';
    return;
  }

  const traces = [];
  const clusterKeys = Object.keys(spreads).sort((a, b) => parseInt(a) - parseInt(b));

  clusterKeys.forEach((key, idx) => {
    const s = spreads[key];
    const trace = {
      type: 'box',
      name: `C${key}`,
      q1: [s.p25],
      median: [s.median_dist],
      q3: [s.p75],
      lowerfence: [s.min_dist],
      upperfence: [s.max_dist],
      mean: [s.mean_dist],
      marker: { color: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] },
      boxpoints: false,
    };
    traces.push(trace);
  });

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 40, r: 10, t: 20, b: 35 },
    xaxis: {
      title: 'Assigned Cluster ID',
      tickfont: { color: '#aaa', size: 10 },
      titlefont: { color: '#888', size: 10 },
      gridcolor: '#222',
    },
    yaxis: {
      title: 'Distance to Centroid (50-D Euclidean)',
      tickfont: { color: '#aaa', size: 10 },
      titlefont: { color: '#888', size: 10 },
      gridcolor: '#262626',
    },
    showlegend: false,
    autosize: true,
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, traces, layout, config);
}

function renderClusterSizes() {
  const container = document.getElementById('clusterSizesContainer');
  if (!clusterData || !window.Plotly) return;

  const labels = clusterData.cluster_data.labels;
  const counts = {};
  labels.forEach(l => {
    counts[l] = (counts[l] || 0) + 1;
  });

  const sortedClusters = Object.keys(counts).sort((a, b) => parseInt(a) - parseInt(b));
  const xLabels = sortedClusters.map(c => (parseInt(c) === -1 ? 'Noise (-1)' : `Cluster ${c}`));
  const yCounts = sortedClusters.map(c => counts[c]);

  const trace = {
    x: xLabels,
    y: yCounts,
    type: 'bar',
    marker: {
      color: sortedClusters.map((c, i) => parseInt(c) === -1 ? '#777777' : CLUSTER_COLORS[i % CLUSTER_COLORS.length]),
      line: { color: '#333333', width: 1 },
    },
    text: yCounts.map(cnt => `${cnt}`),
    textposition: 'auto',
    textfont: { color: '#ffffff', size: 10 },
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 40, r: 10, t: 20, b: 40 },
    xaxis: {
      tickfont: { color: '#aaa', size: 10 },
      gridcolor: '#222',
    },
    yaxis: {
      title: 'Point Count',
      tickfont: { color: '#aaa', size: 10 },
      titlefont: { color: '#888', size: 10 },
      gridcolor: '#262626',
    },
    autosize: true,
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, [trace], layout, config);
}

function renderContingencyHeatmap() {
  const container = document.getElementById('contingencyHeatmapContainer');
  if (!clusterData || !window.Plotly) return;

  const matrix = clusterData.cluster_data.heatmap;
  if (!matrix || matrix.length === 0) return;

  const zValues = (matrix[0] && matrix[0].counts) ? matrix.map(r => r.counts) : matrix;
  const nRows = zValues.length;
  const yLabels = Array.from({ length: nRows }, (_, i) => `Cluster ${i}`);
  const xLabels = Array.from({ length: 10 }, (_, i) => `Digit ${i}`);

  const trace = {
    z: zValues,
    x: xLabels,
    y: yLabels,

    type: 'heatmap',
    colorscale: 'Viridis',
    showscale: true,
    colorbar: {
      tickfont: { color: '#aaa', size: 9 },
      thickness: 10,
    },
  };

  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { l: 65, r: 20, t: 20, b: 35 },
    xaxis: {
      tickfont: { color: '#aaa', size: 10 },
    },
    yaxis: {
      tickfont: { color: '#aaa', size: 10 },
      autorange: 'reversed',
    },
    autosize: true,
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react(container, [trace], layout, config);
}

function renderGallery() {
  const container = document.getElementById('clusterGalleryGrid');
  if (!clusterData) return;

  const repSamples = clusterData.cluster_data.representative_samples || {};
  const mixedList = clusterData.cluster_data.mixed_clusters || [];
  const thumbs = clusterData.thumbnails || [];
  const mixedMap = {};
  mixedList.forEach(m => {
    mixedMap[m.cluster] = m;
  });

  container.innerHTML = '';
  const clusterKeys = Object.keys(repSamples).sort((a, b) => parseInt(a) - parseInt(b));

  clusterKeys.forEach((key, idx) => {
    const card = document.createElement('div');
    card.className = 'cluster-card';

    const mixed = mixedMap[parseInt(key)];
    const purityText = mixed ? `${(mixed.purity * 100).toFixed(0)}% purity` : 'Pure';
    const dominantDigit = mixed ? `Digit ${mixed.dominant_digit}` : `Cluster ${key}`;

    let confusionHtml = '';
    if (mixed && mixed.top_confusions && mixed.top_confusions.length > 1) {
      const top2 = mixed.top_confusions.slice(0, 3).map(c => `D${c.digit}:${c.count}`).join(', ');
      confusionHtml = `<div class="confusion-tag">Top Digits: ${top2}</div>`;
    }

    const sampleIndices = repSamples[key] || [];
    const thumbImgs = sampleIndices
      .map(sIdx => {
        const b64 = thumbs[sIdx];
        return b64 ? `<img src="${b64}" class="cluster-thumb" alt="sample" title="Index: ${sIdx}">` : '';
      })
      .join('');

    card.innerHTML = `
      <div class="cluster-card-header">
        <span style="color:${CLUSTER_COLORS[idx % CLUSTER_COLORS.length]};">Cluster #${key} (${dominantDigit})</span>
        <span style="color:#aaa;">${purityText}</span>
      </div>
      <div class="cluster-thumbnails-row">
        ${thumbImgs}
      </div>
      ${confusionHtml}
    `;
    container.appendChild(card);
  });
}

