"""Client-side operation player: playback never blocks the Streamlit script."""
from __future__ import annotations

import json

from image_processing import OperationResult


def operation_html(result: OperationResult, show_values: bool, rescale: bool) -> str:
    data = json.dumps({
        "operation": result.operation,
        "input": result.incoming.tolist(),
        "output": result.output.tolist(),
        "bounds": result.original_bounds,
        "size": result.window_size,
        "stride": result.stride,
        "kernel": result.kernel.tolist() if result.kernel is not None else None,
        "showValues": show_values,
        "rescale": rescale,
    })
    return PLAYER_HTML.replace("__OPERATION_DATA__", data)


PLAYER_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; }
body { margin: 0; padding: 14px; background: #f5f8fc; color: #17324d;
       font: 15px system-ui, sans-serif; border-radius: 12px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
button { padding: 10px 18px; border: 1px solid #7699bb; border-radius: 7px;
         background: white; color: #17324d; cursor: pointer; font: inherit; }
button:disabled { opacity: .5; cursor: default; }
button:focus-visible, input:focus-visible { outline: 3px solid #1877d2; }
#play { background: #1762a5; color: white; }
.grids { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
canvas { width: 100%; aspect-ratio: 1; display: block; }
h3 { font-size: 16px; margin: 14px 0 8px; }
p { line-height: 1.5; margin: 10px 0; }
.timeline { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
#position { flex: 1; min-width: 100px; }
#calculation { white-space: pre-wrap; overflow-wrap: anywhere; min-height: 48px; }
.hint { color: #48627c; font-size: 13px; }
@media (max-width: 600px) { .grids { gap: 8px; } body { padding: 8px; } }
</style></head>
<body>
<div class="controls">
  <button id="play" type="button">Play</button>
  <button id="pause" type="button" disabled>Pause</button>
  <button id="restart" type="button">Restart</button>
  <button id="next" type="button">Next pixel</button>
  <label for="speed">Pixels / second</label>
  <input id="speed" type="range" min="1" max="20" value="4">
  <span id="speed-value">4</span>
</div>
<div class="timeline">
  <label for="position">Built pixels</label><input id="position" type="range" min="0" value="0">
  <span id="progress" role="status" aria-live="off"></span>
</div>
<div class="grids">
  <div><h3>Input / moving window <span id="input-shape"></span></h3><canvas id="input" aria-label="Input with active operation window"></canvas></div>
  <div><h3>Output / built one pixel at a time <span id="output-shape"></span></h3><canvas id="output" aria-label="Partially constructed output"></canvas></div>
</div>
<p class="hint">Orange cells = added zero padding; red outline = active window / output pixel.
Blank output cells have not been computed yet. Scan order: left to right, then top to bottom.</p>
<p class="hint">Numeric labels appear when cells are large enough; the current window's values are always shown below.</p>
<p id="display-mode" class="hint"></p>
<p id="calculation" role="status" aria-live="off"></p>
<script>
const data = __OPERATION_DATA__;
const $ = id => document.getElementById(id);
const rows = data.output.length, cols = data.output[0].length, total = rows * cols;
const flatInput = data.input.flat(), flatOutput = data.output.flat();
const inMin = Math.min(...flatInput), inMax = Math.max(...flatInput);
const outMin = Math.min(...flatOutput), outMax = Math.max(...flatOutput);
let completed = 0, running = false, timer = null;
$("position").max = total;
$("input-shape").textContent = `(${data.input.length} × ${data.input[0].length})`;
$("output-shape").textContent = `(${rows} × ${cols})`;
$("display-mode").textContent = data.rescale
  ? "Output brightness is rescaled using the full output range; numeric values and calculations are unchanged."
  : "Brightness uses 0–255 (clipped); numeric values and calculations are unchanged.";
const fmt = value => Number.isInteger(value) ? String(value) : Number(value.toPrecision(7)).toString();

function canvasGrid(id, matrix, count, input) {
  const canvas = $(id), nrows = matrix.length, ncols = matrix[0].length;
  const side = Math.max(220, canvas.getBoundingClientRect().width);
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(side * scale); canvas.height = Math.round(side * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const cell = side / Math.max(nrows, ncols);
  const [top, left, height, width] = data.bounds;
  for (let r = 0; r < nrows; r++) {
    for (let c = 0; c < ncols; c++) {
      const known = input || r * ncols + c < count, value = matrix[r][c];
      let shade = value;
      if (!input && data.rescale && outMax > outMin) shade = (value - outMin) / (outMax - outMin) * 255;
      shade = Math.max(0, Math.min(255, shade));
      ctx.fillStyle = known ? `rgb(${shade},${shade},${shade})` : "#e4eaf1";
      ctx.fillRect(c * cell, r * cell, cell, cell);
      if (input && (r < top || r >= top + height || c < left || c >= left + width)) {
        ctx.fillStyle = "rgba(255,166,45,.55)";
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
      ctx.strokeStyle = "#7891a9"; ctx.lineWidth = .5;
      ctx.strokeRect(c * cell, r * cell, cell, cell);
      if (known && data.showValues && cell >= 17) {
        ctx.fillStyle = shade > 140 ? "#111" : "#fff";
        ctx.font = `${Math.max(7, Math.min(12, cell / 3.6))}px system-ui`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const label = Number.isInteger(value) ? String(value) : Number(value.toPrecision(3)).toString();
        ctx.fillText(label, (c + .5) * cell, (r + .5) * cell, cell - 2);
      }
    }
  }
  const active = Math.max(0, completed - 1);
  const r = Math.floor(active / cols), c = active % cols;
  ctx.strokeStyle = "#ff2848"; ctx.lineWidth = 3;
  const x = (input ? c * data.stride : c) * cell;
  const y = (input ? r * data.stride : r) * cell;
  const size = (input ? data.size : 1) * cell;
  ctx.strokeRect(x + 1.5, y + 1.5, size - 3, size - 3);
}

function draw() {
  canvasGrid("input", data.input, total, true);
  canvasGrid("output", data.output, completed, false);
  $("position").value = completed;
  $("progress").textContent = `${completed} / ${total}`;
  $("play").disabled = running;
  $("pause").disabled = !running;
  $("next").disabled = completed === total;
  const active = Math.max(0, completed - 1);
  const row = Math.floor(active / cols), col = active % cols;
  const y = row * data.stride, x = col * data.stride;
  const patch = [];
  for (let r = 0; r < data.size; r++)
    for (let c = 0; c < data.size; c++) patch.push(data.input[y + r][x + c]);
  let formula;
  if (data.kernel) {
    formula = patch.map((value, i) => `${fmt(value)} × ${fmt(data.kernel.flat()[i])}`).join(" + ");
  } else if (data.operation === "Normalize 0–1") {
    formula = inMax === inMin ? "Constant input → 0" : `(${fmt(patch[0])} − ${fmt(inMin)}) / (${fmt(inMax)} − ${fmt(inMin)})`;
  } else if (data.operation.startsWith("Invert")) {
    formula = `255 − ${fmt(patch[0])}`;
  } else if (data.operation === "Downsampling (nearest neighbor)") {
    formula = `Keep input (${y}, ${x}): ${fmt(patch[0])}`;
  } else {
    const values = patch.map(fmt).join(", ");
    formula = data.operation === "Average pooling"
      ? `(${patch.map(fmt).join(" + ")}) / ${patch.length}`
      : `${data.operation === "Min pooling" ? "min" : "max"}(${values})`;
  }
  $("calculation").textContent = `${completed === 0 ? "Next" : "Just built"} output (${row}, ${col}): ${formula} = ${fmt(data.output[row][col])}`;
}

function pause() {
  running = false; clearTimeout(timer); timer = null;
}
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(() => {
    if (!running) return;
    completed++;
    if (completed === total) pause();
    draw();
    if (running) schedule();
  }, 1000 / Number($("speed").value));
}
$("play").onclick = () => {
  if (completed === total) completed = 0;
  running = true; draw(); schedule();
};
$("pause").onclick = () => { pause(); draw(); };
$("restart").onclick = () => { pause(); completed = 0; draw(); };
$("next").onclick = () => { pause(); completed = Math.min(total, completed + 1); draw(); };
$("position").oninput = event => { pause(); completed = Number(event.target.value); draw(); };
$("speed").oninput = () => {
  $("speed-value").textContent = $("speed").value;
  if (running) schedule();
};
window.addEventListener("resize", draw);
window.addEventListener("pagehide", pause);
draw();
</script></body></html>
"""
