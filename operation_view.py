"""Client-side operation player: playback never blocks the Streamlit script."""
from __future__ import annotations

import json

from image_processing import KERNELS, OperationResult, Stage, pad_window


def operation_html(result: OperationResult, show_values: bool, rescale: bool) -> str:
    return player_html([operation_data(result)], [], 0, show_values, rescale)


def operation_data(result: OperationResult) -> dict:
    return {
        "operation": result.operation,
        "input": result.incoming.tolist(),
        "output": result.output.tolist(),
        "bounds": result.original_bounds,
        "size": result.window_size,
        "stride": result.stride,
        "kernel": result.kernel.tolist() if result.kernel is not None else None,
    }


def network_html(
    stages: list[Stage], show_values: bool, rescale: bool, selected_id: str | None = None,
) -> str:
    frames = []
    layers = []
    selected = 0
    for index, stage in enumerate(stages):
        layer = stage.layer
        start = len(frames)
        operations = []
        if layer.preprocess != "None":
            operations.append(("Preprocessing", OperationResult(
                layer.preprocess, stage.incoming, stage.preprocessed,
                (0, 0, *stage.incoming.shape), 1, 1,
            )))
        kernel = KERNELS[layer.kernel]
        _, bounds = pad_window(stage.preprocessed, kernel.shape, layer.stride, layer.padding)
        operations.append(("Convolution", OperationResult(
            layer.kernel, stage.padded, stage.convolution, bounds,
            kernel.shape[0], layer.stride, kernel,
        )))
        if stage.convolution.size and layer.activation != "None":
            operations.append(("Activation", OperationResult(
                layer.activation, stage.convolution, stage.activated,
                (0, 0, *stage.convolution.shape), 1, 1,
            )))
        if stage.activated.size and layer.pooling != "None":
            operations.append(("Pooling", OperationResult(
                layer.pooling, stage.activated, stage.output,
                (0, 0, *stage.activated.shape), layer.pool_size, layer.pool_stride,
            )))
        for step, result in operations:
            frame = operation_data(result)
            frame.update(layer=index, step=step)
            if result.output.size == 0:
                frame["error"] = (
                    f"Stopped at layer {index + 1} ({layer.name}): {step.lower()} has no valid "
                    "output positions. Use SAME convolution padding, a smaller pooling window, "
                    "or a larger input. Later layers have not run."
                )
            frames.append(frame)
        layers.append({"name": layer.name or "Untitled", "start": start, "end": len(frames) - 1})
        if layer.id == selected_id:
            selected = index
    return player_html(frames, layers, selected, show_values, rescale)


def player_html(
    frames: list[dict], layers: list[dict], selected: int, show_values: bool, rescale: bool,
) -> str:
    if not frames:
        raise ValueError("Playback needs at least one operation.")
    data = json.dumps({
        "frames": frames, "layers": layers, "selectedLayer": selected,
        "showValues": show_values, "rescale": rescale,
    }, allow_nan=False).replace("<", "\\u003c")
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
.layers { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
.layers button { padding: 7px 10px; font-size: 13px; }
.layers button[aria-current="step"] { background: #dceeff; border: 2px solid #1762a5; }
#network-controls[hidden] { display: none; }
#stage-title { margin: 12px 0 4px; }
#handoff { min-height: 24px; }
#error { color: #a32323; font-weight: 600; }
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
<div id="network-controls" hidden>
  <div class="controls">
    <label><input type="radio" name="mode" id="whole-network" checked> Entire network</label>
    <label><input type="radio" name="mode" id="one-layer"> One layer</label>
    <button id="restart-layer" type="button">Restart selected layer</button>
  </div>
  <p class="hint">Click a layer to restart there. Entire network continues through later layers;
  One layer stops after that layer. Earlier layers are already calculated when you jump ahead.</p>
  <div id="layers" class="layers" aria-label="Playback layers"></div>
</div>
<h3 id="stage-title"></h3>
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
<p id="handoff" class="hint" role="status"></p>
<p id="error" role="alert"></p>
<p id="calculation" role="status" aria-live="off"></p>
<script>
const lesson = __OPERATION_DATA__;
const $ = id => document.getElementById(id);
const frames = lesson.frames;
let frameIndex = 0, endIndex = frames.length - 1, selectedLayer = lesson.selectedLayer;
let data, rows, cols, total, inMin, inMax, outMin, outMax;
let completed = 0, running = false, timer = null, handoff = "";
const layerButtons = [];
const fmt = value => Number.isInteger(value) ? String(value) : Number(value.toPrecision(7)).toString();

function loadFrame(index) {
  data = frames[index]; frameIndex = index; completed = 0;
  rows = data.output.length; cols = data.output[0]?.length || 0; total = rows * cols;
  const flatInput = data.input.flat(), flatOutput = data.output.flat();
  inMin = Math.min(...flatInput); inMax = Math.max(...flatInput);
  outMin = Math.min(...flatOutput); outMax = Math.max(...flatOutput);
  $("position").max = total;
  $("input-shape").textContent = `(${data.input.length} × ${data.input[0]?.length || 0})`;
  $("output-shape").textContent = `(${rows} × ${cols})`;
  $("stage-title").textContent = lesson.layers.length
    ? `Layer ${data.layer + 1}: ${lesson.layers[data.layer].name} · ${data.step} · ${data.operation}`
    : data.operation;
  $("error").textContent = data.error || "";
  $("display-mode").textContent = lesson.rescale
    ? `${lesson.layers.length ? "Both grids use their own" : "Output brightness uses its"} full range for display contrast; numeric values and calculations are unchanged.`
    : "Brightness uses 0–255 (clipped); numeric values and calculations are unchanged.";
  layerButtons.forEach((button, index) => {
    button.setAttribute("aria-current", index === data.layer ? "step" : "false");
    button.setAttribute("aria-pressed", String(index === selectedLayer));
  });
}

function canvasGrid(id, matrix, count, input) {
  const canvas = $(id), nrows = matrix.length, ncols = matrix[0]?.length || 0;
  const side = Math.max(1, canvas.getBoundingClientRect().width);
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(side * scale); canvas.height = Math.round(side * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  if (!nrows || !ncols) {
    ctx.fillStyle = "#e4eaf1"; ctx.fillRect(0, 0, side, side);
    return;
  }
  const cell = side / Math.max(nrows, ncols);
  const [top, left, height, width] = data.bounds;
  for (let r = 0; r < nrows; r++) {
    for (let c = 0; c < ncols; c++) {
      const known = input || r * ncols + c < count, value = matrix[r][c];
      let shade = value;
      const low = input ? inMin : outMin, high = input ? inMax : outMax;
      if (lesson.rescale && high > low && (!input || lesson.layers.length))
        shade = (value - low) / (high - low) * 255;
      shade = Math.max(0, Math.min(255, shade));
      ctx.fillStyle = known ? `rgb(${shade},${shade},${shade})` : "#e4eaf1";
      ctx.fillRect(c * cell, r * cell, cell, cell);
      if (input && (r < top || r >= top + height || c < left || c >= left + width)) {
        ctx.fillStyle = "rgba(255,166,45,.55)";
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
      ctx.strokeStyle = "#7891a9"; ctx.lineWidth = .5;
      ctx.strokeRect(c * cell, r * cell, cell, cell);
      if (known && lesson.showValues && cell >= 17) {
        ctx.fillStyle = shade > 140 ? "#111" : "#fff";
        ctx.font = `${Math.max(7, Math.min(12, cell / 3.6))}px system-ui`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const label = Number.isInteger(value) ? String(value) : Number(value.toPrecision(3)).toString();
        ctx.fillText(label, (c + .5) * cell, (r + .5) * cell, cell - 2);
      }
    }
  }
  if (!total) return;
  const active = Math.max(0, completed - 1);
  const r = Math.floor(active / cols), c = active % cols;
  ctx.strokeStyle = "#ff2848"; ctx.lineWidth = 3;
  const x = (input ? c * data.stride : c) * cell;
  const y = (input ? r * data.stride : r) * cell;
  const size = (input ? data.size : 1) * cell;
  const inset = Math.min(1.5, cell / 4);
  ctx.strokeRect(x + inset, y + inset, size - inset * 2, size - inset * 2);
}

function finished() {
  return completed === total && frameIndex === endIndex;
}

function draw() {
  canvasGrid("input", data.input, total, true);
  canvasGrid("output", data.output, completed, false);
  $("position").value = completed;
  $("progress").textContent = `${completed} / ${total} · step ${frameIndex + 1} / ${frames.length}`;
  $("play").textContent = completed === 0 && frameIndex === 0 ? "Play" : "Continue";
  $("play").disabled = running || !total;
  $("pause").disabled = !running;
  $("next").disabled = finished() || !total;
  $("handoff").textContent = handoff;
  if (!total) {
    $("calculation").textContent = "No output pixels can be built for this operation.";
    return;
  }
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
  } else if (data.operation === "ReLU") {
    formula = `max(0, ${fmt(patch[0])})`;
  } else if (data.operation === "Sigmoid") {
    formula = `1 / (1 + exp(−clip(${fmt(patch[0])}, −30, 30)))`;
  } else if (data.operation === "Tanh") {
    formula = `tanh(${fmt(patch[0])})`;
  } else if (data.operation === "Threshold at 128") {
    formula = `${fmt(patch[0])} ≥ 128 ? 255 : 0`;
  } else if (data.operation === "Contrast × 1.6") {
    formula = `clip((${fmt(patch[0])} − 128) × 1.6 + 128, 0, 255)`;
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
function advance() {
  if (completed < total) {
    completed++;
  } else if (frameIndex < endIndex) {
    const previous = data;
    loadFrame(frameIndex + 1);
    selectedLayer = data.layer ?? 0;
    handoff = previous.layer !== data.layer
      ? `Layer ${previous.layer + 1} output is now on the left as layer ${data.layer + 1} input.`
      : `${previous.step} output is now on the left as ${data.step.toLowerCase()} input.`;
    layerButtons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === selectedLayer)));
  }
  if (finished() || !total) pause();
}
function restartAtLayer(index) {
  pause(); handoff = "";
  selectedLayer = index;
  endIndex = $("one-layer").checked ? lesson.layers[index].end : frames.length - 1;
  loadFrame(lesson.layers[index].start);
  draw();
}
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(() => {
    if (!running) return;
    advance();
    draw();
    if (running) schedule();
  }, 1000 / Number($("speed").value));
}
$("play").onclick = () => {
  if (finished()) {
    if (lesson.layers.length) restartAtLayer($("one-layer").checked ? selectedLayer : 0);
    else { loadFrame(0); handoff = ""; }
  }
  running = true; draw(); schedule();
};
$("pause").onclick = () => { pause(); draw(); };
$("restart").onclick = () => {
  if (lesson.layers.length) {
    $("whole-network").checked = true; $("one-layer").checked = false;
    restartAtLayer(0);
  } else { pause(); loadFrame(0); handoff = ""; draw(); }
};
$("next").onclick = () => { pause(); advance(); draw(); };
$("position").oninput = event => { pause(); completed = Number(event.target.value); draw(); };
$("speed").oninput = () => {
  $("speed-value").textContent = $("speed").value;
  if (running) schedule();
};
if (lesson.layers.length) {
  $("network-controls").hidden = false;
  $("restart").textContent = "Restart from beginning";
  lesson.layers.forEach((layer, index) => {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = `${index + 1}. ${layer.name}`;
    button.onclick = () => restartAtLayer(index);
    $("layers").appendChild(button); layerButtons.push(button);
  });
  $("restart-layer").onclick = () => restartAtLayer(selectedLayer);
  $("whole-network").onchange = () => restartAtLayer(selectedLayer);
  $("one-layer").onchange = () => restartAtLayer(selectedLayer);
}
window.addEventListener("resize", draw);
window.addEventListener("pagehide", pause);
loadFrame(0);
draw();
</script></body></html>
"""
