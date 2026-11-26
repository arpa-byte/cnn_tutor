const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync(0, "utf8");
const script = html.split("<script>")[1].split("</script>")[0];
const elements = new Map();
const timers = new Map();
let nextTimer = 0;

function element(id) {
  return {
    id, textContent: "", disabled: false, checked: id === "whole-network",
    hidden: true, value: id === "speed" ? "4" : "0", children: [], attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    appendChild(child) { this.children.push(child); },
    getBoundingClientRect() { return {width: 320}; },
    getContext() {
      this.context = {
        fills: [], scale() {},
        fillRect(...args) { this.fills.push(args); },
        strokeRect() {}, fillText() {},
      };
      return this.context;
    },
  };
}

for (const match of html.matchAll(/id="([^"]+)"/g)) {
  elements.set(match[1], element(match[1]));
}

const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element(id));
      return elements.get(id);
    },
    createElement() { return element(""); },
  },
  window: {devicePixelRatio: 1, addEventListener() {}},
  setTimeout(callback) {
    const id = ++nextTimer;
    timers.set(id, callback);
    return id;
  },
  clearTimeout(id) { timers.delete(id); },
});
vm.runInContext(script, context);
const $ = id => elements.get(id);
const state = expression => vm.runInContext(expression, context);
const click = id => {
  assert.equal($(id).disabled, false, `${id} must be enabled`);
  $(id).onclick();
};
const tick = () => {
  assert.equal(timers.size, 1);
  const [id, callback] = timers.entries().next().value;
  timers.delete(id);
  callback();
};
const finish = () => {
  let remaining = 500;
  while (!$("next").disabled && remaining--) click("next");
  assert.ok(remaining > 0, "Playback must stop");
};

assert.equal(state("completed"), 0);
click("play");
tick();
assert.equal(state("completed"), 1);
click("pause");
assert.equal(timers.size, 0);
click("play");
tick();
assert.equal(state("completed"), 2);
click("pause");
$("position").oninput({target: {value: "0"}});
assert.equal(state("completed"), 0);

if (process.argv[2] === "network") {
  assert.equal($("network-controls").hidden, false);
  assert.equal($("layers").children.length, 2);
  $("position").oninput({target: {value: String(state("total"))}});
  click("next");
  assert.equal(state("frameIndex"), 1);
  assert.equal(state("completed"), 0);
  assert.match($("handoff").textContent, /Preprocessing output is now on the left/);
  assert.deepEqual(JSON.parse(state("JSON.stringify(data.input)")),
                   JSON.parse(state("JSON.stringify(frames[1].input)")));
  assert.match($("calculation").textContent, /×/);
  $("layers").children[1].onclick();
  assert.equal(state("frameIndex"), 4);
  assert.equal(state("completed"), 0);
  assert.match($("stage-title").textContent, /Layer 2: Smooth/);
  click("restart-layer");
  assert.equal(state("frameIndex"), 4);
  click("restart");
  assert.equal(state("frameIndex"), 0);
  $("position").oninput({target: {value: String(state("total"))}});
  click("next");
  $("one-layer").checked = true;
  $("whole-network").checked = false;
  $("one-layer").onchange();
  assert.equal(state("frameIndex"), 0);
  assert.equal(state("endIndex"), 3);
  finish();
  assert.equal(state("frameIndex"), 3);
  assert.equal(state("completed"), state("total"));
  click("restart-layer");
  assert.equal(state("frameIndex"), 0);
  click("restart");
  assert.equal($("whole-network").checked, true);
  assert.equal(state("endIndex"), 4);
  let sawHandoff = false;
  click("play");
  let remaining = 500;
  while (timers.size && remaining--) {
    tick();
    if (state("frameIndex") === 4) {
      sawHandoff = true;
      assert.match($("handoff").textContent, /Layer 1 output is now on the left as layer 2 input/);
    }
  }
  assert.ok(remaining > 0);
  assert.ok(sawHandoff);
  assert.equal(state("frameIndex"), 4);
  assert.equal(state("running"), false);
  assert.equal(state("completed"), state("total"));
} else if (process.argv[2] === "empty") {
  finish();
  assert.equal(state("total"), 0);
  assert.equal($("play").disabled, true);
  assert.match($("error").textContent, /no valid output positions/);
  assert.ok($("input").context.fills.length >= state("data.input.length * data.input[0].length"));
  assert.equal(state("running"), false);
  click("restart");
  assert.equal(state("frameIndex"), 0);
  assert.equal($("play").disabled, false);
} else {
  assert.equal($("network-controls").hidden, true);
  finish();
  assert.equal(state("completed"), state("total"));
  click("restart");
  assert.equal(state("completed"), 0);
  click("next");
  assert.equal(state("completed"), 1);
}
