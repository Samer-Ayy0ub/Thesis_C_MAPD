import argparse
import json
import tempfile
import webbrowser
from pathlib import Path

import yaml


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulation playback</title>
<style>
:root {
  color-scheme: light;
  font-family: "Segoe UI", Arial, sans-serif;
  color: #17242b;
  background: #edf2f5;
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; padding: 18px; }
header, .toolbar, main, .panel, .detail-card { border: 1px solid #d5dfe5; }
header {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 16px; margin: 0 auto 12px; max-width: 1500px;
}
.failure-banner {
  max-width: 1500px; margin: 0 auto 12px; padding: 10px 12px;
  border: 1px solid #d88; border-radius: 8px; color: #8a1f1f;
  background: #fff0f0;
}
h1 { margin: 0; font-size: 1.25rem; }
#clock { font-variant-numeric: tabular-nums; color: #176b87; font-weight: 700; }
.toolbar {
  max-width: 1500px; margin: 0 auto 12px; padding: 10px 12px;
  display: flex; flex-wrap: wrap; align-items: center; gap: 9px;
  border-radius: 10px; background: #fff;
}
button, select, input { font: inherit; }
button, select {
  border: 1px solid #b9c7d0; background: #fff; color: #17242b;
  border-radius: 6px; padding: 6px 10px;
}
button { cursor: pointer; }
button:hover { background: #e8f2f5; }
button:focus-visible, select:focus-visible, input:focus-visible {
  outline: 2px solid #1683a3; outline-offset: 2px;
}
.timeline { display: flex; flex: 1 1 260px; align-items: center; gap: 8px; }
#timeRange { width: 100%; accent-color: #176b87; }
.speed { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.loop { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
.toggle { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
.path-dropdown { position: relative; }
.path-menu-button::after {
  content: ""; display: inline-block; width: 7px; height: 7px; margin: 0 0 3px 8px;
  border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor;
  transform: rotate(45deg);
}
.path-menu-button[aria-expanded="true"]::after {
  margin-bottom: -1px; transform: rotate(225deg);
}
.path-menu-options {
  position: absolute; top: calc(100% + 6px); right: 0; z-index: 5;
  min-width: 220px; padding: 6px; border: 1px solid #b9c7d0;
  border-radius: 7px; background: #fff; box-shadow: 0 5px 16px rgba(23, 36, 43, .16);
}
.path-menu-option {
  display: flex; align-items: center; gap: 8px; padding: 7px;
  border-radius: 4px; white-space: nowrap; cursor: pointer;
}
.path-menu-option:hover { background: #e8f2f5; }
.path-menu-option input { margin: 0; }
.path-swatch { width: 20px; border-top: 3px solid; }
.path-swatch.history { border-color: #e07a25; }
.path-swatch.planned { border-top-style: dashed; border-color: #1769aa; }
main {
  max-width: 1500px; margin: 0 auto; padding: 12px;
  display: grid; grid-template-columns: minmax(0, 1fr) 290px;
  gap: 14px; align-items: start; border-radius: 10px; background: #fff;
}
.canvas-wrap { min-width: 0; }
#world {
  display: block; width: 100%; aspect-ratio: __ASPECT_RATIO__;
  border: 1px solid #d5dfe5; border-radius: 8px; background: #fbfdfe;
}
.panel { border-radius: 8px; padding: 12px; background: #f8fafb; }
.panel h2 { margin: 0 0 10px; font-size: 1rem; }
.panel label { display: block; margin-bottom: 5px; font-size: .9rem; }
#agentSelect { width: 100%; margin-bottom: 12px; }
.detail-card { border-radius: 7px; padding: 10px; background: #fff; }
.detail-row { margin: 0 0 9px; }
.detail-row:last-child { margin-bottom: 0; }
.detail-name { display: block; color: #637782; font-size: .78rem; }
.detail-value { font-weight: 600; overflow-wrap: anywhere; }
.location-task-list { display: grid; gap: 7px; margin-top: 10px; }
.location-task-entry > button { display: flex; align-items: center; gap: 8px; width: 100%; text-align: left; }
.task-color-swatch { flex: 0 0 12px; width: 12px; height: 12px; border: 1px solid #263238; border-radius: 50%; }
.location-task-details { margin-top: 5px; padding: 8px; border-left: 3px solid #1683a3; background: #f3f7f8; }
.location-task-details p { margin: 0 0 6px; }
.location-task-details p:last-child { margin-bottom: 0; }
[hidden] { display: none !important; }
.global-stats { margin-top: 12px; }
.global-stats h2 { margin: 0 0 8px; font-size: 1rem; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-card { min-width: 0; border: 1px solid #d5dfe5; border-radius: 7px; padding: 8px; background: #fff; }
.stat-name { display: block; color: #637782; font-size: .75rem; }
.stat-value { display: block; margin-top: 3px; font-size: .95rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.demo-settings { margin-top: 12px; }
.demo-settings h2 { margin: 0 0 8px; font-size: 1rem; }
.demo-settings-list {
  display: grid; gap: 7px; margin: 0; padding: 9px 10px;
  border: 1px solid #d5dfe5; border-radius: 7px; background: #fff;
}
.demo-setting { display: grid; grid-template-columns: 105px minmax(0, 1fr); gap: 8px; }
.demo-setting dt { color: #637782; font-size: .78rem; }
.demo-setting dd { margin: 0; font-size: .84rem; font-weight: 600; text-align: right; overflow-wrap: anywhere; }
.hint { color: #637782; font-size: .8rem; line-height: 1.45; }
@media (max-width: 900px) {
  body { padding: 9px; }
  main { grid-template-columns: 1fr; }
  .panel { display: grid; grid-template-columns: minmax(150px, 1fr) 2fr; gap: 8px 14px; }
  .panel h2 { grid-column: 1 / -1; }
  #agentSelect { margin-bottom: 0; }
  .demo-settings, .global-stats { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<header>
  <h1>Simulation playback</h1>
  <span id="clock">Time 0.0 / 0</span>
</header>
<div id="failureBanner" class="failure-banner" hidden></div>
<div class="toolbar" role="toolbar" aria-label="Playback controls">
  <button id="backButton" type="button" title="Step backward">Step -1</button>
  <button id="playButton" type="button" aria-label="Play simulation">Play</button>
  <button id="forwardButton" type="button" title="Step forward">Step +1</button>
  <label class="timeline" for="timeRange">
    <span>Time</span>
    <input id="timeRange" type="range" min="0" max="0" step="0.05" value="0">
  </label>
  <label class="speed" for="speedRange">
    Speed <input id="speedRange" type="range" min="0.1" max="4" step="0.1" value="__INITIAL_SPEED__">
    <span id="speedValue"></span>
  </label>
  <label class="loop"><input id="loopToggle" type="checkbox"> Loop</label>
  <label class="toggle"><input id="taskConnectionsToggle" type="checkbox" checked> Pickup-delivery line</label>
  <label class="toggle"><input id="cellValuesToggle" type="checkbox"> Cell values</label>
  <div id="pathDropdown" class="path-dropdown">
    <button id="pathMenuButton" class="path-menu-button" type="button"
            aria-haspopup="true" aria-expanded="false" aria-controls="pathMenu">Agent paths</button>
    <div id="pathMenu" class="path-menu-options" role="group"
         aria-label="Agent path overlays" hidden>
      <label class="path-menu-option">
        <input id="historyPathToggle" type="checkbox" checked>
        <span class="path-swatch history" aria-hidden="true"></span>
        <span>Traveled history</span>
      </label>
      <label class="path-menu-option">
        <input id="plannedPathToggle" type="checkbox">
        <span class="path-swatch planned" aria-hidden="true"></span>
        <span>Current planned path</span>
      </label>
    </div>
  </div>
</div>
<main>
  <div class="canvas-wrap">
    <canvas id="world" aria-label="Animated map"></canvas>
  </div>
  <aside class="panel">
    <h2 id="detailsHeading">Agent details</h2>
    <div>
      <label for="agentSelect">Select agent</label>
      <select id="agentSelect"></select>
    </div>
    <div id="agentDetailCard" class="detail-card">
      <div class="detail-row"><span class="detail-name">Status</span><span id="statusValue" class="detail-value">-</span></div>
      <div class="detail-row"><span class="detail-name">Position</span><span id="positionValue" class="detail-value">-</span></div>
      <div class="detail-row"><span id="taskLabel" class="detail-name">Assigned task</span><span id="taskValue" class="detail-value">-</span></div>
      <div class="detail-row"><span id="goalLabel" class="detail-name">Goal</span><span id="goalValue" class="detail-value">-</span></div>
      <div class="detail-row"><span id="progressLabel" class="detail-name">Progress</span><span id="progressValue" class="detail-value">-</span></div>
    </div>
    <div id="locationDetailPanel" class="detail-card" hidden>
      <div class="detail-row"><span class="detail-name">Endpoint</span><span id="locationPosition" class="detail-value">-</span></div>
      <div id="locationTaskList" class="location-task-list" aria-label="Active task locations"></div>
      <p id="noActiveLocationTasks" class="hint" hidden>No active tasks at this endpoint at this time.</p>
    </div>
    <section class="demo-settings">
      <h2>Demo settings</h2>
      <dl class="demo-settings-list">
        <div class="demo-setting"><dt>Map</dt><dd id="demoMapName">Not recorded</dd></div>
        <div class="demo-setting"><dt>Alpha</dt><dd id="demoAlpha">Not recorded</dd></div>
        <div class="demo-setting"><dt>Terraforming radius</dt><dd id="demoTerraformingRadius">Not recorded</dd></div>
        <div class="demo-setting"><dt>Laziness</dt><dd id="demoLaziness">Not recorded</dd></div>
        <div class="demo-setting"><dt>Predictor</dt><dd id="demoPredictor">Not recorded</dd></div>
      </dl>
    </section>
    <section class="global-stats">
      <h2>Global stats</h2>
      <div class="stats-grid">
        <div class="stat-card"><span class="stat-name">Terraform count</span><span id="terraformCount" class="stat-value">0</span></div>
        <div class="stat-card"><span class="stat-name">Terraform cost (alpha x moves)</span><span id="terraformCost" class="stat-value">0</span></div>
        <div class="stat-card"><span class="stat-name">Final makespan</span><span id="finalMakespan" class="stat-value">-</span></div>
        <div class="stat-card"><span class="stat-name">Final avg. delivery</span><span id="averageDeliveryTime" class="stat-value">-</span></div>
      </div>
    </section>
    <p class="hint">Outlines: idle/recovery yellow, delayed red, active task green (red while delayed). The selected agent has a neutral outer ring. Its traveled history is orange; its current plan is blue dashed. Toggle either or both in Agent paths.</p>
    <p class="hint">Click an agent or a pickup/delivery marker; when an agent is on a marker, clicking it selects the agent.</p>
  </aside>
</main>
<script>
"use strict";
const DATA = __DATA_JSON__;
const mapData = DATA.mapData;
const map = mapData.map || mapData;
const output = DATA.outputData;
const demoSettings = output.demo_settings || {};
if (output.failure_message) {
  const failureBanner = document.getElementById("failureBanner");
  failureBanner.textContent = `Failed: ${output.failure_message}`;
  failureBanner.hidden = false;
}
const width = map.dimensions[0];
const height = map.dimensions[1];
const schedule = output.schedule || {};
const stateHistory = output.agent_state_history || [];
const pathDropdown = document.getElementById("pathDropdown");
const pathMenuButton = document.getElementById("pathMenuButton");
const pathMenu = document.getElementById("pathMenu");
const historyPathToggle = document.getElementById("historyPathToggle");
const plannedPathToggle = document.getElementById("plannedPathToggle");
const hasPlannedPaths = stateHistory.some(snapshot =>
  Object.values(snapshot.agents || {}).some(state => Array.isArray(state.planned_path))
);
plannedPathToggle.disabled = !hasPlannedPaths;
const taskCompletionTimes = output.completed_tasks_times || {};
const terraformingEvents = output.terraforming_events || [];
const terraformingAlpha = output.terraforming_alpha || 0;
const settingValue = value =>
  value === undefined || value === null || value === "" ? "Not recorded" : String(value);
const predictorLabel = value => settingValue(value)
  .replace(/_predictor$/i, "")
  .replace(/_/g, " ")
  .replace(/\b\w/g, character => character.toUpperCase());
document.getElementById("demoMapName").textContent =
  settingValue(demoSettings.map_name || mapData.name);
document.getElementById("demoAlpha").textContent =
  settingValue(demoSettings.alpha === undefined ? output.terraforming_alpha : demoSettings.alpha);
document.getElementById("demoTerraformingRadius").textContent =
  settingValue(demoSettings.terraforming_radius);
document.getElementById("demoLaziness").textContent = settingValue(demoSettings.laziness);
document.getElementById("demoPredictor").textContent = predictorLabel(demoSettings.predictor);
const taskIndices = new Map((mapData.tasks || []).map((task, index) => [task.task_name, index]));
const taskAssignmentStarts = new Map();
const taskAgentHistory = new Map();
const pickupArrivalCache = new Map();
const endpointArrivalCache = new Map();
for (const snapshot of stateHistory) {
  for (const [name, state] of Object.entries(snapshot.agents)) {
    if (!state.task_name || state.task_name === "safe_idle") continue;
    const key = JSON.stringify([name, state.task_name]);
    if (!taskAssignmentStarts.has(key)) taskAssignmentStarts.set(key, snapshot.time);
    if (!taskAgentHistory.has(state.task_name)) taskAgentHistory.set(state.task_name, []);
    const history = taskAgentHistory.get(state.task_name);
    if (!history.length || history[history.length - 1].agent !== name) {
      history.push({time: snapshot.time, agent: name});
    }
  }
}
const agents = Object.keys(schedule).filter(name => !name.startsWith("obs_"));
const obstacles = Object.keys(schedule).filter(name => name.startsWith("obs_"));
const staticObstacleSet = new Set((map.obstacles || []).map(([x, y]) => `${x},${y}`));
const canvas = document.getElementById("world");
const context = canvas.getContext("2d");
const timeRange = document.getElementById("timeRange");
const playButton = document.getElementById("playButton");
const speedRange = document.getElementById("speedRange");
const loopToggle = document.getElementById("loopToggle");
const taskConnectionsToggle = document.getElementById("taskConnectionsToggle");
const cellValuesToggle = document.getElementById("cellValuesToggle");
const cellHistory = output.cell_values_history || [];
const cellValues = output.cell_values || [];
let maxTime = 0;
for (const path of Object.values(schedule)) {
  if (path.length) maxTime = Math.max(maxTime, path[path.length - 1].t);
}
for (const snapshot of stateHistory) maxTime = Math.max(maxTime, snapshot.time);
timeRange.max = String(maxTime);
timeRange.value = "0";
document.getElementById("agentSelect").innerHTML = agents.map(name =>
  `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`
).join("");
let selectedAgent = agents[0] || "";
let selectedMode = "agent";
let selectedLocation = null;
let expandedLocationTaskKey = null;
document.getElementById("agentSelect").value = selectedAgent;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}
function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}
function lastSnapshotAtOrBefore(history, time) {
  let lower = 0;
  let upper = history.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (history[middle].time <= time) lower = middle + 1;
    else upper = middle;
  }
  return lower - 1;
}
function positionAt(path, time) {
  if (!path || !path.length) return null;
  if (time <= path[0].t) return [path[0].x, path[0].y];
  let lower = 0;
  let upper = path.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (path[middle].t < time) lower = middle + 1;
    else upper = middle;
  }
  if (lower === path.length) {
    const last = path[path.length - 1];
    return [last.x, last.y];
  }
  const previous = path[lower - 1];
  const next = path[lower];
  const duration = next.t - previous.t;
  const part = duration ? (time - previous.t) / duration : 0;
  return [
    previous.x + (next.x - previous.x) * part,
    previous.y + (next.y - previous.y) * part
  ];
}
function stateAt(name, time) {
  const index = lastSnapshotAtOrBefore(stateHistory, time);
  return index < 0 ? null : stateHistory[index].agents[name] || null;
}
function valuesAt(time) {
  const index = lastSnapshotAtOrBefore(cellHistory, time);
  return index < 0 ? cellValues : cellHistory[index].values;
}
function agentColor(name) {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360} 58% 40%)`;
}
function interpolateColor(value, minimum, maximum) {
  const part = clamp((value - minimum) / (maximum - minimum || 1), 0, 1);
  const stops = [[244, 249, 250], [112, 190, 177], [28, 111, 143]];
  const scaled = part * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const rgb = stops[index].map((start, channel) =>
    Math.round(start + (stops[index + 1][channel] - start) * fraction)
  );
  return `rgb(${rgb.join(",")})`;
}
function taskColor(index) {
  return `hsl(${(index * 137.5) % 360} 67% 42%)`;
}
function taskColorForName(taskName) {
  const index = taskIndices.get(taskName);
  return index === undefined ? agentColor(taskName) : taskColor(index);
}
function taskIsVisible(task, time) {
  const started = task.start_time || 0;
  const finished = taskCompletionTimes[task.task_name];
  return time >= started && (finished === undefined || time <= finished);
}
function taskLocationsAt(time) {
  const locations = [];
  for (const task of mapData.tasks || []) {
    if (!taskIsVisible(task, time)) continue;
    locations.push({task, kind: "Pickup", position: task.start});
    locations.push({task, kind: "Delivery", position: task.goal});
  }
  return locations;
}
function taskAssignmentAt(taskName, time) {
  const history = taskAgentHistory.get(taskName) || [];
  const index = lastSnapshotAtOrBefore(history, time);
  return index < 0 ? null : history[index];
}
function endpointArrivalTime(task, kind, assignment) {
  const key = JSON.stringify([task.task_name, kind, assignment.agent, assignment.time]);
  if (endpointArrivalCache.has(key)) return endpointArrivalCache.get(key);
  const endpoint = kind === "Pickup" ? task.start : task.goal;
  const path = schedule[assignment.agent] || [];
  let firstVisit = Infinity;
  const positionBeforeAssignment = positionAt(path, assignment.time - 1);
  if (positionBeforeAssignment &&
      positionBeforeAssignment[0] === endpoint[0] &&
      positionBeforeAssignment[1] === endpoint[1]) {
    firstVisit = assignment.time;
  }
  for (const step of path) {
    if (step.t >= assignment.time && step.x === endpoint[0] && step.y === endpoint[1]) {
      firstVisit = Math.min(firstVisit, step.t);
      break;
    }
  }
  endpointArrivalCache.set(key, firstVisit);
  return firstVisit;
}
function taskEndpointReached(task, kind, time, assignment) {
  return !!assignment && endpointArrivalTime(task, kind, assignment) <= time;
}
function cellValueLabel(value, cell) {
  if (!Number.isFinite(value) || cell < 18) return "";
  const fontSize = Math.max(7, Math.min(12, cell * 0.25));
  const maxWidth = cell * 0.9;
  context.font = `${fontSize}px "Segoe UI", Arial, sans-serif`;
  const initialDecimals = Math.min(5, Math.max(0, Math.floor((cell - 18) / 7)));
  for (let decimals = initialDecimals; decimals >= 0; decimals -= 1) {
    let text = value.toFixed(decimals);
    if (text.includes(".")) text = text.replace(/0+$/, "").replace(/\.$/, "");
    if ((Number(text) !== 0 || value === 0) &&
        context.measureText(text).width <= maxWidth) return text;
  }
  const scientific = value.toExponential(1).replace("e+", "e");
  return context.measureText(scientific).width <= maxWidth ? scientific : "";
}

let currentTime = 0;
let playing = false;
let previousFrameTime = 0;
let layout = null;

function drawTrail(name, time, cell, mapX, selected, mode) {
  const path = schedule[name] || [];
  const showingPlan = mode === "planned";
  let points;
  if (showingPlan) {
    const state = stateAt(name, time);
    points = (state && Array.isArray(state.planned_path) ? state.planned_path : [])
      .map(position => [position[0], position[1]]);
    const current = positionAt(path, time);
    if (points.length && current) points[0] = current;
  } else {
    points = path.filter(step => step.t <= time).map(step => [step.x, step.y]);
    const current = positionAt(path, time);
    if (current && (!points.length || points[points.length - 1][0] !== current[0] ||
        points[points.length - 1][1] !== current[1])) points.push(current);
  }
  if (points.length < 2) return;
  context.beginPath();
  points.forEach((point, index) => {
    const pixel = mapX(point[0], point[1]);
    if (index === 0) context.moveTo(pixel[0], pixel[1]);
    else context.lineTo(pixel[0], pixel[1]);
  });
  context.strokeStyle = selected
    ? (showingPlan ? "#1769aa" : "#e07a25") : agentColor(name);
  context.globalAlpha = selected ? 0.9 : 0.2;
  context.lineWidth = selected ? Math.max(3, cell * 0.08) : Math.max(1, cell * 0.035);
  context.setLineDash(showingPlan ? [cell * 0.14, cell * 0.1] : []);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.stroke();
  context.setLineDash([]);
  context.globalAlpha = 1;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(rect.width * pixelRatio);
  const pixelHeight = Math.round(rect.height * pixelRatio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  const cell = Math.min(rect.width / width, rect.height / height);
  const left = (rect.width - width * cell) / 2;
  const top = (rect.height - height * cell) / 2;
  const mapX = (x, y) => [left + (x + 0.5) * cell, top + (height - y - 0.5) * cell];
  layout = {left, top, cell, width, height};
  context.fillStyle = "#fbfdfe";
  context.fillRect(0, 0, rect.width, rect.height);

  const values = valuesAt(currentTime);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const row of values) for (const value of row) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  if (!Number.isFinite(minimum)) { minimum = 0; maximum = 1; }
  if (minimum === maximum) maximum = minimum + 1;
  context.save();
  context.beginPath();
  context.rect(left, top, width * cell, height * cell);
  context.clip();
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const value = values[y] && values[y][x];
      context.fillStyle = Number.isFinite(value)
        ? interpolateColor(value, minimum, maximum) : "#f4f7f8";
      context.fillRect(left + x * cell, top + (height - y - 1) * cell, cell, cell);
    }
  }
  context.strokeStyle = "#b8c5cb";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += 1) {
    context.beginPath();
    context.moveTo(left + x * cell, top);
    context.lineTo(left + x * cell, top + height * cell);
    context.stroke();
  }
  for (let y = 0; y <= height; y += 1) {
    context.beginPath();
    context.moveTo(left, top + y * cell);
    context.lineTo(left + width * cell, top + y * cell);
    context.stroke();
  }
  for (const [x, y] of map.obstacles || []) {
    context.fillStyle = "#aeb9bf";
    context.fillRect(left + x * cell, top + (height - y - 1) * cell, cell, cell);
  }
  if (cellValuesToggle.checked) {
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (let x = 0; x < width; x += 1) {
      for (let y = 0; y < height; y += 1) {
        if (staticObstacleSet.has(`${x},${y}`)) continue;
        const value = values[y] && values[y][x];
        const text = cellValueLabel(value, cell);
        if (!text) continue;
        const point = mapX(x, y);
        context.fillStyle = (value - minimum) / (maximum - minimum) > 0.58
          ? "#ffffff" : "#17242b";
        context.fillText(text, point[0], point[1], cell * 0.9);
      }
    }
  }
  for (const [x, y] of map.non_task_endpoints || []) {
    const point = mapX(x, y);
    context.beginPath();
    context.arc(point[0], point[1], cell * 0.27, 0, Math.PI * 2);
    context.fillStyle = "rgba(22, 131, 93, 0.22)";
    context.fill();
    context.strokeStyle = "#16835d";
    context.lineWidth = Math.max(1, cell * 0.035);
    context.stroke();
  }

  for (let index = 0; index < (mapData.tasks || []).length; index += 1) {
    const task = mapData.tasks[index];
    if (!taskIsVisible(task, currentTime)) continue;
    const color = taskColor(index);
    const start = mapX(task.start[0], task.start[1]);
    const goal = mapX(task.goal[0], task.goal[1]);
    if (taskConnectionsToggle.checked) {
      context.save();
      context.globalAlpha = 0.68;
      context.strokeStyle = color;
      context.setLineDash([cell * 0.12, cell * 0.1]);
      context.beginPath();
      context.moveTo(start[0], start[1]);
      context.lineTo(goal[0], goal[1]);
      context.stroke();
      context.restore();
    }
    context.fillStyle = color;
    context.beginPath();
    context.arc(start[0], start[1], cell * 0.16, 0, Math.PI * 2);
    context.fill();
    context.fillRect(goal[0] - cell * 0.14, goal[1] - cell * 0.14, cell * 0.28, cell * 0.28);
    if (selectedMode === "location" && selectedLocation) {
      context.strokeStyle = "#263238";
      context.lineWidth = Math.max(2, cell * 0.045);
      if (selectedLocation.x === task.start[0] && selectedLocation.y === task.start[1]) {
        context.beginPath();
        context.arc(start[0], start[1], cell * 0.25, 0, Math.PI * 2);
        context.stroke();
      }
      if (selectedLocation.x === task.goal[0] && selectedLocation.y === task.goal[1]) {
        context.strokeRect(
          goal[0] - cell * 0.23, goal[1] - cell * 0.23, cell * 0.46, cell * 0.46
        );
      }
    }
  }

  if (selectedMode === "agent" && selectedAgent) {
    if (plannedPathToggle.checked) {
      drawTrail(selectedAgent, currentTime, cell, mapX, true, "planned");
    }
    if (historyPathToggle.checked) {
      drawTrail(selectedAgent, currentTime, cell, mapX, true, "history");
    }
  }

  for (const name of obstacles) {
    const position = positionAt(schedule[name], currentTime);
    if (!position) continue;
    const point = mapX(position[0], position[1]);
    const side = cell * 0.62;
    context.fillStyle = "#596970";
    context.fillRect(point[0] - side / 2, point[1] - side / 2, side, side);
    context.strokeStyle = "#e36b35";
    context.lineWidth = Math.max(1, cell * 0.04);
    context.strokeRect(point[0] - side / 2, point[1] - side / 2, side, side);
  }

  for (const name of agents) {
    const position = positionAt(schedule[name], currentTime);
    if (!position) continue;
    const point = mapX(position[0], position[1]);
    const radius = cell * 0.29;
    const appearance = agentAppearance(name, currentTime);
    context.beginPath();
    context.arc(point[0], point[1], radius, 0, Math.PI * 2);
    context.fillStyle = appearance.fill;
    context.fill();
    context.lineWidth = Math.max(2, cell * 0.065);
    context.strokeStyle = appearance.outline;
    context.stroke();
    if (selectedMode === "agent" && name === selectedAgent) {
      context.beginPath();
      context.arc(point[0], point[1], radius + cell * 0.14, 0, Math.PI * 2);
      context.lineWidth = Math.max(2, cell * 0.035);
      context.strokeStyle = "#34434a";
      context.stroke();
    }
    context.fillStyle = "#ffffff";
    context.font = `bold ${Math.max(9, cell * 0.2)}px sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(name.replace("agent", "A"), point[0], point[1]);
  }
  context.restore();
  updateDetails();
  updateGlobalStats();
}

function assignmentStart(name, taskName, time) {
  return taskAssignmentStarts.get(JSON.stringify([name, taskName])) || 0;
}
function hasReachedPickup(name, state, time) {
  if (!state.start) return false;
  const key = JSON.stringify([name, state.task_name]);
  if (pickupArrivalCache.has(key)) return pickupArrivalCache.get(key) <= time;
  const path = schedule[name] || [];
  const pickup = state.start;
  const assignedAt = assignmentStart(name, state.task_name, time);
  let arrivalTime = Infinity;
  if (assignedAt > 0) {
    const positionBeforeAssignment = positionAt(path, assignedAt - 1);
    if (positionBeforeAssignment &&
        positionBeforeAssignment[0] === pickup[0] &&
        positionBeforeAssignment[1] === pickup[1]) arrivalTime = assignedAt - 1;
  }
  if (!Number.isFinite(arrivalTime)) {
    for (const step of path) {
      if (step.t >= assignedAt && step.x === pickup[0] && step.y === pickup[1]) {
        arrivalTime = step.t;
        break;
      }
    }
  }
  pickupArrivalCache.set(key, arrivalTime);
  return arrivalTime <= time;
}
function agentAppearance(name, time) {
  const state = stateAt(name, time);
  let fill = agentColor(name);
  let outline = "#ffffff";
  const delayed = state && state.status === "Delayed";
  const completed = state && state.status === "Completed";
  if (state && state.task_name && state.task_name !== "safe_idle" && !completed) {
    outline = delayed ? "#d62828" : "#2a9d55";
    if (hasReachedPickup(name, state, time)) {
      fill = taskColorForName(state.task_name);
    }
  } else if (delayed) {
    outline = "#d62828";
  } else if (completed) {
    return {fill, outline};
  } else {
    outline = "#f2c230";
  }
  return {fill, outline};
}
function movedSteps(name, start, time) {
  const path = schedule[name] || [];
  let previous = null;
  let moves = 0;
  for (const step of path) {
    if (step.t < start || step.t > time) continue;
    if (previous && (previous.x !== step.x || previous.y !== step.y)) moves += 1;
    previous = step;
  }
  return moves;
}
function updateDetails() {
  if (selectedMode === "location" && selectedLocation) {
    document.getElementById("agentDetailCard").hidden = true;
    document.getElementById("locationDetailPanel").hidden = false;
    updateLocationDetails();
    return;
  }
  document.getElementById("agentDetailCard").hidden = false;
  document.getElementById("locationDetailPanel").hidden = true;
  document.getElementById("detailsHeading").textContent = "Agent details";
  document.getElementById("taskLabel").textContent = "Assigned task";
  document.getElementById("goalLabel").textContent = "Goal";
  document.getElementById("progressLabel").textContent = "Progress";
  const position = positionAt(schedule[selectedAgent], currentTime);
  const state = stateAt(selectedAgent, currentTime);
  const taskName = state && state.task_name;
  document.getElementById("statusValue").textContent = state ? state.status : "Idle";
  document.getElementById("positionValue").textContent = position
    ? `[${position[0].toFixed(2)}, ${position[1].toFixed(2)}]` : "-";
  document.getElementById("taskValue").textContent = taskName
    ? (taskName === "safe_idle" ? "Moving to safe idle position" : taskName) : "Unassigned";
  document.getElementById("goalValue").textContent = state && state.goal
    ? `[${state.goal[0]}, ${state.goal[1]}]` : "-";
  let progress = "No active task";
  if (taskName && taskName !== "safe_idle") {
    const completedAt = taskCompletionTimes[taskName];
    if (state.status === "Completed" || (completedAt !== undefined && currentTime >= completedAt)) {
      progress = `Completed at t=${completedAt === undefined ? currentTime : completedAt}`;
    } else {
      const start = assignmentStart(selectedAgent, taskName, currentTime);
      progress = `Active for ${(currentTime - start).toFixed(1)} ticks; ${movedSteps(selectedAgent, start, currentTime)} moves`;
    }
  } else if (taskName === "safe_idle") {
    progress = state.status;
  }
  document.getElementById("progressValue").textContent = progress;
}
function updateLocationDetails() {
  document.getElementById("detailsHeading").textContent = "Task locations at endpoint";
  document.getElementById("locationPosition").textContent =
    `[${selectedLocation.x}, ${selectedLocation.y}]`;
  const list = document.getElementById("locationTaskList");
  list.replaceChildren();
  const locations = taskLocationsAt(currentTime).filter(location =>
    location.position[0] === selectedLocation.x &&
    location.position[1] === selectedLocation.y
  );
  document.getElementById("noActiveLocationTasks").hidden = locations.length > 0;
  for (const location of locations) {
    const key = JSON.stringify([location.task.task_name, location.kind]);
    const expanded = key === expandedLocationTaskKey;
    const entry = document.createElement("div");
    entry.className = "location-task-entry";
    const button = document.createElement("button");
    button.type = "button";
    const swatch = document.createElement("span");
    swatch.className = "task-color-swatch";
    swatch.setAttribute("aria-hidden", "true");
    swatch.style.backgroundColor = taskColorForName(location.task.task_name);
    button.append(swatch, document.createTextNode(`${location.kind} ${location.task.task_name}`));
    button.setAttribute("aria-expanded", String(expanded));
    button.addEventListener("click", () => {
      expandedLocationTaskKey = expanded ? null : key;
      draw();
    });
    entry.appendChild(button);
    if (expanded) {
      const assignment = taskAssignmentAt(location.task.task_name, currentTime);
      const details = document.createElement("div");
      details.className = "location-task-details";
      const agentRow = document.createElement("p");
      const agentLabel = document.createElement("span");
      agentLabel.className = "detail-name";
      agentLabel.textContent = "Assigned agent";
      const agentValue = document.createElement("span");
      agentValue.className = "detail-value";
      agentValue.textContent = assignment ? assignment.agent : "None";
      agentRow.append(agentLabel, agentValue);
      const statusRow = document.createElement("p");
      const statusLabel = document.createElement("span");
      statusLabel.className = "detail-name";
      statusLabel.textContent = "Status";
      const statusValue = document.createElement("span");
      statusValue.className = "detail-value";
      statusValue.textContent =
        taskEndpointReached(location.task, location.kind, currentTime, assignment)
          ? "Reached" : "Not reached";
      statusRow.append(statusLabel, statusValue);
      details.append(agentRow, statusRow);
      entry.appendChild(details);
    }
    list.appendChild(entry);
  }
}
function updateGlobalStats() {
  const eventIndex = lastSnapshotAtOrBefore(terraformingEvents, currentTime);
  const count = eventIndex + 1;
  document.getElementById("terraformCount").textContent = String(count);
  document.getElementById("terraformCost").textContent = (count * terraformingAlpha).toFixed(2);
  document.getElementById("finalMakespan").textContent =
    output.final_makespan === undefined ? "-" : String(output.final_makespan);
  document.getElementById("averageDeliveryTime").textContent =
    output.average_delivery_time === undefined ? "-" : Number(output.average_delivery_time).toFixed(2);
}
function setTime(value) {
  currentTime = clamp(Number(value) || 0, 0, maxTime);
  timeRange.value = String(currentTime);
  document.getElementById("clock").textContent = `Time ${currentTime.toFixed(1)} / ${maxTime}`;
  draw();
}
function stepTime(amount) {
  setTime(clamp(Math.round(currentTime) + amount, 0, maxTime));
}
function setPlaying(value) {
  playing = value;
  playButton.textContent = playing ? "Pause" : "Play";
  playButton.setAttribute("aria-label", playing ? "Pause simulation" : "Play simulation");
  previousFrameTime = 0;
  if (playing) requestAnimationFrame(playFrame);
}
function playFrame(now) {
  if (!playing) return;
  if (previousFrameTime) {
    const rate = Number(speedRange.value) * 10;
    currentTime += (now - previousFrameTime) * rate / 1000;
    if (currentTime >= maxTime) {
      if (loopToggle.checked && maxTime > 0) currentTime = 0;
      else {
        currentTime = maxTime;
        setPlaying(false);
      }
    }
  }
  previousFrameTime = now;
  setTime(currentTime);
  if (playing) requestAnimationFrame(playFrame);
}

document.getElementById("agentSelect").addEventListener("change", event => {
  selectedAgent = event.target.value;
  selectedMode = "agent";
  selectedLocation = null;
  draw();
});
taskConnectionsToggle.addEventListener("change", draw);
cellValuesToggle.addEventListener("change", draw);
historyPathToggle.addEventListener("change", draw);
plannedPathToggle.addEventListener("change", draw);
pathMenuButton.addEventListener("click", () => {
  const open = pathMenu.hidden;
  pathMenu.hidden = !open;
  pathMenuButton.setAttribute("aria-expanded", String(open));
});
document.addEventListener("click", event => {
  if (!pathDropdown.contains(event.target)) {
    pathMenu.hidden = true;
    pathMenuButton.setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !pathMenu.hidden) {
    pathMenu.hidden = true;
    pathMenuButton.setAttribute("aria-expanded", "false");
    pathMenuButton.focus();
  }
});
document.getElementById("backButton").addEventListener("click", () => stepTime(-1));
document.getElementById("forwardButton").addEventListener("click", () => stepTime(1));
playButton.addEventListener("click", () => setPlaying(!playing));
timeRange.addEventListener("input", event => setTime(event.target.value));
speedRange.addEventListener("input", event => {
  document.getElementById("speedValue").textContent = `${Number(event.target.value).toFixed(1)}x`;
});
canvas.addEventListener("click", event => {
  if (!layout) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left - layout.left) / layout.cell - 0.5;
  const y = layout.height - 0.5 - (event.clientY - rect.top - layout.top) / layout.cell;
  let nearest = null;
  let nearestDistance = Infinity;
  for (const name of agents) {
    const position = positionAt(schedule[name], currentTime);
    if (!position) continue;
    const distance = Math.hypot(position[0] - x, position[1] - y);
    if (distance < nearestDistance) { nearest = name; nearestDistance = distance; }
  }
  let nearestLocation = null;
  let nearestLocationDistance = Infinity;
  for (const location of taskLocationsAt(currentTime)) {
    const distance = Math.hypot(location.position[0] - x, location.position[1] - y);
    if (distance < nearestLocationDistance) {
      nearestLocation = location;
      nearestLocationDistance = distance;
    }
  }
  if (nearestLocation && nearestLocationDistance <= 0.4) {
    let agentOnLocation = null;
    let distanceToLocation = Infinity;
    for (const name of agents) {
      const position = positionAt(schedule[name], currentTime);
      if (!position) continue;
      const distance = Math.hypot(
        position[0] - nearestLocation.position[0],
        position[1] - nearestLocation.position[1]
      );
      if (distance <= 0.2 && distance < distanceToLocation) {
        agentOnLocation = name;
        distanceToLocation = distance;
      }
    }
    if (agentOnLocation) {
      selectedAgent = agentOnLocation;
      selectedMode = "agent";
      selectedLocation = null;
      document.getElementById("agentSelect").value = selectedAgent;
    } else {
      selectedMode = "location";
      selectedLocation = {x: nearestLocation.position[0], y: nearestLocation.position[1]};
      expandedLocationTaskKey = null;
    }
    draw();
  } else if (nearest && nearestDistance <= 0.55) {
    selectedAgent = nearest;
    selectedMode = "agent";
    selectedLocation = null;
    document.getElementById("agentSelect").value = nearest;
    draw();
  }
});
window.addEventListener("resize", draw);
window.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); setPlaying(!playing); }
  else if (event.code === "ArrowLeft") stepTime(-1);
  else if (event.code === "ArrowRight") stepTime(1);
});
document.getElementById("speedValue").textContent = `${Number(speedRange.value).toFixed(1)}x`;
document.getElementById("clock").textContent = `Time 0.0 / ${maxTime}`;
draw();
</script>
</body>
</html>
"""


def build_html(map_data, output_data, slow_factor=1):
    payload = json.dumps(
        {'mapData': map_data, 'outputData': output_data},
        ensure_ascii=True,
    ).replace('<', '\\u003c')
    initial_speed = max(0.1, min(4.0, 1 / max(1, slow_factor)))
    aspect_ratio = max(1, map_data['map']['dimensions'][0]) / max(
        1, map_data['map']['dimensions'][1]
    )
    return (
        HTML_TEMPLATE
        .replace('__DATA_JSON__', payload)
        .replace('__INITIAL_SPEED__', f'{initial_speed:.2f}')
        .replace('__ASPECT_RATIO__', f'{aspect_ratio:g}')
    )


def open_visualizer(map_path, schedule_path, slow_factor=1):
    with open(map_path, encoding='utf-8') as map_file:
        map_data = yaml.safe_load(map_file)
    with open(schedule_path, encoding='utf-8') as schedule_file:
        output_data = yaml.safe_load(schedule_file)
    demo_settings = output_data.setdefault('demo_settings', {})
    demo_settings.setdefault('map_name', Path(map_path).name)
    page = build_html(map_data, output_data, slow_factor)
    visualizer_temp_dir = Path(tempfile.gettempdir()) / 'thesis_simulation_visualizer'
    visualizer_temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', suffix='.html', prefix='simulation_',
        dir=visualizer_temp_dir, delete=False
    ) as html_file:
        html_file.write(page)
        page_path = Path(html_file.name)
    print(f'Visualizer pages are saved in: {visualizer_temp_dir}')
    if not webbrowser.open_new_tab(page_path.as_uri()):
        print(f'Could not open a browser automatically. Open this file: {page_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', required=True, help='Map and task YAML file')
    parser.add_argument('-schedule', required=True, help='Simulation output YAML file')
    parser.add_argument('-slow_factor', default=1, type=int)
    args = parser.parse_args()
    open_visualizer(args.map, args.schedule, args.slow_factor)
