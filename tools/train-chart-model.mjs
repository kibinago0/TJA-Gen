#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { access, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { TextDecoder } from "node:util";

const SAMPLE_RATE = 11025;
const FRAME_SAMPLES = 110;
const FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE;
const CLASS_COUNT = 10;
let updateCount = 0;
const FEATURE_NAMES = [
  "rms_before",
  "rms_center",
  "rms_after",
  "rms_local_max",
  "rms_local_mean",
  "onset_before",
  "onset_center",
  "onset_after",
  "onset_local_max",
  "onset_local_mean",
  "measure_phase_sin",
  "measure_phase_cos",
  "beat_phase_sin",
  "beat_phase_cos",
  "bpm",
  "level",
  "course_easy",
  "course_normal",
  "course_hard",
  "course_oni",
  "course_edit"
];

const COURSE_INDEX = new Map([
  ["easy", 0], ["0", 0],
  ["normal", 1], ["1", 1],
  ["hard", 2], ["2", 2],
  ["oni", 3], ["3", 3],
  ["edit", 4], ["4", 4]
]);

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  printHelp();
  process.exit(0);
}

const root = path.resolve(options.dataDir);
const outputDir = path.join(root, ".tja-gen-training");
const summary = {
  tjaFiles: 0,
  charts: 0,
  missingAudio: 0,
  unsupported: {},
  trainingCharts: 0,
  validationCharts: 0,
  validationSlots: 0
};

try {
  const groups = await loadChartGroups(root, summary);
  if (groups.length < 3) {
    throw new Error("音源が3曲未満です。少なくとも3つの対応音源が必要です。");
  }

  const selectedGroups = selectGroups(groups, options.limit);
  let hashedGroups = await hashAndGroupAudio(selectedGroups);
  if (hashedGroups.length < 3) {
    throw new Error("重複音源をまとめた後、学習に使える音源が3曲未満です。");
  }
  assignValidationSplit(hashedGroups);
  const trainingGroups = hashedGroups.filter((group) => !group.validation);
  const validationGroups = hashedGroups.filter((group) => group.validation);
  if (!trainingGroups.length || !validationGroups.length) {
    throw new Error("学習用と評価用に音源を分けられませんでした。対象音源数を増やしてください。");
  }

  const classCounts = new Array(CLASS_COUNT).fill(0);
  for (const group of hashedGroups) {
    for (const chart of group.charts) {
      const prepared = prepareChart(chart);
      if (!prepared.ok) {
        countUnsupported(summary, prepared.reason);
        continue;
      }
      if (group.validation) continue;
      for (const measure of prepared.measures) {
        for (const label of measure.labels) classCounts[label] += 1;
      }
    }
  }

  const classWeights = makeClassWeights(classCounts);
  const model = createModel(FEATURE_NAMES.length);
  const random = seededRandom(0x544a4147);
  let trainedCharts = 0;
  let trainedSlots = 0;
  let processed = 0;

  for (const group of trainingGroups) {
    const envelope = await decodeAudio(group.audioPath);
    for (const chart of group.charts) {
      const prepared = prepareChart(chart);
      if (!prepared.ok) continue;
      const chartExamples = makeTrainingExamples(prepared, envelope, group.hash, trainedCharts);
      if (chartExamples.length) {
        shuffle(chartExamples, random);
        for (const example of chartExamples) {
          trainExample(model, example.features, example.label, classWeights);
        }
        trainedSlots += chartExamples.length;
        trainedCharts += 1;
      }
    }
    processed += 1;
    printProgress("学習", processed, trainingGroups.length);
  }
  console.log();

  let validationCharts = 0;
  let validationSlots = 0;
  const confusion = Array.from({ length: CLASS_COUNT }, () => new Array(CLASS_COUNT).fill(0));
  for (const group of validationGroups) {
    const envelope = await decodeAudio(group.audioPath);
    for (const chart of group.charts) {
      const prepared = prepareChart(chart);
      if (!prepared.ok) continue;
      validationCharts += 1;
      for (const sample of chartSamples(prepared, envelope)) {
        const prediction = predict(model, sample.features);
        confusion[sample.label][prediction] += 1;
        validationSlots += 1;
      }
    }
    processed += 1;
    printProgress("評価", processed - trainingGroups.length, validationGroups.length);
  }
  console.log();

  if (!trainedSlots || !validationSlots) {
    throw new Error("学習または評価に使えるノーツがありません。譜面の形式と対応音源を確認してください。");
  }

  summary.audioGroups = hashedGroups.length;
  summary.audioGroupsDiscovered = groups.length;
  summary.selectedAudioGroups = selectedGroups.length;
  summary.scope = options.limit ? "limited smoke run" : "all discovered audio references";
  summary.trainingAudioGroups = trainingGroups.length;
  summary.validationAudioGroups = validationGroups.length;
  summary.trainingCharts = trainedCharts;
  summary.validationCharts = validationCharts;
  summary.trainingExamples = trainedSlots;
  summary.validationSlots = validationSlots;
  summary.noteClassCounts = classCounts;
  summary.metrics = calculateMetrics(confusion);
  summary.model = "multiclass logistic regression (one local pass; sampled empty slots)";
  summary.audioFeatures = "10 ms mono RMS and positive RMS differences";
  summary.split = "SHA-256 audio-content groups; deterministic 20% holdout";
  summary.offsetRule = "audio time = chart time - TJA OFFSET";

  await mkdir(outputDir, { recursive: true });
  await writeFile(path.join(outputDir, "model.json"), JSON.stringify({
    formatVersion: 1,
    classes: CLASS_COUNT,
    features: FEATURE_NAMES,
    weights: model.map((row) => Array.from(row))
  }, null, 2), "utf8");
  await writeFile(path.join(outputDir, "report.json"), JSON.stringify(summary, null, 2), "utf8");

  console.log("完了しました。");
  console.log(`TJAファイル: ${summary.tjaFiles}`);
  console.log(`音源グループ: ${summary.audioGroups}（学習 ${summary.trainingAudioGroups} / 評価 ${summary.validationAudioGroups}）`);
  console.log(`利用チャート: 学習 ${summary.trainingCharts} / 評価 ${summary.validationCharts}`);
  console.log(`ノーツ検出 F1: ${summary.metrics.notePresence.f1.toFixed(3)}（評価用音源のみ）`);
  console.log(`全クラス正解率: ${summary.metrics.accuracy.toFixed(3)}`);
  console.log(`出力先: ${outputDir}`);
  console.log("※ 初回結果は基準値です。生成譜面としての品質を保証するものではありません。");
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  if (Object.keys(summary.unsupported).length) {
    console.error("除外理由（チャート数）:", JSON.stringify(summary.unsupported));
  }
  process.exitCode = 1;
}

async function loadChartGroups(dataDir, report) {
  const files = await findFiles(dataDir, ".tja");
  report.tjaFiles = files.length;
  const byAudio = new Map();
  const decoder = new TextDecoder("shift_jis");

  for (const file of files) {
    const buffer = await readFile(file);
    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
    } catch {
      text = decoder.decode(buffer);
    }
    for (const chart of parseTja(text, path.dirname(file))) {
      report.charts += 1;
      if (!chart.audioPath || chart.courseIndex < 0) {
        countUnsupported(report, "missing_audio_or_course");
        continue;
      }
      if (!chart.bpm || !Number.isFinite(chart.bpm) || chart.bpm <= 0) {
        countUnsupported(report, "invalid_bpm");
        continue;
      }
      if (!chart.level || chart.level < 1 || chart.level > 10) {
        countUnsupported(report, "invalid_level");
        continue;
      }
      try {
        await access(chart.audioPath);
      } catch {
        report.missingAudio += 1;
        continue;
      }
      const key = path.resolve(chart.audioPath).toLowerCase();
      if (!byAudio.has(key)) byAudio.set(key, { audioPath: chart.audioPath, charts: [] });
      byAudio.get(key).charts.push(chart);
    }
  }
  return [...byAudio.values()].sort((a, b) => a.audioPath.localeCompare(b.audioPath));
}

function parseTja(text, directory) {
  const charts = [];
  const headers = {};
  let chart = null;
  let bpm = 0;
  let beatsPerMeasure = 4;
  let elapsed = 0;
  let measureDigits = "";
  let reason = "";
  let currentMeasure = null;

  const finishMeasure = () => {
    if (!measureDigits.length) return;
    currentMeasure = {
      start: elapsed,
      duration: beatsPerMeasure * 60 / bpm,
      beats: beatsPerMeasure,
      bpm,
      digits: measureDigits
    };
    chart.measures.push(currentMeasure);
    elapsed += currentMeasure.duration;
    measureDigits = "";
  };

  const finishChart = () => {
    if (!chart) return;
    finishMeasure();
    chart.parseReason = reason;
    charts.push(chart);
    chart = null;
  };

  for (const originalLine of text.split(/\r?\n/)) {
    const line = originalLine.trim();
    if (!chart) {
      const header = line.match(/^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$/i);
      if (header) headers[header[1].toUpperCase()] = header[2].trim();
      if (/^#START(?:\s|$)/i.test(line)) {
        const courseIndex = COURSE_INDEX.get((headers.COURSE || "").toLowerCase());
        chart = {
          audioPath: headers.WAVE ? path.resolve(directory, headers.WAVE) : "",
          courseIndex: courseIndex ?? -1,
          level: Number(headers.LEVEL),
          bpm: Number(headers.BPM),
          offset: Number(headers.OFFSET || 0),
          measures: []
        };
        bpm = chart.bpm;
        beatsPerMeasure = 4;
        elapsed = 0;
        measureDigits = "";
        reason = "";
        continue;
      }
      continue;
    }

    if (/^#END\b/i.test(line)) {
      finishChart();
      continue;
    }
    if (line.startsWith("#")) {
      if (measureDigits && /^#(?:BPMCHANGE|MEASURE|DELAY|BRANCHSTART)\b/i.test(line)) {
        reason ||= "timing_command_inside_measure";
      }
      if (/^#(?:BRANCHSTART|BRANCHEND)\b/i.test(line)) reason ||= "branching";
      else if (/^#DELAY\b/i.test(line)) reason ||= "delay";
      else if (/^#BPMCHANGE\b/i.test(line)) {
        const value = Number(line.replace(/^#BPMCHANGE\s*/i, ""));
        if (Number.isFinite(value) && value > 0) bpm = value;
        else reason ||= "invalid_bpm_change";
      } else if (/^#MEASURE\b/i.test(line)) {
        const match = line.match(/^#MEASURE\s+(\d+)\s*\/\s*(\d+)/i);
        if (!match || Number(match[2]) === 0) {
          reason ||= "invalid_measure";
        } else {
          const nextBeats = Number(match[1]) * 4 / Number(match[2]);
          if (!Number.isFinite(nextBeats) || Math.round(nextBeats * 4) !== nextBeats * 4) {
            reason ||= "unsupported_measure";
          } else {
            beatsPerMeasure = nextBeats;
          }
        }
      }
      continue;
    }

    const noteText = line.split("//", 1)[0];
    for (const character of noteText) {
      if (character >= "0" && character <= "9") measureDigits += character;
      else if (character === ",") finishMeasure();
    }
  }
  finishChart();
  return charts;
}

async function findFiles(directory, extension) {
  const results = [];
  async function visit(current) {
    const entries = await readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== ".tja-gen-training") await visit(fullPath);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(extension)) {
        results.push(fullPath);
      }
    }
  }
  await visit(directory);
  return results.sort((a, b) => a.localeCompare(b));
}

async function hashAndGroupAudio(groups) {
  const byHash = new Map();
  for (const group of groups) {
    const hash = await hashFile(group.audioPath);
    if (!byHash.has(hash)) byHash.set(hash, { ...group, hash });
    else byHash.get(hash).charts.push(...group.charts);
  }
  return [...byHash.values()].sort((a, b) => a.hash.localeCompare(b.hash));
}

function selectGroups(groups, limit) {
  if (!limit || limit >= groups.length) return groups;
  return Array.from({ length: limit }, (_, index) => groups[Math.floor(index * groups.length / limit)]);
}

function hashFile(file) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(file);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

function assignValidationSplit(groups) {
  for (const group of groups) {
    group.validation = Number.parseInt(group.hash.slice(0, 8), 16) % 5 === 0;
  }
  if (!groups.some((group) => group.validation)) groups.at(-1).validation = true;
  if (groups.every((group) => group.validation)) groups[0].validation = false;
}

async function decodeAudio(file) {
  const chunks = [];
  await new Promise((resolve, reject) => {
    const decoder = spawn("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-nostdin", "-i", file,
      "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", String(SAMPLE_RATE),
      "-f", "f32le", "pipe:1"
    ], { windowsHide: true });
    let stderr = "";
    decoder.stdout.on("data", (chunk) => chunks.push(chunk));
    decoder.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    decoder.on("error", (error) => reject(new Error(`ffmpegを起動できません: ${error.message}`)));
    decoder.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`音源をデコードできませんでした（ffmpeg終了コード ${code}）。${stderr.trim()}`));
    });
  });

  const raw = Buffer.concat(chunks);
  const sampleCount = Math.floor(raw.length / 4);
  const frameCount = Math.ceil(sampleCount / FRAME_SAMPLES);
  const rms = new Float32Array(frameCount);
  let sample = 0;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const end = Math.min(sampleCount, sample + FRAME_SAMPLES);
    let energy = 0;
    for (; sample < end; sample += 1) {
      const value = raw.readFloatLE(sample * 4);
      energy += value * value;
    }
    rms[frame] = Math.sqrt(energy / Math.max(1, end - frame * FRAME_SAMPLES));
  }

  const onset = new Float32Array(frameCount);
  for (let index = 1; index < frameCount; index += 1) {
    onset[index] = Math.max(0, rms[index] - rms[index - 1]);
  }
  return {
    duration: sampleCount / SAMPLE_RATE,
    rms,
    onset,
    rmsScale: percentile(rms, 0.95),
    onsetScale: percentile(onset, 0.95)
  };
}

function prepareChart(chart) {
  if (chart.parseReason) return { ok: false, reason: chart.parseReason };
  if (!chart.measures.length) return { ok: false, reason: "empty_chart" };
  const measures = [];
  for (const measure of chart.measures) {
    const baseSteps = Math.round(measure.beats * 12);
    const stepCount = leastCommonMultiple(baseSteps, measure.digits.length);
    if (stepCount < 1 || stepCount > 384) return { ok: false, reason: "unsupported_grid" };
    const labels = new Array(stepCount).fill(0);
    for (let index = 0; index < measure.digits.length; index += 1) {
      const label = Number(measure.digits[index]);
      if (label === 0) continue;
      const position = index / measure.digits.length * stepCount;
      const slot = Math.round(position);
      if (Math.abs(position - slot) > 0.12 || slot >= stepCount) {
        return { ok: false, reason: "off_grid_note" };
      }
      if (labels[slot] !== 0 && labels[slot] !== label) {
        return { ok: false, reason: "colliding_notes" };
      }
      labels[slot] = label;
    }
    measures.push({ ...measure, stepCount, labels });
  }
  return { ok: true, chart, measures };
}

function makeTrainingExamples(prepared, envelope, groupHash, chartIndex) {
  let positiveCount = 0;
  let slotCount = 0;
  for (const measure of prepared.measures) {
    for (const label of measure.labels) {
      if (label !== 0) positiveCount += 1;
      slotCount += 1;
    }
  }
  const negativeCount = slotCount - positiveCount;
  if (!positiveCount) return [];
  const negativeProbability = Math.min(1, positiveCount * 2 / Math.max(1, negativeCount));
  const result = [];
  let slotIndex = 0;
  for (const sample of chartSamples(prepared, envelope)) {
    if (sample.label === 0) {
      const key = `${groupHash}:${chartIndex}:${slotIndex}`;
      if (hashFraction(key) >= negativeProbability) {
        slotIndex += 1;
        continue;
      }
    }
    result.push(sample);
    slotIndex += 1;
  }
  return result;
}

function* chartSamples(prepared, envelope) {
  const chart = prepared.chart;
  for (const measure of prepared.measures) {
    for (let slot = 0; slot < measure.stepCount; slot += 1) {
      const chartTime = measure.start + measure.duration * slot / measure.stepCount;
      const audioTime = chartTime - chart.offset;
      if (audioTime < 0 || audioTime >= envelope.duration) continue;
      yield {
        label: measure.labels[slot],
        features: makeFeatures(envelope, audioTime, slot, measure.stepCount, measure.beats, measure.bpm, chart.level, chart.courseIndex)
      };
    }
  }
}

function makeFeatures(envelope, seconds, slot, stepCount, beatsPerMeasure, bpm, level, courseIndex) {
  const rms = (offset) => normalized(envelope.rms, seconds + offset, envelope.rmsScale);
  const onset = (offset) => normalized(envelope.onset, seconds + offset, envelope.onsetScale);
  const rmsValues = [-0.04, -0.02, 0, 0.02, 0.04];
  const onsetValues = [-0.05, -0.02, 0, 0.02, 0.05];
  const rmsSamples = rmsValues.map(rms);
  const onsetSamples = onsetValues.map(onset);
  const measurePhase = (slot / stepCount) * 2 * Math.PI;
  const beatPosition = slot / stepCount * beatsPerMeasure;
  const beatPhase = (beatPosition - Math.floor(beatPosition)) * 2 * Math.PI;
  const features = [
    rmsSamples[1], rmsSamples[2], rmsSamples[3],
    Math.max(...rmsSamples),
    rmsSamples.reduce((sum, value) => sum + value, 0) / rmsSamples.length,
    onsetSamples[1], onsetSamples[2], onsetSamples[3],
    Math.max(...onsetSamples),
    onsetSamples.reduce((sum, value) => sum + value, 0) / onsetSamples.length,
    Math.sin(measurePhase), Math.cos(measurePhase),
    Math.sin(beatPhase), Math.cos(beatPhase),
    Math.min(1, bpm / 400), Math.min(1, level / 10)
  ];
  for (let index = 0; index < 5; index += 1) features.push(index === courseIndex ? 1 : 0);
  return features;
}

function normalized(frames, seconds, scale) {
  const index = Math.max(0, Math.min(frames.length - 1, Math.round(seconds / FRAME_SECONDS)));
  if (!frames.length || !scale) return 0;
  return Math.min(1, frames[index] / scale);
}

function createModel(featureCount) {
  return Array.from({ length: CLASS_COUNT }, () => new Float64Array(featureCount + 1));
}

function makeClassWeights(counts) {
  const maxPositive = Math.max(1, ...counts.slice(1));
  return counts.map((count, index) => index === 0 ? 1 : Math.min(4, Math.sqrt(maxPositive / Math.max(1, count))));
}

function trainExample(model, features, label, classWeights) {
  const logits = model.map((weights) => {
    let value = weights[features.length];
    for (let index = 0; index < features.length; index += 1) value += weights[index] * features[index];
    return value;
  });
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const normalizer = exponentials.reduce((sum, value) => sum + value, 0);
  const learningRate = 0.025 / (1 + updateCount / 250000);
  for (let category = 0; category < CLASS_COUNT; category += 1) {
    const error = (exponentials[category] / normalizer - (category === label ? 1 : 0)) * classWeights[category];
    const weights = model[category];
    for (let index = 0; index < features.length; index += 1) {
      weights[index] -= learningRate * (error * features[index] + 0.0001 * weights[index]);
    }
    weights[features.length] -= learningRate * error;
  }
  updateCount += 1;
}

function predict(model, features) {
  let bestClass = 0;
  let bestScore = -Infinity;
  for (let category = 0; category < CLASS_COUNT; category += 1) {
    let score = model[category][features.length];
    for (let index = 0; index < features.length; index += 1) {
      score += model[category][index] * features[index];
    }
    if (score > bestScore) {
      bestScore = score;
      bestClass = category;
    }
  }
  return bestClass;
}

function calculateMetrics(confusion) {
  const total = confusion.flat().reduce((sum, count) => sum + count, 0);
  const correct = confusion.reduce((sum, row, index) => sum + row[index], 0);
  const scores = confusion.map((row, index) => {
    const tp = row[index];
    const actual = row.reduce((sum, count) => sum + count, 0);
    const predicted = confusion.reduce((sum, values) => sum + values[index], 0);
    const precision = tp / Math.max(1, predicted);
    const recall = tp / Math.max(1, actual);
    return { support: actual, precision, recall, f1: 2 * precision * recall / Math.max(1e-12, precision + recall) };
  });
  const noteActual = total - confusion[0].reduce((sum, count) => sum + count, 0);
  const notePredicted = total - confusion.reduce((sum, row) => sum + row[0], 0);
  const noteTruePositive = confusion.slice(1).reduce((sum, row) => sum + row.slice(1).reduce((inner, count) => inner + count, 0), 0);
  const notePrecision = noteTruePositive / Math.max(1, notePredicted);
  const noteRecall = noteTruePositive / Math.max(1, noteActual);
  return {
    accuracy: correct / Math.max(1, total),
    notePresence: {
      precision: notePrecision,
      recall: noteRecall,
      f1: 2 * notePrecision * noteRecall / Math.max(1e-12, notePrecision + noteRecall)
    },
    classMetrics: scores,
    confusionMatrix: confusion
  };
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] ?? 0;
}

function leastCommonMultiple(left, right) {
  const gcd = (a, b) => b ? gcd(b, a % b) : a;
  return Math.abs(left * right) / gcd(left, right);
}

function countUnsupported(report, reason) {
  report.unsupported[reason] = (report.unsupported[reason] || 0) + 1;
}

function hashFraction(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 0x100000000;
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [values[index], values[target]] = [values[target], values[index]];
  }
}

function printProgress(label, current, total) {
  process.stdout.write(`\r${label}: ${current}/${total}`);
}

function parseArgs(args) {
  const result = { dataDir: "", limit: null, help: false };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") result.help = true;
    else if (arg === "--data-dir") result.dataDir = args[++index] || "";
    else if (arg === "--limit") {
      const limit = Number(args[++index]);
      if (!Number.isInteger(limit) || limit < 3) throw new Error("--limitには3以上の整数を指定してください。");
      result.limit = limit;
    } else {
      throw new Error(`不明なオプションです: ${arg}`);
    }
  }
  if (!result.help && !result.dataDir) throw new Error("--data-dirでTJAデータのフォルダーを指定してください。");
  return result;
}

function printHelp() {
  console.log("Usage: node tools/train-chart-model.mjs --data-dir <TJAフォルダー> [--limit <音源数>]");
  console.log("音源はローカルのffmpegで解析し、学習結果はデータフォルダー内に保存します。");
}
