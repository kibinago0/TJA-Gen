#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { access, mkdir, readFile, readdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { TextDecoder } from "node:util";
import { fileURLToPath } from "node:url";

const SAMPLE_RATE = 11025;
const FRAME_SAMPLES = 110;
const FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE;
const CLASS_COUNT = 10;
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
const trainingDataPath = path.join(outputDir, "training-samples.bin");
const validationDataPath = path.join(outputDir, "validation-samples.bin");
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
  const densityByCourse = new Map();
  for (const group of hashedGroups) {
    for (const chart of group.charts) {
      const prepared = prepareChart(chart);
      if (!prepared.ok) {
        countUnsupported(summary, prepared.reason);
        continue;
      }
      const totalDuration = prepared.measures.reduce((sum, measure) => sum + measure.duration, 0);
      const totalNotes = prepared.measures.reduce(
        (sum, measure) => sum + measure.labels.filter((label) => label !== 0).length,
        0
      );
      const courseName = ["Easy", "Normal", "Hard", "Oni", "Edit"][chart.courseIndex];
      if (!densityByCourse.has(courseName)) densityByCourse.set(courseName, []);
      densityByCourse.get(courseName).push(totalNotes / Math.max(totalDuration, 1e-8));
      if (group.validation) continue;
      for (const measure of prepared.measures) {
        for (const label of measure.labels) classCounts[label] += 1;
      }
    }
  }

  await mkdir(outputDir, { recursive: true });
  const trainingStream = createWriteStream(trainingDataPath);
  const validationStream = createWriteStream(validationDataPath);
  let trainedCharts = 0;
  let trainingExamples = 0;
  let validationCharts = 0;
  let validationSlots = 0;
  let processed = 0;
  try {
    for (const group of hashedGroups) {
      const envelope = await decodeAudio(group.audioPath);
      let groupChartIndex = 0;
      for (const chart of group.charts) {
        const prepared = prepareChart(chart);
        if (prepared.ok) {
          if (group.validation) {
            const examples = Array.from(chartSamples(prepared, envelope));
            if (examples.length) {
              await writeExamples(validationStream, examples);
              validationCharts += 1;
              validationSlots += examples.length;
            }
          } else {
            const examples = makeTrainingExamples(prepared, envelope, group.hash, groupChartIndex);
            if (examples.length) {
              await writeExamples(trainingStream, examples);
              trainingExamples += examples.length;
              trainedCharts += 1;
            }
          }
        }
        groupChartIndex += 1;
      }
      processed += 1;
      if (processed % 25 === 0 || processed === hashedGroups.length) {
        printProgress("音源解析", processed, hashedGroups.length);
      }
    }
    await Promise.all([endStream(trainingStream), endStream(validationStream)]);
  } catch (error) {
    trainingStream.destroy();
    validationStream.destroy();
    throw error;
  }
  console.log();

  if (!trainingExamples || !validationSlots) {
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
  summary.trainingExamplesPerEpoch = trainingExamples;
  summary.validationSlots = validationSlots;
  summary.noteClassCounts = classCounts;
  summary.densityByCourse = Object.fromEntries(
    [...densityByCourse.entries()].sort(([left], [right]) => left.localeCompare(right))
      .map(([course, values]) => [course, summarize(values)])
  );
  summary.trainingEpochs = options.epochs;
  summary.model = `numpy minibatch multiclass logistic regression (${options.epochs} epochs; sampled empty slots)`;
  summary.audioFeatures = "10 ms mono RMS and positive RMS differences";
  summary.split = "SHA-256 audio-content groups; deterministic 20% holdout";
  summary.offsetRule = "audio time = chart time - TJA OFFSET";

  const modelPath = path.join(outputDir, "model.json");
  const metricsPath = path.join(outputDir, "metrics.json");
  await runPythonTrainer({
    python: options.python,
    script: path.join(path.dirname(fileURLToPath(import.meta.url)), "fit-chart-model.py"),
    trainPath: trainingDataPath,
    validationPath: validationDataPath,
    modelPath,
    metricsPath,
    epochs: options.epochs,
    featureCount: FEATURE_NAMES.length,
    classCount: CLASS_COUNT,
    classCounts
  });
  summary.metrics = JSON.parse(await readFile(metricsPath, "utf8"));
  await writeFile(path.join(outputDir, "report.json"), JSON.stringify(summary, null, 2), "utf8");
  await Promise.all([unlink(trainingDataPath), unlink(validationDataPath), unlink(metricsPath)]);

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

async function writeExamples(stream, examples) {
  const recordSize = FEATURE_NAMES.length * 4 + 1;
  const buffer = Buffer.allocUnsafe(examples.length * recordSize);
  let offset = 0;
  for (const example of examples) {
    for (const feature of example.features) {
      buffer.writeFloatLE(feature, offset);
      offset += 4;
    }
    buffer.writeUInt8(example.label, offset);
    offset += 1;
  }
  if (!stream.write(buffer)) await once(stream, "drain");
}

async function endStream(stream) {
  stream.end();
  await finished(stream);
}

function runPythonTrainer(options) {
  return new Promise((resolve, reject) => {
    const args = [
      options.script,
      "--train", options.trainPath,
      "--validation", options.validationPath,
      "--model", options.modelPath,
      "--metrics", options.metricsPath,
      "--epochs", String(options.epochs),
      "--feature-count", String(options.featureCount),
      "--class-count", String(options.classCount),
      "--class-counts", options.classCounts.join(",")
    ];
    const trainer = spawn(options.python, args, {
      windowsHide: true,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" }
    });
    trainer.stdout.on("data", (chunk) => process.stdout.write(chunk));
    trainer.stderr.on("data", (chunk) => process.stderr.write(chunk));
    trainer.on("error", (error) => reject(new Error(`Python学習処理を起動できません: ${error.message}`)));
    trainer.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Python学習処理が終了コード ${code} で失敗しました。Python 3 と NumPy が必要です。`));
    });
  });
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
  const negativeProbability = Math.min(1, positiveCount * 0.5 / Math.max(1, negativeCount));
  const result = [];
  let slotIndex = 0;
  for (const measure of prepared.measures) {
    for (let slot = 0; slot < measure.stepCount; slot += 1) {
      const label = measure.labels[slot];
      if (label === 0) {
        const key = `${groupHash}:${chartIndex}:${slotIndex}`;
        if (hashFraction(key) >= negativeProbability) {
          slotIndex += 1;
          continue;
        }
      }
      const chartTime = measure.start + measure.duration * slot / measure.stepCount;
      const audioTime = chartTime - prepared.chart.offset;
      if (audioTime >= 0 && audioTime < envelope.duration) {
        result.push({
          label,
          features: makeFeatures(envelope, audioTime, slot, measure.stepCount, measure.beats, measure.bpm, prepared.chart.level, prepared.chart.courseIndex)
        });
      }
      slotIndex += 1;
    }
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

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] ?? 0;
}

function summarize(values) {
  if (!values.length) return { chartCount: 0 };
  const sorted = [...values].sort((left, right) => left - right);
  const quantile = (fraction) => sorted[Math.floor((sorted.length - 1) * fraction)];
  return {
    chartCount: sorted.length,
    medianNotesPerSecond: quantile(0.5),
    p75NotesPerSecond: quantile(0.75),
    p90NotesPerSecond: quantile(0.9)
  };
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

function printProgress(label, current, total) {
  process.stdout.write(`\r${label}: ${current}/${total}`);
}

function parseArgs(args) {
  const result = { dataDir: "", limit: null, epochs: 3, python: process.env.PYTHON || "python", help: false };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") result.help = true;
    else if (arg === "--data-dir") result.dataDir = args[++index] || "";
    else if (arg === "--limit") {
      const limit = Number(args[++index]);
      if (!Number.isInteger(limit) || limit < 3) throw new Error("--limitには3以上の整数を指定してください。");
      result.limit = limit;
    } else if (arg === "--python") {
      result.python = args[++index] || "";
      if (!result.python) throw new Error("--pythonにはPython実行ファイルのパスを指定してください。");
    } else if (arg === "--epochs") {
      const epochs = Number(args[++index]);
      if (!Number.isInteger(epochs) || epochs < 1 || epochs > 20) {
        throw new Error("--epochsには1から20の整数を指定してください。");
      }
      result.epochs = epochs;
    } else {
      throw new Error(`不明なオプションです: ${arg}`);
    }
  }
  if (!result.help && !result.dataDir) throw new Error("--data-dirでTJAデータのフォルダーを指定してください。");
  return result;
}

function printHelp() {
  console.log("Usage: node tools/train-chart-model.mjs --data-dir <TJAフォルダー> [--limit <音源数>] [--epochs <1-20>] [--python <Python実行ファイル>]");
  console.log("音源はローカルのffmpegで解析し、NumPyで学習します。学習結果はデータフォルダー内に保存します。");
}
