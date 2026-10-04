#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const EXPECTED_FEATURES = [
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

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const defaultInput = path.resolve(scriptDirectory, "../../TJA-Gen-data/.tja-gen-training/model.json");
const inputPath = path.resolve(process.argv[2] || defaultInput);
const outputPath = path.resolve(process.argv[3] || path.join(scriptDirectory, "../trained-model.js"));

try {
  const model = JSON.parse(await readFile(inputPath, "utf8"));
  validateModel(model);
  const reportPath = path.join(path.dirname(inputPath), "report.json");
  let evaluation = null;
  try {
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    evaluation = {
      audioGroups: report.audioGroups,
      validationAudioGroups: report.validationAudioGroups,
      noteF1: report.metrics?.notePresence?.f1 ?? null
    };
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }

  const artifact = {
    formatVersion: model.formatVersion,
    classes: model.classes,
    features: model.features,
    weights: model.weights,
    evaluation
  };
  await writeFile(outputPath, `window.TJAChartModel = ${JSON.stringify(artifact)};\n`, "utf8");
  console.log(`Web用モデルを書き出しました: ${outputPath}`);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}

function validateModel(model) {
  if (model.formatVersion !== 1 || model.classes !== 10) {
    throw new Error("未対応のモデル形式です。formatVersion 1 / classes 10 が必要です。");
  }
  if (JSON.stringify(model.features) !== JSON.stringify(EXPECTED_FEATURES)) {
    throw new Error("モデルの特徴量がWebアプリと一致しません。");
  }
  if (!Array.isArray(model.weights) || model.weights.length !== model.classes) {
    throw new Error("モデルの重みのクラス数が不正です。");
  }
  if (model.weights.some((row) => !Array.isArray(row) || row.length !== model.features.length + 1
    || row.some((value) => !Number.isFinite(value)))) {
    throw new Error("モデルの重みの形状または数値が不正です。");
  }
}
