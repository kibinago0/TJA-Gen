const form = document.querySelector("#generator-form");
const audioInput = document.querySelector("#audio-file");
const uploadArea = document.querySelector("#upload-area");
const fileName = document.querySelector("#file-name");
const generateButton = document.querySelector("#generate-button");
const errorMessage = document.querySelector("#error-message");
const resultCard = document.querySelector("#result-card");
const algorithmInput = document.querySelector("#algorithm");
const bpmInput = document.querySelector("#bpm");
const bpmStatus = document.querySelector("#bpm-status");

let generatedTja = "";
let generatedFileName = "chart.tja";
let decodedFile = null;
let decodedAudioBuffer = null;
let audioLoadId = 0;
const modelCourseIndex = { Easy: 0, Normal: 1, Hard: 2, Oni: 3, Edit: 4 };

bpmInput.addEventListener("input", () => {
  if (decodedFile) bpmStatus.textContent = "手動設定したBPMを使用します。";
});

audioInput.addEventListener("change", async () => {
  const file = audioInput.files?.[0];
  fileName.textContent = file?.name ?? "音源を選択、またはここにドロップ";
  resultCard.hidden = true;
  errorMessage.hidden = true;
  decodedFile = null;
  decodedAudioBuffer = null;
  bpmInput.value = "";
  const loadId = ++audioLoadId;
  if (!file) {
    bpmStatus.textContent = "音源を選ぶとBPMを推定します。";
    return;
  }
  bpmStatus.textContent = "音源を解析してBPMを検出中…";
  try {
    const audioBuffer = await decodeAudio(file);
    if (loadId !== audioLoadId) return;
    decodedFile = file;
    decodedAudioBuffer = audioBuffer;
    const estimate = estimateBpm(createModelEnvelope(audioBuffer).onset);
    bpmInput.value = estimate.bpm.toFixed(1);
    bpmStatus.textContent = `自動推定: ${estimate.bpm.toFixed(1)} BPM（必要なら修正できます）`;
  } catch (error) {
    if (loadId !== audioLoadId) return;
    bpmStatus.textContent = "BPMを自動検出できませんでした。BPMを手入力してください。";
    showError(error instanceof Error ? error.message : "音源を読み込めませんでした。");
  }
});

uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadArea.classList.add("is-dragging");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("is-dragging");
});

uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadArea.classList.remove("is-dragging");
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  audioInput.files = transfer.files;
  audioInput.dispatchEvent(new Event("change", { bubbles: true }));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  resultCard.hidden = true;

  const file = audioInput.files?.[0];
  const bpm = Number(form.elements.bpm.value);
  const targetRate = Number(form.elements["notes-per-second"].value);
  const measureResolution = Number(form.elements["max-notes-per-measure"].value);
  if (!file) {
    showError("譜面にする音源ファイルを選択してください。");
    return;
  }
  if (!Number.isFinite(bpm) || bpm < 30 || bpm > 300) {
    showError("BPMを30から300の範囲で入力してください。自動検出できない場合は手入力してください。");
    return;
  }
  if (!Number.isFinite(targetRate) || targetRate < 0.1 || targetRate > 128) {
    showError("目標密度は0.1から128打/秒の範囲で入力してください。");
    return;
  }
  if (!Number.isInteger(measureResolution) || measureResolution < 1 || measureResolution > 256) {
    showError("1小節の分割数は1から256の整数で入力してください。");
    return;
  }

  generateButton.disabled = true;
  generateButton.querySelector("span").textContent = "譜面を生成中…";
  try {
    const decodedAudio = decodedFile === file && decodedAudioBuffer
      ? decodedAudioBuffer
      : await decodeAudio(file);
    decodedFile = file;
    decodedAudioBuffer = decodedAudio;
    const course = form.elements.course.value;
    const algorithm = algorithmInput.value;
    const result = generateChart(decodedAudio, bpm, targetRate, measureResolution, course, algorithm);
    const title = cleanHeaderValue(form.elements.title.value.trim() || stripExtension(file.name));
    const wave = cleanHeaderValue(file.name);
    const estimatedLevel = estimatedLevelFromRate(targetRate);
    const tja = buildTja({ title, wave, bpm, course, level: estimatedLevel, result });

    generatedTja = tja;
    generatedFileName = `${safeFileName(title)}.tja`;
    document.querySelector("#result-title").textContent = title;
    document.querySelector("#result-notes").textContent = result.noteCount.toLocaleString("ja-JP");
    document.querySelector("#result-measures").textContent = result.measures.length.toLocaleString("ja-JP");
    document.querySelector("#result-density").textContent = `${result.actualRate.toFixed(1)} 打/秒`;
    document.querySelector("#result-course").textContent = `${courseLabel(course)} / ★${estimatedLevel}`;
    document.querySelector("#result-method").textContent = algorithm === "learned"
      ? `学習モデルによる生成（実験版・目標 ${targetRate} 打/秒）`
      : `音量ベースによる生成（目標 ${targetRate} 打/秒）`;
    resultCard.hidden = false;
  } catch (error) {
    showError(error instanceof Error ? error.message : "音源を解析できませんでした。別の音源をお試しください。");
  } finally {
    generateButton.disabled = false;
    generateButton.querySelector("span").textContent = "譜面を生成する";
  }
});

document.querySelector("#download-button").addEventListener("click", () => {
  if (!generatedTja) return;
  const blob = new Blob([generatedTja], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = generatedFileName;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function decodeAudio(file) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("このブラウザーは音源解析に対応していません。最新版のChrome、Edge、Firefoxをお試しください。");
  }
  const context = new AudioContextClass();
  return file.arrayBuffer()
    .then((buffer) => context.decodeAudioData(buffer))
    .catch((error) => {
      if (error instanceof Error && error.message.startsWith("このブラウザー")) throw error;
      throw new Error("音源を読み込めませんでした。対応形式の音源を選択してください。");
    })
    .finally(() => context.close());
}

function createModelEnvelope(audioBuffer) {
  const targetSampleRate = 11025;
  const frameSize = 110;
  const targetSampleCount = Math.ceil(audioBuffer.duration * targetSampleRate);
  const frameCount = Math.ceil(targetSampleCount / frameSize);
  const rms = new Float32Array(frameCount);
  const channels = Array.from(
    { length: audioBuffer.numberOfChannels },
    (_, channelIndex) => audioBuffer.getChannelData(channelIndex)
  );
  const sourceRateRatio = audioBuffer.sampleRate / targetSampleRate;
  let targetSample = 0;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const frameEnd = Math.min(targetSampleCount, targetSample + frameSize);
    let energy = 0;
    for (; targetSample < frameEnd; targetSample += 1) {
      const sourcePosition = targetSample * sourceRateRatio;
      const sourceIndex = Math.min(audioBuffer.length - 1, Math.floor(sourcePosition));
      const nextIndex = Math.min(audioBuffer.length - 1, sourceIndex + 1);
      const fraction = sourcePosition - sourceIndex;
      let monoSample = 0;
      for (const channel of channels) {
        monoSample += channel[sourceIndex] * (1 - fraction) + channel[nextIndex] * fraction;
      }
      monoSample /= channels.length;
      energy += monoSample * monoSample;
    }
    rms[frame] = Math.sqrt(energy / Math.max(1, frameEnd - frame * frameSize));
  }

  const onset = new Float32Array(frameCount);
  for (let frame = 1; frame < frameCount; frame += 1) {
    onset[frame] = Math.max(0, rms[frame] - rms[frame - 1]);
  }
  return {
    duration: audioBuffer.duration,
    frameSeconds: frameSize / targetSampleRate,
    rms,
    onset,
    rmsScale: percentile(rms, 0.95),
    onsetScale: percentile(onset, 0.95)
  };
}

function generateChart(audioBuffer, bpm, targetRate, measureResolution, course, algorithm) {
  const envelope = createModelEnvelope(audioBuffer);
  if (envelope.rmsScale < 0.00001) {
    throw new Error("音源から音を検出できませんでした。音量のある音源をお試しください。");
  }
  const level = estimatedLevelFromRate(targetRate);
  const model = algorithm === "learned" ? window.TJAChartModel : null;
  if (algorithm === "learned" && (!model || model.formatVersion !== 1 || model.weights?.length !== 10)) {
    throw new Error("学習モデルを読み込めません。trained-model.js を確認するか、音量ベース方式を選んでください。");
  }
  const courseIndex = modelCourseIndex[course];
  if (algorithm === "learned" && courseIndex === undefined) {
    throw new Error("選択したコースは学習モデルに対応していません。");
  }

  const measureDuration = 240 / bpm;
  const measureCount = Math.max(1, Math.ceil(audioBuffer.duration / measureDuration));
  const measureNoteCounts = allocateMeasureNoteCounts(envelope, measureCount, measureDuration, targetRate, measureResolution);
  const measures = [];
  const measurePatterns = [];
  let noteCount = 0;
  for (let measureIndex = 0; measureIndex < measureCount; measureIndex += 1) {
    const measureStart = measureIndex * measureDuration;
    const targetCount = measureNoteCounts[measureIndex];
    const candidates = [];
    for (let slot = 0; slot < measureResolution; slot += 1) {
      const time = measureStart + measureDuration * slot / measureResolution;
      if (time >= audioBuffer.duration) continue;
      const candidate = algorithm === "learned"
        ? scoreWithModel(model, envelope, time, slot, measureResolution, bpm, level, courseIndex)
        : scoreByAmplitude(envelope, time, slot, measureResolution);
      candidates.push({ ...candidate, slot });
    }
    const phraseReferenceIndex = measureIndex % 4 === 0 ? measureIndex - 4 : measureIndex - 1;
    const phraseReference = phraseReferenceIndex >= 0 ? measurePatterns[phraseReferenceIndex] : null;
    const rhythmProfile = model?.rhythm?.courses?.[course] ?? null;
    const selected = selectHumanizedNotes(
      candidates,
      targetCount,
      measureResolution,
      rhythmProfile,
      phraseReference,
      measureIndex % 4 === 0
    );
    measurePatterns.push(new Set(selected.keys()));
    let row = "";
    for (let slot = 0; slot < measureResolution; slot += 1) {
      row += String(selected.get(slot) ?? 0);
    }
    noteCount += selected.size;
    measures.push(row);
  }
  if (noteCount === 0) {
    throw new Error("目標密度が低すぎるため音符がありません。打/秒を上げてください。");
  }
  return { measures, noteCount, actualRate: noteCount / audioBuffer.duration };
}

function allocateMeasureNoteCounts(envelope, measureCount, measureDuration, targetRate, measureResolution) {
  const capacities = [];
  const activities = [];
  for (let measure = 0; measure < measureCount; measure += 1) {
    const start = measure * measureDuration;
    const duration = Math.min(measureDuration, envelope.duration - start);
    const firstFrame = Math.max(0, Math.floor(start / envelope.frameSeconds));
    const lastFrame = Math.min(envelope.rms.length, Math.ceil((start + duration) / envelope.frameSeconds));
    let energy = 0;
    for (let frame = firstFrame; frame < lastFrame; frame += 1) {
      energy += envelope.rms[frame] * envelope.rms[frame];
    }
    const meanRms = Math.sqrt(energy / Math.max(1, lastFrame - firstFrame));
    capacities.push(Math.min(measureResolution, Math.ceil(duration / measureDuration * measureResolution)));
    activities.push(meanRms);
  }

  const totalCapacity = capacities.reduce((sum, capacity) => sum + capacity, 0);
  let remaining = Math.min(totalCapacity, Math.round(targetRate * envelope.duration));
  const counts = new Array(measureCount).fill(0);
  const maxActivity = Math.max(...activities, 1e-8);
  const weights = activities.map((activity) => 0.65 + 0.7 * Math.sqrt(activity / maxActivity));

  while (remaining > 0) {
    const active = capacities.map((capacity, index) => index).filter((index) => counts[index] < capacities[index]);
    const totalWeight = active.reduce((sum, index) => sum + weights[index], 0);
    const quotas = active.map((index) => ({
      index,
      exact: remaining * weights[index] / totalWeight
    }));
    let assigned = 0;
    for (const quota of quotas) {
      const add = Math.min(
        capacities[quota.index] - counts[quota.index],
        Math.floor(quota.exact)
      );
      if (add > 0) {
        counts[quota.index] += add;
        assigned += add;
      }
    }
    remaining -= assigned;
    if (remaining === 0) break;
    const ranked = quotas
      .filter(({ index }) => counts[index] < capacities[index])
      .sort((left, right) => (right.exact - Math.floor(right.exact)) - (left.exact - Math.floor(left.exact)));
    if (!ranked.length) break;
    for (const quota of ranked) {
      if (remaining === 0) break;
      counts[quota.index] += 1;
      remaining -= 1;
    }
  }
  return counts;
}

function selectHumanizedNotes(candidates, targetCount, measureResolution, profile, phraseReference, reprise) {
  if (targetCount <= 0 || !candidates.length) return new Map();
  if (targetCount >= candidates.length) {
    return new Map(candidates.map((candidate) => [candidate.slot, candidate.note]));
  }

  const scoreMean = candidates.reduce((sum, candidate) => sum + candidate.score, 0) / candidates.length;
  const scoreVariance = candidates.reduce((sum, candidate) => sum + (candidate.score - scoreMean) ** 2, 0) / candidates.length;
  const scoreScale = Math.max(0.0001, Math.sqrt(scoreVariance));
  const phaseCounts = profile?.phaseCounts ?? [];
  const phaseTotal = phaseCounts.reduce((sum, value) => sum + value, 0);
  const phaseScores = candidates.map((candidate) => {
    const phase = Math.min(15, Math.floor(candidate.slot / measureResolution * 16));
    const phaseProbability = phaseTotal
      ? (phaseCounts[phase] + 1) / (phaseTotal + 16)
      : defaultPhaseProbability(phase);
    return 0.3 * Math.log(phaseProbability * 16);
  });
  const noteScores = candidates.map((candidate, index) => {
    const acoustic = Math.max(-3, Math.min(3, (candidate.score - scoreMean) / scoreScale));
    const matchingPhraseSlot = phraseReference?.has(candidate.slot) ?? false;
    const phraseVariation = matchingPhraseSlot ? (reprise ? 0.12 : -0.2) : 0;
    return acoustic + phaseScores[index] + phraseVariation;
  });

  let previous = new Float64Array(candidates.length).fill(-Infinity);
  for (let index = 0; index < candidates.length; index += 1) {
    previous[index] = noteScores[index];
  }
  const backPointers = Array.from(
    { length: targetCount },
    () => new Int16Array(candidates.length).fill(-1)
  );

  for (let count = 2; count <= targetCount; count += 1) {
    const current = new Float64Array(candidates.length).fill(-Infinity);
    for (let right = count - 1; right < candidates.length; right += 1) {
      for (let left = count - 2; left < right; left += 1) {
        if (!Number.isFinite(previous[left])) continue;
        const gapUnits = (candidates[right].slot - candidates[left].slot) / measureResolution * 16;
        const gapScore = rhythmGapScore(profile, gapUnits);
        const transitionScore = rhythmTransitionScore(profile, candidates[left].note, candidates[right].note);
        const score = previous[left] + 0.55 * gapScore + 0.25 * transitionScore;
        if (score > current[right]) {
          current[right] = score;
          backPointers[count - 1][right] = left;
        }
      }
      current[right] += noteScores[right];
    }
    previous = current;
  }

  let last = -1;
  let bestScore = -Infinity;
  for (let index = targetCount - 1; index < candidates.length; index += 1) {
    if (previous[index] > bestScore) {
      bestScore = previous[index];
      last = index;
    }
  }
  const selected = new Map();
  for (let count = targetCount - 1; count >= 0; count -= 1) {
    if (last < 0) throw new Error("リズムに沿った音符配置を決められませんでした。");
    selected.set(candidates[last].slot, candidates[last].note);
    last = backPointers[count][last];
  }
  return selected;
}

function rhythmGapScore(profile, gapUnits) {
  const gapCounts = profile?.gapCounts;
  if (!Array.isArray(gapCounts) || gapCounts.length !== 513) {
    return Math.log(defaultGapProbability(gapUnits) * 64);
  }
  const bucket = Math.max(1, Math.min(512, Math.round(gapUnits * 8)));
  const count = gapCounts[bucket] + (gapCounts[bucket - 1] ?? 0) * 0.5 + (gapCounts[bucket + 1] ?? 0) * 0.5;
  const total = gapCounts.reduce((sum, value) => sum + value, 0);
  return Math.log((count + 0.5) / (total + 256));
}

function rhythmTransitionScore(profile, previousNote, nextNote) {
  const transitions = profile?.noteTransitions;
  if (!Array.isArray(transitions) || transitions.length !== 25) {
    return previousNote === nextNote ? -0.2 : 0.1;
  }
  const rowStart = previousNote * 5;
  const rowTotal = transitions.slice(rowStart + 1, rowStart + 5).reduce((sum, value) => sum + value, 0);
  const count = transitions[rowStart + nextNote] ?? 0;
  return Math.log((count + 1) / (rowTotal + 4)) * 0.25;
}

function defaultPhaseProbability(phase) {
  return [0, 4, 8, 12].includes(phase) ? 0.11 : [2, 6, 10, 14].includes(phase) ? 0.08 : 0.06;
}

function defaultGapProbability(gapUnits) {
  const favored = [
    [1, 0.4], [2, 0.8], [3, 0.35], [4, 1], [6, 0.45],
    [8, 0.75], [12, 0.4], [16, 0.65], [24, 0.3], [32, 0.5]
  ];
  return favored.reduce((sum, [gap, weight]) => sum + weight * Math.exp(-Math.abs(gapUnits - gap) / 0.5), 0) / 5.6;
}

function estimateBpm(onset) {
  const frameSeconds = 110 / 11025;
  const halfWindow = 50;
  const prefix = new Float64Array(onset.length + 1);
  for (let index = 0; index < onset.length; index += 1) {
    prefix[index + 1] = prefix[index] + onset[index];
  }
  const centered = new Float64Array(onset.length);
  for (let index = 0; index < onset.length; index += 1) {
    const start = Math.max(0, index - halfWindow);
    const end = Math.min(onset.length, index + halfWindow + 1);
    const localMean = (prefix[end] - prefix[start]) / (end - start);
    centered[index] = Math.max(0, onset[index] - localMean);
  }

  const maxLag = Math.min(Math.floor(60 / (30 * frameSeconds)), Math.floor(onset.length / 3));
  const minLag = Math.max(2, Math.floor(60 / (300 * frameSeconds)));
  const correlations = new Float32Array(maxLag + 1);
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let dot = 0;
    let leftEnergy = 0;
    let rightEnergy = 0;
    for (let index = 0; index + lag < centered.length; index += 1) {
      const left = centered[index];
      const right = centered[index + lag];
      dot += left * right;
      leftEnergy += left * left;
      rightEnergy += right * right;
    }
    correlations[lag] = dot / Math.sqrt(Math.max(1e-12, leftEnergy * rightEnergy));
  }

  let bestLag = 0;
  let bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    const score = correlations[lag]
      + 0.35 * (correlations[lag * 2] ?? 0)
      + 0.15 * (correlations[lag * 3] ?? 0)
      + 0.2 * (correlations[Math.round(lag / 2)] ?? 0);
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }
  if (!bestLag || bestScore < 0.015) {
    throw new Error("BPMを自動検出できませんでした。BPMを手入力してください。");
  }
  return {
    bpm: Math.round(60 / (bestLag * frameSeconds) * 10) / 10,
    confidence: bestScore
  };
}

function scoreByAmplitude(envelope, seconds, slot, measureResolution) {
  const center = Math.round(seconds / envelope.frameSeconds);
  const radius = Math.max(1, Math.round(0.035 / envelope.frameSeconds));
  let peak = 0;
  for (let frame = Math.max(0, center - radius); frame <= Math.min(envelope.onset.length - 1, center + radius); frame += 1) {
    peak = Math.max(peak, envelope.onset[frame] / Math.max(envelope.onsetScale, 1e-8));
  }
  const beatFraction = (slot / measureResolution * 4) % 1;
  const isKat = beatFraction >= 0.5;
  const isBig = peak >= 0.75;
  return { score: peak, note: isBig ? (isKat ? 4 : 3) : (isKat ? 2 : 1) };
}

function scoreWithModel(model, envelope, seconds, slot, measureResolution, bpm, level, courseIndex) {
  const features = makeHighResolutionFeatures(envelope, seconds, slot, measureResolution, bpm, level, courseIndex);
  const scores = [];
  for (let note = 0; note <= 4; note += 1) {
    const weights = model.weights[note];
    if (!Array.isArray(weights) || weights.length !== features.length + 1) {
      throw new Error("学習モデルの重みの形式が正しくありません。");
    }
    let score = weights[features.length];
    for (let index = 0; index < features.length; index += 1) {
      score += weights[index] * features[index];
    }
    scores.push(score);
  }
  let bestNote = 1;
  for (let note = 2; note <= 4; note += 1) {
    if (scores[note] > scores[bestNote]) bestNote = note;
  }
  const maxNoteScore = Math.max(...scores.slice(1));
  const logSumExpNotes = maxNoteScore
    + Math.log(scores.slice(1).reduce((sum, score) => sum + Math.exp(score - maxNoteScore), 0));
  return { score: logSumExpNotes - scores[0], note: bestNote };
}

function makeHighResolutionFeatures(envelope, seconds, slot, measureResolution, bpm, level, courseIndex) {
  const rmsValues = [-0.04, -0.02, 0, 0.02, 0.04]
    .map((offset) => normalizedFrame(envelope.rms, seconds + offset, envelope.rmsScale, envelope.frameSeconds));
  const onsetValues = [-0.05, -0.02, 0, 0.02, 0.05]
    .map((offset) => normalizedFrame(envelope.onset, seconds + offset, envelope.onsetScale, envelope.frameSeconds));
  const measurePhase = slot / measureResolution * 2 * Math.PI;
  const beatPosition = slot / measureResolution * 4;
  const beatPhase = (beatPosition - Math.floor(beatPosition)) * 2 * Math.PI;
  const features = [
    rmsValues[1], rmsValues[2], rmsValues[3],
    Math.max(...rmsValues),
    rmsValues.reduce((sum, value) => sum + value, 0) / rmsValues.length,
    onsetValues[1], onsetValues[2], onsetValues[3],
    Math.max(...onsetValues),
    onsetValues.reduce((sum, value) => sum + value, 0) / onsetValues.length,
    Math.sin(measurePhase), Math.cos(measurePhase),
    Math.sin(beatPhase), Math.cos(beatPhase),
    Math.min(1, bpm / 400), Math.min(1, level / 10)
  ];
  for (let index = 0; index < 5; index += 1) features.push(index === courseIndex ? 1 : 0);
  return features;
}

function normalizedFrame(frames, seconds, scale, frameSeconds) {
  if (!frames.length || !scale) return 0;
  const index = Math.max(0, Math.min(frames.length - 1, Math.round(seconds / frameSeconds)));
  return Math.min(1, frames[index] / scale);
}

function estimatedLevelFromRate(rate) {
  return Math.max(1, Math.min(10, Math.round(rate * 1.5)));
}

function percentile(values, fraction) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] ?? 0;
}

function buildTja({ title, wave, bpm, course, level, result }) {
  const lines = [
    `TITLE:${title}`,
    `BPM:${bpm}`,
    "OFFSET:0",
    `WAVE:${wave}`,
    `COURSE:${course}`,
    `LEVEL:${level}`,
    "SCOREINIT:1000",
    "#START",
    ...result.measures.map((measure) => `${measure},`),
    "#END",
    ""
  ];
  return lines.join("\r\n");
}

function cleanHeaderValue(value) {
  return value.replace(/[\r\n]/g, " ").trim();
}

function stripExtension(name) {
  return name.replace(/\.[^.]+$/, "");
}

function safeFileName(value) {
  return value.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").trim() || "chart";
}

function courseLabel(course) {
  return { Oni: "おに", Hard: "むずかしい", Normal: "ふつう", Easy: "かんたん" }[course] ?? course;
}
