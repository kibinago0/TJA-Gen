const form = document.querySelector("#generator-form");
const audioInput = document.querySelector("#audio-file");
const uploadArea = document.querySelector("#upload-area");
const fileName = document.querySelector("#file-name");
const generateButton = document.querySelector("#generate-button");
const errorMessage = document.querySelector("#error-message");
const resultCard = document.querySelector("#result-card");
const levelInput = document.querySelector("#level");
const levelValue = document.querySelector("#level-value");

let generatedTja = "";
let generatedFileName = "chart.tja";

levelInput.addEventListener("input", () => {
  levelValue.value = levelInput.value;
});

audioInput.addEventListener("change", () => {
  const file = audioInput.files?.[0];
  fileName.textContent = file?.name ?? "音源を選択、またはここにドロップ";
  resultCard.hidden = true;
  errorMessage.hidden = true;
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
  const level = Number(form.elements.level.value);
  if (!file) {
    showError("譜面にする音源ファイルを選択してください。");
    return;
  }
  if (!Number.isFinite(bpm) || bpm < 30 || bpm > 300) {
    showError("BPMは30から300の範囲で入力してください。");
    return;
  }

  generateButton.disabled = true;
  generateButton.querySelector("span").textContent = "音源を解析中…";
  try {
    const decodedAudio = await decodeAudio(file);
    const result = generateChart(decodedAudio, bpm, level);
    const title = cleanHeaderValue(form.elements.title.value.trim() || stripExtension(file.name));
    const wave = cleanHeaderValue(file.name);
    const course = form.elements.course.value;
    const tja = buildTja({ title, wave, bpm, course, level, result });

    generatedTja = tja;
    generatedFileName = `${safeFileName(title)}.tja`;
    document.querySelector("#result-title").textContent = title;
    document.querySelector("#result-notes").textContent = result.noteCount.toLocaleString("ja-JP");
    document.querySelector("#result-measures").textContent = result.measures.length.toLocaleString("ja-JP");
    document.querySelector("#result-course").textContent = `${courseLabel(course)} / ★${level}`;
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

function generateChart(audioBuffer, bpm, level) {
  const sampleRate = audioBuffer.sampleRate;
  const frameSize = Math.max(1, Math.round(sampleRate * 0.01));
  const frameCount = Math.ceil(audioBuffer.length / frameSize);
  const energy = new Float32Array(frameCount);
  const channelData = Array.from({ length: audioBuffer.numberOfChannels }, (_, index) => audioBuffer.getChannelData(index));

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * frameSize;
    const end = Math.min(start + frameSize, audioBuffer.length);
    let sum = 0;
    let count = 0;
    for (const channel of channelData) {
      for (let sample = start; sample < end; sample += 1) {
        const value = channel[sample];
        sum += value * value;
        count += 1;
      }
    }
    energy[frame] = Math.sqrt(sum / Math.max(count, 1));
  }

  const onset = new Float32Array(frameCount);
  for (let frame = 1; frame < frameCount; frame += 1) {
    onset[frame] = Math.max(0, energy[frame] - energy[frame - 1]);
  }

  const positiveOnsets = Array.from(onset).filter((value) => value > 0.000001);
  const scale = percentile(positiveOnsets, 0.9);
  if (scale < 0.00001) {
    throw new Error("音源から音の立ち上がりを検出できませんでした。音量のある音源をお試しください。");
  }

  const secondsPerBeat = 60 / bpm;
  const slotDuration = secondsPerBeat / 4;
  const measureDuration = secondsPerBeat * 4;
  const measureCount = Math.max(1, Math.ceil(audioBuffer.duration / measureDuration));
  const measures = [];
  const threshold = 0.16 + (10 - level) * 0.022;
  let noteCount = 0;
  let previousWasBig = false;

  for (let measureIndex = 0; measureIndex < measureCount; measureIndex += 1) {
    let measure = "";
    for (let slot = 0; slot < 16; slot += 1) {
      const time = (measureIndex * 16 + slot) * slotDuration;
      if (time >= audioBuffer.duration) {
        measure += "0";
        continue;
      }
      const center = Math.round(time / 0.01);
      const radius = Math.max(1, Math.ceil(slotDuration * 0.32 / 0.01));
      let peak = 0;
      for (let frame = Math.max(0, center - radius); frame <= Math.min(frameCount - 1, center + radius); frame += 1) {
        peak = Math.max(peak, onset[frame] / scale);
      }

      if (peak < threshold) {
        measure += "0";
        previousWasBig = false;
        continue;
      }
      const isKat = slot % 4 === 2 || slot % 4 === 3;
      const isBig = level >= 5 && peak >= 0.78 && !previousWasBig;
      measure += isBig ? (isKat ? "4" : "3") : (isKat ? "2" : "1");
      previousWasBig = isBig;
      noteCount += 1;
    }
    measures.push(measure);
  }

  if (noteCount === 0) {
    throw new Error("この設定ではノーツを検出できませんでした。BPMや難易度を調整して再生成してください。");
  }
  return { measures, noteCount };
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
