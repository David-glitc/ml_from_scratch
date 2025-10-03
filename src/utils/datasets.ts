import { parseCsv } from "./csv";

export interface SupervisedSplit {
  Xtrain: number[][];
  ytrain: Array<string | number>;
  Xtest: number[][];
  ytest: Array<string | number>;
}

export function getToyDataset(): SupervisedSplit {
  const Xtrain = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 7],
    [7, 8],
    [8, 9],
  ];
  const ytrain = ["A", "A", "A", "B", "B", "B"] as const;

  const Xtest = [
    [2, 2],
    [7, 7],
  ];
  const ytest = ["A", "B"] as const;

  return { Xtrain, ytrain: [...ytrain], Xtest, ytest: [...ytest] };
}

export interface IrisSplit extends SupervisedSplit {}

export async function loadIrisCsv(path: string = "data/iris.csv"): Promise<{ header: string[]; rows: string[][] }> {
  const file = Bun.file(path);
  if (!(await file.exists())) {
    throw new Error(`Missing ${path}. Run: bun run download:iris`);
  }
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  const header = lines[0].split(",");
  const rows = lines.slice(1).map((l) => l.split(","));
  return { header, rows };
}

export function irisToXY(rows: string[][]): { X: number[][]; y: string[] } {
  const X: number[][] = [];
  const y: string[] = [];
  for (const r of rows) {
    if (r.length < 5) continue;
    X.push([Number(r[0]), Number(r[1]), Number(r[2]), Number(r[3])]);
    y.push(r[4]);
  }
  return { X, y };
}

export function trainTestSplit<TX, TY>(
  X: TX[],
  y: TY[],
  testSize = 0.2,
  seed = 42
): { Xtrain: TX[]; Xtest: TX[]; ytrain: TY[]; ytest: TY[] } {
  if (X.length !== y.length) throw new Error("X and y length mismatch");
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  // simple seeded shuffle
  let s = seed;
  for (let i = n - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    const j = s % (i + 1);
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.max(1, Math.floor(n * testSize));
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);

  const Xtrain = trainIdx.map((i) => X[i]);
  const ytrain = trainIdx.map((i) => y[i]);
  const Xtest = testIdx.map((i) => X[i]);
  const ytest = testIdx.map((i) => y[i]);
  return { Xtrain, Xtest, ytrain, ytest };
}

export function generateLinearDataset(
  nSamples: number = 200,
  nFeatures: number = 1,
  noiseStd: number = 1,
  seed = 123
): { X: number[][]; y: number[]; trueWeights: number[]; trueBias: number } {
  if (nSamples <= 0) throw new Error("nSamples must be > 0");
  if (nFeatures <= 0) throw new Error("nFeatures must be > 0");

  // Simple seeded RNG (LCG)
  let s = seed >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };

  // Ground-truth parameters
  const trueWeights = Array.from({ length: nFeatures }, () => (rand() * 4 - 2)); // [-2, 2]
  const trueBias = rand() * 2 - 1; // [-1, 1]

  const X: number[][] = new Array(nSamples);
  const y: number[] = new Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const xi = new Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      xi[j] = rand() * 10 - 5; // [-5, 5]
    }
    let yi = trueBias;
    for (let j = 0; j < nFeatures; j++) yi += trueWeights[j] * xi[j];
    // Gaussian-ish noise via CLT sum of uniforms
    let noise = 0;
    for (let k = 0; k < 12; k++) noise += rand();
    noise -= 6; // approx N(0,1)
    yi += noise * noiseStd;
    X[i] = xi;
    y[i] = yi;
  }
  return { X, y, trueWeights, trueBias };
}

export async function saveRegressionCsv(
  path: string,
  X: ReadonlyArray<ReadonlyArray<number>>,
  y: ReadonlyArray<number>,
  header?: string[]
): Promise<void> {
  if (X.length !== y.length) throw new Error("X and y length mismatch");
  const cols = X[0]?.length ?? 0;
  const h = header ?? [
    ...Array.from({ length: cols }, (_, i) => `x${i + 1}`),
    "y",
  ];
  const lines: string[] = [];
  lines.push(h.join(","));
  for (let i = 0; i < X.length; i++) {
    const row = [...X[i], y[i]].join(",");
    lines.push(row);
  }
  await Bun.write(path, lines.join("\n"));
}

export async function loadRegressionCsv(
  path: string
): Promise<{ X: number[][]; y: number[]; header: string[] }> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`Missing ${path}`);
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return { X: [], y: [], header: [] };
  const header = lines[0].split(",");
  const rows = lines.slice(1).map((l) => l.split(",").map(Number));
  const X = rows.map((r) => r.slice(0, -1));
  const y = rows.map((r) => r[r.length - 1]);
  return { X, y, header };
}

export async function downloadHousingCsv(
  url: string = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
  outPath: string = "data/housing.csv"
): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download housing dataset: ${res.status} ${res.statusText}`);
  const text = await res.text();
  await Bun.write(outPath, text);
  return outPath;
}

export function parseHousingCsvToXY(
  csvText: string,
  threshold: number = 180000
): { X: number[][]; y: number[]; header: string[] } {
  const { header, rows } = parseCsv(csvText, { hasHeader: true });
  if (!header) return { X: [], y: [], header: [] };
  const targetCol = header.indexOf("median_house_value");
  if (targetCol === -1) throw new Error("median_house_value column not found");

  // Determine numeric columns by sampling rows (exclude target and known categorical 'ocean_proximity')
  const candidateCols: number[] = header.map((_, i: number) => i).filter((i: number) => i !== targetCol);
  const categoricalByName = new Set(["ocean_proximity"]);
  const categoricalIdx = new Set<number>(
    header
      .map((name, i) => ({ name, i }))
      .filter(({ name }) => categoricalByName.has(name))
      .map(({ i }) => i)
  );
  const numericCols: number[] = [];
  for (const i of candidateCols) {
    if (categoricalIdx.has(i)) continue;
    let numericCount = 0;
    let checked = 0;
    for (let r = 0; r < rows.length && checked < 200; r++) {
      const v = Number(rows[r][i]);
      if (!Number.isNaN(v) && Number.isFinite(v)) numericCount++;
      checked++;
    }
    if (checked > 0 && numericCount / checked > 0.95) numericCols.push(i);
  }

  const X: number[][] = [];
  const y: number[] = [];
  for (const r of rows) {
    if (r.length !== header.length) continue;
    const target = Number(r[targetCol]);
    if (!isFinite(target)) continue;
    const xi: number[] = [];
    let hasNaN = false;
    for (const i of numericCols) {
      const v = Number(r[i]);
      if (!isFinite(v)) { hasNaN = true; break; }
      xi.push(v);
    }
    if (hasNaN) continue;
    X.push(xi);
    y.push(target >= threshold ? 1 : 0);
  }
  return { X, y, header };
}

export function parseHousingCsvToXYWithTarget(
  csvText: string,
  threshold?: number
): { X: number[][]; y: number[]; header: string[]; target: number[] } {
  const { header, rows } = parseCsv(csvText, { hasHeader: true });
  if (!header) return { X: [], y: [], header: [], target: [] };
  const targetCol = header.indexOf("median_house_value");
  if (targetCol === -1) throw new Error("median_house_value column not found");

  const candidateCols: number[] = header.map((_, i: number) => i).filter((i: number) => i !== targetCol);
  const categoricalByName = new Set(["ocean_proximity"]);
  const categoricalIdx = new Set<number>(
    header
      .map((name, i) => ({ name, i }))
      .filter(({ name }) => categoricalByName.has(name))
      .map(({ i }) => i)
  );
  const numericCols: number[] = [];
  for (const i of candidateCols) {
    if (categoricalIdx.has(i)) continue;
    let numericCount = 0;
    let checked = 0;
    for (let r = 0; r < rows.length && checked < 200; r++) {
      const v = Number(rows[r][i]);
      if (!Number.isNaN(v) && Number.isFinite(v)) numericCount++;
      checked++;
    }
    if (checked > 0 && numericCount / checked > 0.95) numericCols.push(i);
  }

  const X: number[][] = [];
  const targetValues: number[] = [];
  for (const r of rows) {
    if (r.length !== header.length) continue;
    const t = Number(r[targetCol]);
    if (!isFinite(t)) continue;
    const xi: number[] = [];
    let hasNaN = false;
    for (const i of numericCols) {
      const v = Number(r[i]);
      if (!isFinite(v)) { hasNaN = true; break; }
      xi.push(v);
    }
    if (hasNaN) continue;
    X.push(xi);
    targetValues.push(t);
  }

  let usedThreshold: number;
  if (typeof threshold === "number") {
    usedThreshold = threshold;
  } else {
    // median threshold for balance
    const sorted = [...targetValues].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    usedThreshold = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }
  const y = targetValues.map((t) => (t >= usedThreshold ? 1 : 0));
  return { X, y, header, target: targetValues };
}

export interface StandardScaler {
  mean: Float64Array;
  std: Float64Array;
}

export function fitStandardScaler(X: ReadonlyArray<ReadonlyArray<number>>): StandardScaler {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    for (let j = 0; j < d; j++) mean[j] += xi[j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n || 1;
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    for (let j = 0; j < d; j++) {
      const diff = xi[j] - mean[j];
      std[j] += diff * diff;
    }
  }
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / (n || 1)) || 1;
  return { mean, std };
}

export function transformStandardScaler(
  X: ReadonlyArray<ReadonlyArray<number>>,
  scaler: StandardScaler
): number[][] {
  const n = X.length;
  const d = scaler.mean.length;
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    const row = new Array(d);
    for (let j = 0; j < d; j++) row[j] = (xi[j] - scaler.mean[j]) / scaler.std[j];
    out[i] = row as number[];
  }
  return out;
}

