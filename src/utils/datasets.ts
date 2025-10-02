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

