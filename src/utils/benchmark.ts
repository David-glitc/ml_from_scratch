export interface BenchmarkResult {
  algorithm: string;
  dataset: string;
  fitTimeMs: number;
  predictTimeMs: number;
  accuracy: number;
  memoryMB?: number;
}

export async function benchmarkAlgo<L extends string | number>(
  algo: { fit: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<L>) => void; score: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<L>) => number },
  Xtrain: ReadonlyArray<ReadonlyArray<number>>,
  ytrain: ReadonlyArray<L>,
  Xtest: ReadonlyArray<ReadonlyArray<number>>,
  ytest: ReadonlyArray<L>,
  name: string,
  dataset: string
): Promise<BenchmarkResult> {
  const startFit = performance.now();
  algo.fit(Xtrain, ytrain);
  const endFit = performance.now();

  const startPred = performance.now();
  const acc = algo.score(Xtest, ytest);
  const endPred = performance.now();

  const memBytes = typeof process !== "undefined" && (process as any).memoryUsage ? (process as any).memoryUsage().rss : undefined;
  const memoryMB = memBytes ? memBytes / (1024 * 1024) : undefined;

  return {
    algorithm: name,
    dataset,
    fitTimeMs: endFit - startFit,
    predictTimeMs: endPred - startPred,
    accuracy: acc,
    memoryMB,
  };
}

