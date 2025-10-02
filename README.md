## ml-playground

### Install

```bash
bun install
```

### Run tests

```bash
bun test
```

### Run benchmarks

```bash
bun run bench
```

Download Iris and benchmark:

```bash
bun run bench:iris
```

This project uses [Bun](https://bun.sh) as the runtime.

### Project layout

```
src/
  algorithms/
    KNN.ts
    index.ts
  utils/
    benchmark.ts
    datasets.ts
    csv.ts
benchmarks/
  run_bun.ts
data/
  iris.csv (optional)
```

## Extend: add a new algorithm

1) Create the algorithm file in `src/algorithms/YourAlgo.ts` exporting a class with `fit(X, y)`, `predict(X)`, and optionally `score(X, y)`.

```ts
export class YourAlgo<L = string | number> {
  fit(X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<L>): void { /* ... */ }
  predict(X: ReadonlyArray<ReadonlyArray<number>>): L[] { /* ... */ return []; }
  score(X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<L>): number {
    const yPred = this.predict(X);
    let ok = 0; for (let i = 0; i < y.length; i++) if (yPred[i] === y[i]) ok++;
    return ok / y.length;
  }
}
```

2) Export it from `src/algorithms/index.ts`.

```ts
export { YourAlgo } from "./YourAlgo";
```

3) Add it to the benchmark runner `benchmarks/run_bun.ts`.

```ts
import { YourAlgo } from "../src/algorithms/YourAlgo";
import { benchmarkAlgo } from "../src/utils/benchmark";

const algo = new YourAlgo<string>();
const res = await benchmarkAlgo(algo, Xtrain, ytrain, Xtest, ytest, "YourAlgo (Bun)", datasetName);
```

## Add datasets

- CSV (general):

```ts
import { parseCsv, csvToNumbers } from "../src/utils/csv";

const text = await Bun.file("data/your.csv").text();
const { header, rows } = parseCsv(text); // header?: string[]; rows: string[][]
const X = csvToNumbers(rows.map(r => r.slice(0, -1)));
const y = rows.map(r => r[r.length - 1]);
```

- Iris (built-in helper):

```bash
bun run download:iris
```

```ts
import { loadIrisCsv, irisToXY, trainTestSplit } from "../src/utils/datasets";

const { rows } = await loadIrisCsv();
const { X, y } = irisToXY(rows);
const split = trainTestSplit(X, y, 0.2, 42);
```

## Add a benchmark

Use the provided harness `benchmarkAlgo` that records fit time, predict time, accuracy, and memory.

```ts
import { benchmarkAlgo } from "../src/utils/benchmark";

const res = await benchmarkAlgo(algo, split.Xtrain, split.ytrain, split.Xtest, split.ytest, "YourAlgo (Bun)", "YourDataset");
console.table([{ algo: res.algorithm, dataset: res.dataset, fit: `${res.fitTimeMs.toFixed(2)} ms`, predict: `${res.predictTimeMs.toFixed(2)} ms`, acc: res.accuracy.toFixed(3) }]);
```
