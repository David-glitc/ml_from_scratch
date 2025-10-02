import { KNN } from "../src/algorithms/KNN";
import { benchmarkAlgo, type BenchmarkResult } from "../src/utils/benchmark";
import { getToyDataset, loadIrisCsv, irisToXY, trainTestSplit } from "../src/utils/datasets";
import fs from "fs";

async function main() {
  const results: BenchmarkResult[] = [];
  const { Xtrain, ytrain, Xtest, ytest } = getToyDataset();
  const knn = new KNN<string | number>(3);
  const res = await benchmarkAlgo<string | number>(knn, Xtrain, ytrain, Xtest, ytest, "KNN (Bun)", "ToyDataset");

  results.push(res);

  const outPath = `${__dirname}/results.jsonl`;
  const line = JSON.stringify(res) + "\n";
  fs.appendFileSync(outPath, line, "utf-8");

  try {
    const { rows } = await loadIrisCsv();
    const { X, y } = irisToXY(rows);
    const split = trainTestSplit(X, y, 0.2, 42);
    const knnIris = new KNN<string>(5);
    const resIris = await benchmarkAlgo<string>(
      knnIris,
      split.Xtrain,
      split.ytrain,
      split.Xtest,
      split.ytest,
      "KNN (Bun)",
      "Iris"
    );
    results.push(resIris);
    fs.appendFileSync(outPath, JSON.stringify(resIris) + "\n", "utf-8");
  } catch (_) {
    console.log("Iris dataset not present");
  }

  // Print results as a table
  if (results.length > 0) {
    console.table(
      results.map((r) => ({
        algorithm: r.algorithm,
        dataset: r.dataset,
        fitMs: `${r.fitTimeMs.toFixed(2)} ms`,
        predictMs: `${r.predictTimeMs.toFixed(2)} ms`,
        accuracy: r.accuracy.toFixed(3),
        memoryMB: r.memoryMB ? r.memoryMB.toFixed(1) : "-",
      }))
    );
  }

  // Print predictions vs truth for each dataset
  console.log("\nPredictions (ToyDataset):");
  const toyPred = new KNN<string | number>(3);
  toyPred.fit(Xtrain, ytrain);
  const yPredToy = toyPred.predict(Xtest);
  console.table(
    ytest.map((truth, i) => ({ index: i, truth, pred: yPredToy[i] }))
  );

  try {
    const { rows } = await loadIrisCsv();
    const { X, y } = irisToXY(rows);
    const split = trainTestSplit(X, y, 0.2, 42);
    const irisPred = new KNN<string>(5);
    irisPred.fit(split.Xtrain, split.ytrain);
    const yPredIris = irisPred.predict(split.Xtest);
    console.log("\nPredictions (Iris):");
    console.table(
      split.ytest.map((truth, i) => ({ index: i, truth, pred: yPredIris[i] }))
    );
  } catch (_) {
    console.log("Iris dataset not present");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

