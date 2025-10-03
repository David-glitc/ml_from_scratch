import { KNN } from "../src/algorithms/KNN";
import { LinearRegression } from "../src/algorithms/linearRegression";
import { benchmarkAlgo, type BenchmarkResult } from "../src/utils/benchmark";
import { getToyDataset, loadIrisCsv, irisToXY, trainTestSplit, generateLinearDataset, downloadHousingCsv, parseHousingCsvToXYWithTarget, fitStandardScaler, transformStandardScaler } from "../src/utils/datasets";
import { LogisticRegression } from "../src/algorithms/logisticRegression";
import fs from "fs";

async function main() {
  const results: BenchmarkResult[] = [];
  type PredPrinter = () => void | Promise<void>;
  const predictionPrinters: Array<{ algorithm: string; dataset: string; print: PredPrinter }> = [];
  const metaByKey: Record<string, { nTrain: number; nTest: number; nFeatures: number; params: string }> = {};

  // Determine whether to print predictions
  const args = process.argv.slice(2);
  const envToggle = (process.env.BENCH_PRINT_PREDICTIONS ?? "1").toLowerCase();
  const envEnabled = envToggle === "1" || envToggle === "true" || envToggle === "yes";
  const cliDisable = args.includes("--no-preds") || args.includes("--no-predictions");
  const cliEnable = args.includes("--preds") || args.includes("--predictions");
  const printPredictions = cliEnable ? true : cliDisable ? false : envEnabled;

  const { Xtrain, ytrain, Xtest, ytest } = getToyDataset();
  const knn = new KNN<string | number>(3);
  const res = await benchmarkAlgo<string | number>(knn, Xtrain, ytrain, Xtest, ytest, "KNN (Bun)", "ToyDataset");

  results.push(res);

  const outPath = `${__dirname}/results.jsonl`;
  const line = JSON.stringify(res) + "\n";
  fs.appendFileSync(outPath, line, "utf-8");

  // meta + predictions for ToyDataset
  metaByKey[`${res.algorithm}|${res.dataset}`] = {
    nTrain: Xtrain.length,
    nTest: Xtest.length,
    nFeatures: Xtrain[0]?.length ?? 0,
    params: "k=3",
  };
  predictionPrinters.push({
    algorithm: res.algorithm,
    dataset: res.dataset,
    print: () => {
      console.log("\nPredictions (ToyDataset):");
      const toyPred = new KNN<string | number>(3);
      toyPred.fit(Xtrain, ytrain);
      const yPredToy = toyPred.predict(Xtest);
      console.table(
        ytest.map((truth, i) => ({ index: i, truth, pred: yPredToy[i] }))
      );
    },
  });

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

    metaByKey[`${resIris.algorithm}|${resIris.dataset}`] = {
      nTrain: split.Xtrain.length,
      nTest: split.Xtest.length,
      nFeatures: split.Xtrain[0]?.length ?? 0,
      params: "k=5",
    };
    predictionPrinters.push({
      algorithm: resIris.algorithm,
      dataset: resIris.dataset,
      print: () => {
        const irisPred = new KNN<string>(5);
        irisPred.fit(split.Xtrain, split.ytrain);
        const yPredIris = irisPred.predict(split.Xtest);
        console.log("\nPredictions (Iris):");
        console.table(
          split.ytest.map((truth, i) => ({ index: i, truth, pred: yPredIris[i] }))
        );
      },
    });
  } catch (_) {
    console.log("Iris dataset not present");
  }

  // Linear Regression synthetic benchmark (R^2 reported in accuracy field)
  try {
    const { X, y } = generateLinearDataset(2000, 5, 1.0, 1234);
    const split = trainTestSplit(X, y, 0.25, 99);
    const lr = new LinearRegression(0.05, 1500);
    const lrWrapper: {
      fit: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => void;
      score: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => number;
    } = {
      fit: (X, y) => lr.fit(X, y as number[]),
      score: (X, y) => lr.score(X, y as number[]),
    };
    const resLR = await benchmarkAlgo<number>(
      lrWrapper as unknown as { fit: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => void; score: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => number },
      split.Xtrain,
      split.ytrain,
      split.Xtest,
      split.ytest,
      "LinearRegression (Bun)",
      "SyntheticLinear"
    );
    results.push(resLR);
    fs.appendFileSync(outPath, JSON.stringify(resLR) + "\n", "utf-8");

    metaByKey[`${resLR.algorithm}|${resLR.dataset}`] = {
      nTrain: split.Xtrain.length,
      nTest: split.Xtest.length,
      nFeatures: split.Xtrain[0]?.length ?? 0,
      params: "lr=0.05, iters=1500",
    };
    predictionPrinters.push({
      algorithm: resLR.algorithm,
      dataset: resLR.dataset,
      print: () => {
        const yPredLR = lr.predict(split.Xtest);
        console.log("\nPredictions (SyntheticLinear):");
        const rows = split.ytest.map((truth, i) => ({ index: i, truth, pred: Number(yPredLR[i].toFixed(3)) }));
        console.table(rows.slice(0, 30));
      },
    });
  } catch (e) {
    console.log("LinearRegression benchmark failed:", e);
  }

  // Logistic Regression on Housing (binary target from median_house_value)
  try {
    const path = await downloadHousingCsv();
    const text = await Bun.file(path).text();
    const { X, y } = (() => {
      const parsed = parseHousingCsvToXYWithTarget(text);
      const scaler = fitStandardScaler(parsed.X);
      const Xs = transformStandardScaler(parsed.X, scaler);
      return { X: Xs, y: parsed.y };
    })();
    const split = trainTestSplit(X, y, 0.25, 123);
    const clf = new LogisticRegression(0.1, 2000, Math.max(1, navigator?.hardwareConcurrency ?? 4));
    const logWrapper: {
      fit: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => void;
      score: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => number;
    } = {
      fit: (X, y) => clf.fit(X, y as number[]),
      score: (X, y) => clf.score(X, y as number[]),
    };
    const resLog = await benchmarkAlgo<number>(
      logWrapper as unknown as { fit: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => void | Promise<void>; score: (X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>) => number },
      split.Xtrain,
      split.ytrain,
      split.Xtest,
      split.ytest,
      "LogisticRegression (Bun)",
      "HousingBinary"
    );
    results.push(resLog);
    fs.appendFileSync(outPath, JSON.stringify(resLog) + "\n", "utf-8");

    metaByKey[`${resLog.algorithm}|${resLog.dataset}`] = {
      nTrain: split.Xtrain.length,
      nTest: split.Xtest.length,
      nFeatures: split.Xtrain[0]?.length ?? 0,
      params: "lr=0.1, iters=1500",
    };
    predictionPrinters.push({
      algorithm: resLog.algorithm,
      dataset: resLog.dataset,
      print: () => {
        const probs = clf.predictProba(split.Xtest);
        const preds = clf.predict(split.Xtest);
        console.log("\nPredictions (HousingBinary):");
        const rows = split.ytest.slice(0, 30).map((truth, i) => ({ index: i, truth, prob1: Number(probs[i].toFixed(3)), pred: preds[i] }));
        console.table(rows);
      },
    });
  } catch (e) {
    console.log("LogisticRegression benchmark failed:", e);
  }

  // Print results as a table with more details
  if (results.length > 0) {
    console.table(
      results.map((r) => {
        const meta = metaByKey[`${r.algorithm}|${r.dataset}`];
        return {
          algorithm: r.algorithm,
          dataset: r.dataset,
          nTrain: meta ? meta.nTrain : "-",
          nTest: meta ? meta.nTest : "-",
          nFeatures: meta ? meta.nFeatures : "-",
          params: meta ? meta.params : "-",
          fitMs: `${r.fitTimeMs.toFixed(2)} ms`,
          predictMs: `${r.predictTimeMs.toFixed(2)} ms`,
          accuracy: r.accuracy.toFixed(3),
          memoryMB: r.memoryMB ? r.memoryMB.toFixed(1) : "-",
        };
      })
    );
  }

  // Print predictions in the same order as the results table
  if (printPredictions) {
    for (const r of results) {
      const entry = predictionPrinters.find((p) => p.algorithm === r.algorithm && p.dataset === r.dataset);
      if (entry) await entry.print();
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

