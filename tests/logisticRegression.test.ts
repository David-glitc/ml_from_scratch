import { describe, expect, it } from "bun:test";
import { LogisticRegression } from "../src/algorithms/logisticRegression";

function makeToyLogistic(seed = 42) {
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const n = 400;
  const w = [2, -1.5];
  const b = 0.3;
  const X: number[][] = new Array(n);
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const x1 = rand() * 4 - 2;
    const x2 = rand() * 4 - 2;
    const z = w[0] * x1 + w[1] * x2 + b;
    const p = 1 / (1 + Math.exp(-z));
    const noise = rand();
    const label = noise < p ? 1 : 0;
    X[i] = [x1, x2];
    y[i] = label;
  }
  return { X, y };
}

describe("LogisticRegression", () => {
  it("achieves good accuracy on separable-ish data", () => {
    const { X, y } = makeToyLogistic(123);
    // simple split
    const nTrain = Math.floor(X.length * 0.75);
    const Xtrain = X.slice(0, nTrain);
    const ytrain = y.slice(0, nTrain);
    const Xtest = X.slice(nTrain);
    const ytest = y.slice(nTrain);

    const clf = new LogisticRegression(0.1, 2000);
    clf.fit(Xtrain, ytrain);
    const acc = clf.score(Xtest, ytest);
    expect(acc).toBeGreaterThan(0.8);
  });
});


