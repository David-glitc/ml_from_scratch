import { describe, expect, it } from "bun:test";
import { LinearRegression } from "../src/algorithms/linearRegression";
import { generateLinearDataset, trainTestSplit } from "../src/utils/datasets";

describe("LinearRegression", () => {
  it("fits a synthetic linear dataset with high R^2", () => {
    const { X, y } = generateLinearDataset(500, 3, 0.5, 123);
    const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(X, y, 0.25, 7);

    const lr = new LinearRegression(0.05, 2000);
    lr.fit(Xtrain, ytrain);
    const r2 = lr.score(Xtest, ytest);

    expect(r2).toBeGreaterThan(0.9);
  });

  it("perfectly fits noiseless univariate data", () => {
    const X: number[][] = [];
    const y: number[] = [];
    const wTrue = 2.5;
    const bTrue = -1.2;
    for (let i = 0; i < 50; i++) {
      const x = i / 5;
      X.push([x]);
      y.push(wTrue * x + bTrue);
    }
    const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(X, y, 0.2, 1);
    const lr = new LinearRegression(0.05, 1500);
    lr.fit(Xtrain, ytrain);
    const r2 = lr.score(Xtest, ytest);
    expect(r2).toBeGreaterThan(0.999);
  });

  it("recovers parameters approximately", () => {
    const { X, y, trueWeights, trueBias } = generateLinearDataset(800, 2, 0.3, 999);
    const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(X, y, 0.25, 11);
    const lr = new LinearRegression(0.05, 2500);
    lr.fit(Xtrain, ytrain);
    // Parameter closeness on average
    const w = lr.weights!;
    let wErr = 0;
    for (let j = 0; j < w.length; j++) wErr += Math.abs(w[j] - trueWeights[j]);
    wErr /= w.length;
    const bErr = Math.abs(lr.bias - trueBias);
    // Loose thresholds to avoid flakiness
    expect(wErr).toBeLessThan(0.2);
    expect(bErr).toBeLessThan(0.2);
    expect(lr.score(Xtest, ytest)).toBeGreaterThan(0.9);
  });
});


