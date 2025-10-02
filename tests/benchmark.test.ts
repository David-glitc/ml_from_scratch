import { describe, expect, it } from "bun:test";
import { KNN } from "../src/algorithms/KNN";
import { benchmarkAlgo } from "../src/utils/benchmark";
import { getToyDataset } from "../src/utils/datasets";

describe("benchmark harness", () => {
  it("runs and returns metrics", async () => {
    const { Xtrain, ytrain, Xtest, ytest } = getToyDataset();
    const knn = new KNN(3);
    const res = await benchmarkAlgo(knn, Xtrain, ytrain, Xtest, ytest, "KNN (Bun)", "ToyDataset");
    expect(res.algorithm).toBe("KNN (Bun)");
    expect(res.dataset).toBe("ToyDataset");
    expect(res.accuracy).toBeGreaterThan(0.5);
    expect(res.fitTimeMs).toBeGreaterThanOrEqual(0);
    expect(res.predictTimeMs).toBeGreaterThanOrEqual(0);
  });
});

