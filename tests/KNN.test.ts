import { describe, expect, it } from "bun:test";
import { KNN } from "../src/algorithms/KNN";

describe("KNN", () => {
  it("predicts simple classes on toy dataset", () => {
    const knn = new KNN(3);
    const X = [[1,2],[2,3],[3,3],[6,7],[7,8],[8,9]];
    const y = ["A","A","A","B","B","B"];
    knn.fit(X, y);

    const preds = knn.predict([[2,2],[7,7]]);
    expect(preds[0]).toBe("A");
    expect(preds[1]).toBe("B");
  });
});

