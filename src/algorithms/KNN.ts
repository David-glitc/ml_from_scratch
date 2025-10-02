export class KNN<L = string | number> {
  private featuresMatrix: number[][] = [];
  private labels: L[] = [];
  private readonly k: number;
  private readonly labelToKey: (label: L) => string;

  constructor(k: number = 3, labelToKey?: (label: L) => string) {
    if (k <= 0) throw new Error("k must be greater than 0");
    this.k = k;
    this.labelToKey = labelToKey ?? ((l: L) => String(l));
  }

  public fit(X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<L>): void {
    if (X.length !== y.length) {
      throw new Error("X and y must have the same length");
    }
    if (X.length === 0) {
      throw new Error("Empty training set");
    }
    // Copy into internal mutable arrays to avoid external mutations
    this.featuresMatrix = X.map((row) => row.slice() as number[]);
    this.labels = y.slice() as L[];
  }

  private euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must have the same length");
    }
    let sumSquares = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sumSquares += diff * diff;
    }
    return Math.sqrt(sumSquares);
  }

  public predictOne(x: ReadonlyArray<number>): L {
    if (this.featuresMatrix.length === 0) throw new Error("Model not fitted yet");

    const distances = this.featuresMatrix.map((point, idx) => ({
      dist: this.euclideanDistance(point, Array.from(x) as number[]),
      label: this.labels[idx],
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const kNearest = distances.slice(0, this.k);

    const voteCount: Record<string, number> = {};
    for (const neighbor of kNearest) {
      const key = this.labelToKey(neighbor.label as L);
      voteCount[key] = (voteCount[key] || 0) + 1;
    }

    const majorityKey = Object.entries(voteCount).sort((a, b) => b[1] - a[1])[0][0];
    // Map the majority string key back to a label of type L by finding an example
    const labelExample = kNearest.find((n) => this.labelToKey(n.label as L) === majorityKey)?.label;
    return labelExample as L;
  }

  public predict(X: ReadonlyArray<ReadonlyArray<number>>): L[] {
    return X.map((row) => this.predictOne(row));
  }

  public score(X: ReadonlyArray<ReadonlyArray<number>>, yTrue: ReadonlyArray<L>): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yPred[i] === yTrue[i]) correct++;
    }
    return correct / yTrue.length;
  }
}

export default KNN;

