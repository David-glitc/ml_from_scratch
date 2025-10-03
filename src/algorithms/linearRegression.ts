// y = wx + b
// MSE j(w,b) = 1/N E(n;i=1) (yi - (wxi + b))**2

export class LinearRegression {
  lr: number;
  n_iters: number;
  weights: number[] | null;
  bias: number;

  constructor(lr: number = 0.01, n_iters: number = 1000) {
    this.lr = lr;
    this.n_iters = n_iters;
    this.weights = null;
    this.bias = 0;
  }

  private dot(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  }

  fit(X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>): void {
    if (X.length !== y.length) throw new Error("X and y must have the same length");
    if (X.length === 0) throw new Error("Empty training set");

    const nSamples = X.length;
    const nFeatures = X[0].length;
    for (let i = 1; i < nSamples; i++) {
      if (X[i].length !== nFeatures) throw new Error("All rows in X must have the same number of features");
    }

    // Initialize
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    // Gradient descent
    for (let iter = 0; iter < this.n_iters; iter++) {
      const gradW = new Array(nFeatures).fill(0);
      let gradB = 0;

      for (let i = 0; i < nSamples; i++) {
        const xi = X[i] as number[];
        const yi = y[i] as number;
        const yPred = this.dot(this.weights, xi) + this.bias;
        const error = yPred - yi;
        // Accumulate gradients
        for (let j = 0; j < nFeatures; j++) gradW[j] += error * xi[j];
        gradB += error;
      }

      // Average gradients and update parameters
      const scale = this.lr / nSamples;
      for (let j = 0; j < nFeatures; j++) this.weights[j] -= scale * gradW[j];
      this.bias -= scale * gradB;
    }
  }

  predict(X: ReadonlyArray<ReadonlyArray<number>>): number[] {
    if (!this.weights) throw new Error("Model not fitted yet");
    const preds: number[] = new Array(X.length);
    for (let i = 0; i < X.length; i++) {
      preds[i] = this.dot(this.weights, X[i]) + this.bias;
    }
    return preds;
  }

  // Coefficient of determination (R^2)
  score(X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>): number {
    const yPred = this.predict(X);
    let mean = 0;
    for (let i = 0; i < y.length; i++) mean += y[i];
    mean /= y.length || 1;

    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < y.length; i++) {
      const diffRes = y[i] - yPred[i];
      const diffTot = y[i] - mean;
      ssRes += diffRes * diffRes;
      ssTot += diffTot * diffTot;
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }
}

export default LinearRegression;
