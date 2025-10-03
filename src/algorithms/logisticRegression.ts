export class LogisticRegression {
  lr: number;
  n_iters: number;
  weights: Float64Array | null;
  bias: number;
  n_jobs: number;

  constructor(lr: number = 0.1, n_iters: number = 1000, n_jobs: number = 1) {
    this.lr = lr;
    this.n_iters = n_iters;
    this.weights = null;
    this.bias = 0;
    this.n_jobs = Math.max(1, Math.floor(n_jobs));
  }

  private sigmoid(z: number): number {
    // Guard against overflow
    if (z >= 0) {
      const ez = Math.exp(-z);
      return 1 / (1 + ez);
    } else {
      const ez = Math.exp(z);
      return ez / (1 + ez);
    }
  }

  private dot(a: Float64Array, b: ReadonlyArray<number>): number {
    let sum = 0;
    const len = a.length;
    for (let i = 0; i < len; i++) sum += a[i] * b[i];
    return sum;
  }

  async fit(X: ReadonlyArray<ReadonlyArray<number>>, y: ReadonlyArray<number>): Promise<void> {
    if (X.length !== y.length) throw new Error("X and y must have the same length");
    if (X.length === 0) throw new Error("Empty training set");

    const nSamples = X.length;
    const nFeatures = X[0].length;
    for (let i = 1; i < nSamples; i++) if (X[i].length !== nFeatures) throw new Error("Inconsistent feature length");

    this.weights = new Float64Array(nFeatures);
    this.bias = 0;

    const weights = this.weights; // local reference
    const scale = this.lr / nSamples;

    if (this.n_jobs <= 1) {
      const gradW = new Float64Array(nFeatures);
      for (let iter = 0; iter < this.n_iters; iter++) {
        for (let j = 0; j < nFeatures; j++) gradW[j] = 0;
        let gradB = 0;
        for (let i = 0; i < nSamples; i++) {
          const xi = X[i];
          let z = 0;
          for (let j = 0; j < nFeatures; j++) z += weights[j] * xi[j];
          z += this.bias;
          const p = this.sigmoid(z);
          const error = p - (y[i] as number);
          for (let j = 0; j < nFeatures; j++) gradW[j] += error * xi[j];
          gradB += error;
        }
        for (let j = 0; j < nFeatures; j++) weights[j] -= scale * gradW[j];
        this.bias -= scale * gradB;
      }
      return;
    }

    // Parallel path using workers
    const jobs = Math.min(this.n_jobs, nSamples);
    const chunkSize = Math.ceil(nSamples / jobs);
    for (let iter = 0; iter < this.n_iters; iter++) {
      const promises: Promise<{ gradW: number[]; gradB: number }>[] = [];
      for (let j = 0; j < jobs; j++) {
        const start = j * chunkSize;
        const end = Math.min(nSamples, start + chunkSize);
        if (start >= end) continue;
        const Xchunk = X.slice(start, end).map((row) => Array.from(row));
        const ychunk = Array.from(y.slice(start, end)) as number[];
        const wCopy = Array.from(weights);
        const worker = new Worker(new URL("./workers/logisticGradWorker.ts", import.meta.url).href, { type: "module" });
        const p = new Promise<{ gradW: number[]; gradB: number }>((resolve, reject) => {
          worker.onmessage = (ev: MessageEvent<{ gradW: number[]; gradB: number }>) => {
            resolve(ev.data);
            worker.terminate();
          };
          worker.onerror = (e) => reject(e);
        });
        worker.postMessage({ X: Xchunk, y: ychunk, weights: wCopy, bias: this.bias });
        promises.push(p);
      }
      const parts = await Promise.all(promises);
      const gradW = new Float64Array(nFeatures);
      let gradB = 0;
      for (const part of parts) {
        const gw = part.gradW;
        for (let j = 0; j < nFeatures; j++) gradW[j] += gw[j];
        gradB += part.gradB;
      }
      for (let j = 0; j < nFeatures; j++) weights[j] -= scale * gradW[j];
      this.bias -= scale * gradB;
    }
  }

  predictProba(X: ReadonlyArray<ReadonlyArray<number>>): number[] {
    const w = this.weights;
    if (!w) throw new Error("Model not fitted yet");
    const n = X.length;
    const probs = new Array<number>(n);
    const nFeatures = w.length;
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      let z = 0;
      for (let j = 0; j < nFeatures; j++) z += w[j] * xi[j];
      z += this.bias;
      probs[i] = this.sigmoid(z);
    }
    return probs;
  }

  predict(X: ReadonlyArray<ReadonlyArray<number>>): number[] {
    const p = this.predictProba(X);
    return p.map((v) => (v >= 0.5 ? 1 : 0));
  }

  score(X: ReadonlyArray<ReadonlyArray<number>>, yTrue: ReadonlyArray<number>): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) if (yPred[i] === yTrue[i]) correct++;
    return correct / yTrue.length;
  }
}

export default LogisticRegression;


