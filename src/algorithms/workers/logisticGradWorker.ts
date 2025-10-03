// Worker to compute partial gradients for logistic regression

type GradRequest = {
  X: number[][];
  y: number[];
  weights: number[];
  bias: number;
};

type GradResponse = {
  gradW: number[];
  gradB: number;
};

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

self.onmessage = (ev: MessageEvent<GradRequest>) => {
  const { X, y, weights, bias } = ev.data;
  const nFeatures = weights.length;
  const gradW = new Array<number>(nFeatures).fill(0);
  let gradB = 0;

  for (let i = 0; i < X.length; i++) {
    const xi = X[i];
    let z = 0;
    for (let j = 0; j < nFeatures; j++) z += weights[j] * xi[j];
    z += bias;
    const p = sigmoid(z);
    const error = p - y[i];
    for (let j = 0; j < nFeatures; j++) gradW[j] += error * xi[j];
    gradB += error;
  }

  const resp: GradResponse = { gradW, gradB };
  (self as any).postMessage(resp);
};


