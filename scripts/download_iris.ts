import fs from "fs";

const UCI_IRIS = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";

async function main() {
  fs.mkdirSync("data", { recursive: true });

  const res = await fetch(UCI_IRIS);
  if (!res.ok) {
    throw new Error(`Failed to download Iris dataset: ${res.status} ${res.statusText}`);
  }
  const raw = await res.text();

 
  const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const header = "sepal_length,sepal_width,petal_length,petal_width,species";
  const csv = [header, ...lines].join("\n") + "\n";

  fs.writeFileSync("data/iris.csv", csv, "utf-8");
  console.log("Saved data/iris.csv (", lines.length, "rows)");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

