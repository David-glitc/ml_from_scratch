import fs from "fs";

const HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv";

async function main() {
  fs.mkdirSync("data", { recursive: true });

  const res = await fetch(HOUSING_URL);
  if (!res.ok) {
    throw new Error(`Failed to download Housing dataset: ${res.status} ${res.statusText}`);
  }
  const csv = await res.text();

  fs.writeFileSync("data/housing.csv", csv, "utf-8");
  const lines = csv.split(/\r?\n/).filter((l) => l.trim().length > 0);
  console.log("Saved data/housing.csv (", lines.length - 1, "rows)");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


