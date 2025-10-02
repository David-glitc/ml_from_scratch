import { describe, expect, it } from "bun:test";
import { parseCsv, csvToNumbers } from "../src/utils/csv";

describe("CSV reader", () => {
  it("parses header and rows", () => {
    const text = "a,b,c\n1,2,3\n4,5,6\n";
    const { header, rows } = parseCsv(text);
    expect(header).toEqual(["a","b","c"]);
    expect(rows).toEqual([["1","2","3"],["4","5","6"]]);
  });

  it("converts rows to numbers", () => {
    const rows = [["1","2","3"],["4","5","6"]];
    const nums = csvToNumbers(rows);
    expect(nums).toEqual([[1,2,3],[4,5,6]]);
  });
});

