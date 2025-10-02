export interface CsvReadOptions {
  delimiter?: string; // default ","
  hasHeader?: boolean; // default true
  trim?: boolean; // default true
}

export function parseCsv(text: string, options: CsvReadOptions = {}): { header?: string[]; rows: string[][] } {
  const delimiter = options.delimiter ?? ",";
  const hasHeader = options.hasHeader ?? true;
  const trim = options.trim ?? true;

  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return { rows: [] };

  const parseLine = (line: string): string[] => {
    const result: string[] = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === delimiter && !inQuotes) {
        result.push(trim ? current.trim() : current);
        current = "";
      } else {
        current += ch;
      }
    }
    result.push(trim ? current.trim() : current);
    return result;
  };

  const rows = lines.map(parseLine);
  if (hasHeader) {
    const [header, ...data] = rows;
    return { header, rows: data };
  }
  return { rows };
}

export function csvToNumbers(rows: string[][]): number[][] {
  return rows.map((r) => r.map((v) => Number(v)));
}

