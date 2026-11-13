import type { OrderImportRow } from "../api/weighbridgeOrders";

/**
 * Read the planning Excel sheet the plant sends:
 *   farm_code | farm_name | date_exp | Quantity_exp | feed_code | prodiction_num
 * Columns are found by their header text, so their order may change and extra columns are ignored.
 */

const HEADERS: Record<keyof Omit<OrderImportRow, "row">, string[]> = {
  farm_code: ["farm_code", "farmcode"],
  farm_name: ["farm_name", "farmname"],
  quantity_exp: ["quantity_exp", "quantityexp", "quantity"],
  feed_code: ["feed_code", "feedcode"],
  production_num: ["prodiction_num", "production_num", "prodictionnum", "productionnum"],
};
const REQUIRED: (keyof typeof HEADERS)[] = ["farm_code", "feed_code", "production_num"];

const norm = (value: unknown) => String(value ?? "").trim().toLowerCase().replace(/[\s-]+/g, "_");

/** A cell's plain value: formulas give their result, rich text its text. */
function plain(value: unknown): unknown {
  if (value == null) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "object") {
    const v = value as { result?: unknown; text?: unknown; richText?: { text: string }[] };
    if (v.result !== undefined) return plain(v.result);
    if (Array.isArray(v.richText)) return v.richText.map((part) => part.text).join("");
    if (v.text !== undefined) return plain(v.text);
    return null;
  }
  return value;
}

export async function readOrdersSheet(file: File): Promise<OrderImportRow[]> {
  if (!/\.xlsx$/i.test(file.name)) {
    throw new Error("Choose an Excel file that ends in .xlsx");
  }
  const ExcelJS = (await import("exceljs")).default;
  const wb = new ExcelJS.Workbook();
  try {
    await wb.xlsx.load(await file.arrayBuffer());
  } catch {
    throw new Error("This file could not be read as an Excel sheet");
  }
  const ws = wb.worksheets.find((sheet) => sheet.rowCount > 1) ?? wb.worksheets[0];
  if (!ws) throw new Error("The file has no sheet");

  const columns: Partial<Record<keyof typeof HEADERS, number>> = {};
  ws.getRow(1).eachCell((cell, col) => {
    const text = norm(plain(cell.value));
    (Object.keys(HEADERS) as (keyof typeof HEADERS)[]).forEach((key) => {
      if (columns[key] == null && HEADERS[key].includes(text)) columns[key] = col;
    });
  });
  const missing = REQUIRED.filter((key) => columns[key] == null);
  if (missing.length) {
    throw new Error(
      `The first row must have the columns farm_code, farm_name, Quantity_exp, feed_code and prodiction_num (missing: ${missing
        .map((key) => HEADERS[key][0])
        .join(", ")})`
    );
  }

  const rows: OrderImportRow[] = [];
  for (let r = 2; r <= ws.rowCount; r += 1) {
    const row = ws.getRow(r);
    const cell = (key: keyof typeof HEADERS) => (columns[key] == null ? null : plain(row.getCell(columns[key]!).value));
    const item: OrderImportRow = {
      row: r,
      farm_code: cell("farm_code"),
      farm_name: cell("farm_name"),
      quantity_exp: cell("quantity_exp"),
      feed_code: cell("feed_code"),
      production_num: cell("production_num"),
    };
    const empty = [item.farm_code, item.farm_name, item.quantity_exp, item.feed_code, item.production_num].every(
      (v) => v == null || String(v).trim() === ""
    );
    if (!empty) rows.push(item);
  }
  if (rows.length === 0) throw new Error("The sheet has no orders under the header row");
  return rows;
}
