import type { TruckWeighOrder } from "../api/truckEntry";
import { FAKIEH_WORDMARK } from "./fakiehWordmark";

/**
 * The Fakieh "Dispatch and Receive Memo" (QMS F/TRN/04), filled from one weighbridge trip.
 * Printed through a hidden frame, so the browser's own dialog offers both the printer and
 * "Save as PDF" and no popup window is needed.
 */

const escapeHtml = (value: unknown): string =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const pad = (n: number) => String(n).padStart(2, "0");

function fmtDate(ts?: string | null): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()}`;
}

function fmtClock(ts?: string | null): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  const h = d.getHours();
  return `${pad(h % 12 === 0 ? 12 : h % 12)}:${pad(d.getMinutes())} ${h < 12 ? "AM" : "PM"}`;
}

function fmtKg(kg?: number | null): string {
  if (kg == null || Number.isNaN(kg)) return "";
  return `${kg.toLocaleString("en-US", { maximumFractionDigits: 0 })} kg`;
}

/** The values that go on the paper, in one place so they can be checked without printing. */
export function memoValues(o: TruckWeighOrder) {
  const product = [o.material_name, o.material_code].filter(Boolean).join(" - ");
  return {
    operationNo: o.ticket ?? "",
    date: fmtDate(o.second_ts || o.first_ts || o.created_at),
    clientName: o.order_client_name ?? "",
    product,
    entryWeight: fmtKg(o.first_weight_kg),
    exitWeight: fmtKg(o.second_weight_kg),
    netWeight: fmtKg(o.net_kg),
    bags: o.bags == null ? "" : String(o.bags),
    truckNo: o.truck_plate ?? "",
    carrier: o.truck_company ?? "",
    arrivalTime: fmtClock(o.first_ts),
    departureTime: fmtClock(o.second_ts),
    driverName: o.driver_name || o.truck_driver || "",
    scaleOperator: o.exit_operator ?? "",
    notes: o.memo_notes ?? "",
  };
}

const cell = (en: string, ar: string, value: string, key: string) => `
      <div class="cell">
        <div class="lbl"><span>${en}</span><span class="ar">${ar}</span></div>
        <div class="val" data-memo="${key}">${escapeHtml(value) || "&nbsp;"}</div>
      </div>`;

const section = (en: string, ar: string, rows: string[]) => `
    <div class="box">
      <div class="bar"><span>${en}</span><span class="ar">${ar}</span></div>
      ${rows.map((r) => `<div class="row">${r}</div>`).join("\n      ")}
    </div>`;

export function buildMemoHtml(order: TruckWeighOrder): string {
  const v = memoValues(order);
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Dispatch and Receive Memo ${escapeHtml(v.operationNo)}</title>
  <style>
    @page { size: A4 portrait; margin: 12mm 13mm; }
    * { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    html, body { margin: 0; padding: 0; background: #fff; color: #000; }
    body { font-family: "Segoe UI", Tahoma, Arial, sans-serif; font-size: 11.5pt; }
    .sheet { min-height: 271mm; display: flex; flex-direction: column; }
    .ar { direction: rtl; unicode-bidi: isolate; font-family: "Segoe UI", Tahoma, Arial, sans-serif; }
    .head { display: flex; justify-content: space-between; align-items: flex-start; }
    .pill { margin: 3mm 0 0 34mm; background: #000; color: #fff; border-radius: 9mm; padding: 2.6mm 9mm 3mm; text-align: center; font-weight: 700; line-height: 1.35; }
    .pill .ar { display: block; font-size: 13.5pt; }
    .pill .en { display: block; font-size: 11pt; }
    .brand { text-align: right; }
    .brand img { width: 37mm; display: block; margin-left: auto; }
    .qms { margin-top: 1.5mm; font-size: 7.5pt; font-weight: 700; line-height: 1.35; }
    .top { display: flex; margin: 9mm 0 3mm; font-weight: 700; }
    .top > div { flex: 1; display: flex; align-items: baseline; gap: 3mm; padding: 0 2mm; }
    .top .v { flex: 1; text-align: center; font-weight: 600; border-bottom: 0.3mm solid #999; min-height: 6mm; }
    .box { border: 0.5mm solid #000; margin-bottom: 4.5mm; }
    .bar { background: #d9d9d9; border-bottom: 0.5mm solid #000; display: flex; justify-content: space-between; padding: 1.6mm 2.2mm; font-weight: 700; font-size: 12pt; }
    .row { display: flex; }
    .row + .row { border-top: 0.5mm solid #000; }
    .cell { flex: 1; padding: 1.6mm 2.2mm 2mm; min-height: 15.5mm; }
    .cell + .cell { border-left: 0.5mm solid #000; }
    .lbl { display: flex; justify-content: space-between; font-weight: 700; font-size: 10.5pt; }
    .val { margin-top: 1.2mm; min-height: 6.5mm; border-bottom: 0.3mm solid #999; font-size: 12pt; font-weight: 600; text-align: center; unicode-bidi: plaintext; }
    .notes { margin-top: auto; }
    .notes .val { text-align: start; border-bottom: 0; min-height: 13mm; padding: 1.5mm 2.2mm; margin: 0; white-space: pre-wrap; font-weight: 500; }
    .foot .ar { display: block; }
    .foot { border-top: 0.4mm solid #000; margin-top: 5mm; padding-top: 1.5mm; text-align: center; font-size: 8pt; line-height: 1.5; }
  </style>
</head>
<body>
  <div class="sheet">
    <div class="head">
      <div class="pill"><span class="ar">مذكرة إخراج واستلام</span><span class="en">Dispatch and Receive Memo</span></div>
      <div class="brand">
        <img src="${FAKIEH_WORDMARK}" alt="Fakieh" />
        <div class="qms">QMS : F/TRN/04<br /><span class="ar">تاريخ الإصدار : ٢٠١٨/٦/٢٨م</span><br /><span class="ar">رقم الإصدار : ٣/١</span></div>
      </div>
    </div>

    <div class="top">
      <div><span>Operation No.</span><span class="v" data-memo="operationNo">${escapeHtml(v.operationNo)}</span><span class="ar">رقم العملية</span></div>
      <div><span>Date</span><span class="v" data-memo="date">${escapeHtml(v.date)}</span><span class="ar">التاريخ</span></div>
    </div>
${section("Client &amp; Shipment", "بيانات العميل والشحنة", [
  cell("Client Name", "اسم العميل", v.clientName, "clientName"),
  cell("Product", "المنتج", v.product, "product"),
])}
${section("Weights &amp; Quantities", "الأوزان والكميات", [
  cell("Exit Weight", "وزن الخروج", v.exitWeight, "exitWeight") + cell("Entry Weight", "وزن الدخول", v.entryWeight, "entryWeight"),
  cell("No. of Bags", "عدد الأكياس", v.bags, "bags") + cell("Net Weight", "الوزن الصافي", v.netWeight, "netWeight"),
])}
${section("Transport Details", "بيانات النقل", [
  cell("Carrier", "الناقل", v.carrier, "carrier") + cell("Truck No.", "رقم السيارة", v.truckNo, "truckNo"),
  cell("Departure Time", "ساعة المغادرة", v.departureTime, "departureTime") + cell("Arrival Time", "ساعة الوصول", v.arrivalTime, "arrivalTime"),
  cell("Signature", "التوقيع", "", "driverSignature") + cell("Driver Name", "اسم السائق", v.driverName, "driverName"),
  cell("Signature", "التوقيع", "", "operatorSignature") + cell("Scale Operator", "مشغل الميزان", v.scaleOperator, "scaleOperator"),
])}
    <div class="box notes">
      <div class="bar"><span>Notes</span><span class="ar">ملاحظات</span></div>
      <div class="val" data-memo="notes">${escapeHtml(v.notes) || "&nbsp;"}</div>
    </div>
    <div class="foot">
      <div class="ar">جدة - حي المدينة الصناعية - المدخل الشمالي - ص.ب : ٨٧٦٣/٦٠٠٢ - الرمز البريدي : ٢١٤٩٢</div>
      <div class="ar">هاتف : ٠٠٩٦٦١٢٥٥٦٢٨٠٧/٠٠٩٦٦١٢٦٣٦٣٨٣٦ - س.ت : ٤٠٣٠١٦٩٦٠١ - <span dir="ltr">E.mail : feed@fakiehgroup.com</span></div>
    </div>
  </div>
</body>
</html>`;
}

const FRAME_ID = "fakieh-memo-print-frame";

/**
 * Open the browser's print dialog with the filled memo.
 * Returns an error text to show the user, or null when the dialog was asked for.
 */
export function printMemo(order: TruckWeighOrder): string | null {
  try {
    document.getElementById(FRAME_ID)?.remove();
    const frame = document.createElement("iframe");
    frame.id = FRAME_ID;
    frame.setAttribute("aria-hidden", "true");
    frame.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0;opacity:0;";
    frame.onload = () => {
      const win = frame.contentWindow;
      if (!win) return;
      // Give the inlined logo a moment to decode before the dialog takes its snapshot.
      setTimeout(() => {
        try {
          win.focus();
          win.print();
        } catch {
          /* the dialog was refused; the frame is simply left for the next try */
        }
      }, 250);
    };
    frame.srcdoc = buildMemoHtml(order);
    document.body.appendChild(frame);
    return null;
  } catch {
    return "Could not open the memo for printing. Please try again.";
  }
}
