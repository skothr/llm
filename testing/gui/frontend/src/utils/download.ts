// Tiny download helpers shared by experiment export and per-visualization
// CSV/PNG exports. Browser downloads need an anchor element click — there
// is no Web API that streams bytes to the "Save As…" dialog directly.

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke after a tick so the browser has a chance to start the download.
  // Immediate revoke occasionally trips Safari's download pipeline.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadText(filename: string, text: string, mime = "text/plain"): void {
  downloadBlob(filename, new Blob([text], { type: mime }));
}

export function downloadJSON(filename: string, data: unknown): void {
  downloadText(filename, JSON.stringify(data, null, 2), "application/json");
}

export function downloadCSV(filename: string, rows: (string | number)[][]): void {
  // RFC-4180-lite: quote any cell containing a comma, quote, or newline.
  // Embedded quotes are doubled. Good enough for spreadsheet round-trip.
  const escape = (cell: string | number): string => {
    const s = String(cell);
    if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  const csv = rows.map((r) => r.map(escape).join(",")).join("\n");
  downloadText(filename, csv, "text/csv");
}

// Rasterize an inline <svg> to a PNG via a canvas roundtrip. Returns a
// Promise because Image.onload is async. Scale lets callers request retina
// output (e.g. 2× for print/paper figures).
export function downloadSVGAsPNG(
  filename: string,
  svg: SVGSVGElement,
  scale = 2,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xml = new XMLSerializer().serializeToString(svg);
    // Some browsers refuse to rasterize if the svg tag lacks xmlns.
    const xmlWithNs = xml.includes("xmlns=")
      ? xml
      : xml.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
    const svgBlob = new Blob([xmlWithNs], { type: "image/svg+xml;charset=utf-8" });
    const svgUrl = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      try {
        const bbox = svg.getBoundingClientRect();
        const w = Math.max(1, Math.ceil(bbox.width || img.width)) * scale;
        const h = Math.max(1, Math.ceil(bbox.height || img.height)) * scale;
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) { reject(new Error("2D context unavailable")); return; }
        // Paint the app's dark background so legends don't render on
        // transparent PNGs that look washed out on light doc backgrounds.
        ctx.fillStyle = "#0d1b2a";
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(img, 0, 0, w, h);
        canvas.toBlob((blob) => {
          URL.revokeObjectURL(svgUrl);
          if (!blob) { reject(new Error("PNG encode failed")); return; }
          downloadBlob(filename, blob);
          resolve();
        }, "image/png");
      } catch (e) {
        URL.revokeObjectURL(svgUrl);
        reject(e);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(svgUrl);
      reject(new Error("SVG load for rasterization failed"));
    };
    img.src = svgUrl;
  });
}
