import { downloadCSV, downloadSVGAsPNG, downloadJSON } from "../utils/download";

interface Props {
  // Prefix for generated filenames. A timestamp is always appended so
  // consecutive exports don't silently overwrite.
  filenameBase: string;
  // Function that returns the SVG node to rasterize, or null to hide the
  // PNG button (useful for canvas-only or non-graphical results).
  getSVG?: () => SVGSVGElement | null;
  // Function that returns tabular rows [header, ...body]. Null hides CSV.
  getCSVRows?: () => (string | number)[][] | null;
  // Optional raw JSON export — for "export the source data" flows where
  // CSV is lossy (e.g. nested top-k per cell).
  getJSON?: () => unknown;
  // PNG oversample factor. 2 = retina-quality output good for paper figures.
  pngScale?: number;
}

function timestampedName(base: string, ext: string): string {
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  return `${base}_${ts}.${ext}`;
}

// Reusable micro-toolbar placed above or beside a visualization. Skips any
// button whose getter returns null / undefined / empty, so per-viz wiring
// stays declarative: pass what you have, hide what you don't.
export function ExportButtons({
  filenameBase,
  getSVG,
  getCSVRows,
  getJSON,
  pngScale = 2,
}: Props) {
  const handlePNG = async () => {
    const svg = getSVG?.();
    if (!svg) return;
    try {
      await downloadSVGAsPNG(timestampedName(filenameBase, "png"), svg, pngScale);
    } catch (e) {
      // Rasterize can fail in rare cases (SVG with unloaded images, CSP
      // canvas taint). Surface the message so the user knows to try CSV.
      window.alert(`PNG export failed: ${(e as Error).message}`);
    }
  };

  const handleCSV = () => {
    const rows = getCSVRows?.();
    if (!rows || rows.length === 0) return;
    downloadCSV(timestampedName(filenameBase, "csv"), rows);
  };

  const handleJSON = () => {
    const data = getJSON?.();
    if (data == null) return;
    downloadJSON(timestampedName(filenameBase, "json"), data);
  };

  const btnStyle: React.CSSProperties = {
    fontSize: 10, padding: "1px 6px",
    background: "#0d1b2a", border: "1px solid #1a2540",
    color: "#a0a0c0", cursor: "pointer",
    borderRadius: 2,
  };

  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      {getSVG && (
        <button onClick={handlePNG} style={btnStyle} title="Download as PNG (2× retina)">png</button>
      )}
      {getCSVRows && (
        <button onClick={handleCSV} style={btnStyle} title="Download as CSV (values only)">csv</button>
      )}
      {getJSON && (
        <button onClick={handleJSON} style={btnStyle} title="Download raw data as JSON (preserves nested structure)">json</button>
      )}
    </div>
  );
}
