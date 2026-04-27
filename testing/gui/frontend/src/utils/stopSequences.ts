/**
 * Parse the user-typed stop-sequences string into a list of literal
 * stop strings to send to the backend.
 *
 * The textarea accepts comma-separated entries with `\n`, `\t`, `\r`
 * escapes — typed literally as backslash-n etc. The order of trim →
 * unescape matters: trim must run on the RAW user input so it only
 * strips spaces the user typed around the comma boundaries, not the
 * actual newline characters produced by the unescape pass. The earlier
 * implementation did them in the opposite order and "\n\n" was being
 * silently trimmed away to the empty string, dropping the stop entirely
 * and leaving generation to run to max_tokens.
 */
export function parseStopSequences(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim().replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\r/g, "\r"))
    .filter(Boolean);
}
