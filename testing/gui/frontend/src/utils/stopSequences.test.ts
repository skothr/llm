import { describe, it, expect } from "vitest";
import { parseStopSequences } from "./stopSequences";

describe("parseStopSequences", () => {
  it("returns the literal newline string for a `\\n\\n` input (the regression)", () => {
    // The default config ships "\\n\\n" — i.e. backslash-n-backslash-n —
    // which the user means as "stop on a blank line." If trim() runs
    // after the unescape, the resulting "\n\n" gets stripped to empty
    // and no stop fires. This case pins the correct order.
    expect(parseStopSequences("\\n\\n")).toEqual(["\n\n"]);
  });

  it("preserves leading/trailing newlines that came from escapes", () => {
    expect(parseStopSequences("foo\\n")).toEqual(["foo\n"]);
    expect(parseStopSequences("\\nfoo")).toEqual(["\nfoo"]);
  });

  it("strips user-typed whitespace around comma-separated entries", () => {
    expect(parseStopSequences("END, STOP")).toEqual(["END", "STOP"]);
    expect(parseStopSequences("  END  ,  STOP  ")).toEqual(["END", "STOP"]);
  });

  it("drops empty entries", () => {
    expect(parseStopSequences("END,,STOP")).toEqual(["END", "STOP"]);
    expect(parseStopSequences("")).toEqual([]);
    expect(parseStopSequences("  ,  ")).toEqual([]);
  });

  it("handles tab and carriage-return escapes", () => {
    expect(parseStopSequences("\\t")).toEqual(["\t"]);
    expect(parseStopSequences("\\r\\n")).toEqual(["\r\n"]);
  });

  it("supports multiple newline-only stops", () => {
    expect(parseStopSequences("\\n\\n,\\n###")).toEqual(["\n\n", "\n###"]);
  });
});
