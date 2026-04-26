import { describe, it, expect } from "vitest";
import { PROMPT_SET_PRESETS, getPresetById } from "./promptSetLibrary";

describe("PROMPT_SET_PRESETS — structural invariants", () => {
  it("library is non-empty", () => {
    expect(PROMPT_SET_PRESETS.length).toBeGreaterThan(0);
  });

  it("every preset has unique id", () => {
    const ids = PROMPT_SET_PRESETS.map((p) => p.id);
    const unique = new Set(ids);
    expect(unique.size).toBe(ids.length);
  });

  it("every preset has non-empty required fields", () => {
    for (const p of PROMPT_SET_PRESETS) {
      expect(p.id.length).toBeGreaterThan(0);
      expect(p.label.length).toBeGreaterThan(0);
      expect(p.description.length).toBeGreaterThan(0);
      expect(p.recommendedMainPrompt.length).toBeGreaterThan(0);
    }
  });

  it("every preset has at least 3 comparison prompts (heatmap needs ≥2 columns to be informative)", () => {
    for (const p of PROMPT_SET_PRESETS) {
      expect(p.prompts.length).toBeGreaterThanOrEqual(3);
    }
  });

  it("every preset's comparison prompts are non-empty strings", () => {
    for (const p of PROMPT_SET_PRESETS) {
      for (const prompt of p.prompts) {
        expect(prompt.trim().length).toBeGreaterThan(0);
      }
    }
  });

  it("comparison prompts are distinct within each preset", () => {
    for (const p of PROMPT_SET_PRESETS) {
      expect(new Set(p.prompts).size).toBe(p.prompts.length);
    }
  });

  it("recommendedMainPrompt is not in comparison prompts (would be redundant)", () => {
    for (const p of PROMPT_SET_PRESETS) {
      expect(p.prompts).not.toContain(p.recommendedMainPrompt);
    }
  });
});

describe("getPresetById", () => {
  it("returns the matching preset for a known id", () => {
    const result = getPresetById("country-capitals");
    expect(result).toBeDefined();
    expect(result?.label).toBe("Country capitals");
  });

  it("returns undefined for an unknown id", () => {
    expect(getPresetById("does-not-exist")).toBeUndefined();
  });

  it("returns undefined for an empty id", () => {
    expect(getPresetById("")).toBeUndefined();
  });
});
