/**
 * Curated prompt-set presets for the divergence heatmap (Phase 3.19).
 *
 * Each preset is a probe-class: a hypothesis about writer behavior
 * encoded as a small set of prompts that share a structural template
 * but vary on one axis (country, plurality, synonym, etc.). Loading a
 * preset and running it against a circuit reveals which writers carry
 * the variable axis and which carry the structural template.
 *
 * `recommendedMainPrompt` is what we suggest the user puts in the
 * panel's MAIN prompt input; the heatmap then compares each entry of
 * `prompts` against it. The first comparison entry is conventionally
 * the same probe with a different value on the variable axis, so the
 * heatmap shows immediate divergence on the relevant writers.
 */
export type PromptSetPreset = {
  /** Stable identifier — used as <option value> and Playwright key. */
  id: string;
  /** Human-readable label shown in the dropdown. */
  label: string;
  /** What this preset probes for; surfaced as a tooltip + caption. */
  description: string;
  /** Suggested value for the panel's main prompt; informational only. */
  recommendedMainPrompt: string;
  /** Comparison prompts — what fills the textarea on selection. */
  prompts: string[];
};

export const PROMPT_SET_PRESETS: PromptSetPreset[] = [
  {
    id: "country-capitals",
    label: "Country capitals",
    description:
      "Probes writers that carry country-specific knowledge. Structural " +
      "writers (the subject-verb skeleton) should agree across all 5; " +
      "answer writers (the late-layer FFN that emits the capital) should " +
      "diverge on every prompt.",
    recommendedMainPrompt: "The capital of France is",
    prompts: [
      "The capital of Italy is",
      "The capital of Japan is",
      "The capital of Brazil is",
      "The capital of Egypt is",
      "The capital of Germany is",
    ],
  },
  {
    id: "subject-verb-agreement",
    label: "Subject-verb agreement",
    description:
      "Probes plurality-tracking writers. Compares singular vs plural " +
      "subject + matching verb. Writers that track grammatical number " +
      "should diverge between singular and plural rows; lexical writers " +
      "(the noun and verb embeddings) stay agreed within each pair.",
    recommendedMainPrompt: "The cat sleeps",
    prompts: [
      "The cats sleep",
      "The dog runs",
      "The dogs run",
      "The bird flies",
      "The birds fly",
    ],
  },
  {
    id: "synonyms",
    label: "Synonym substitution",
    description:
      "Probes semantic-invariance writers. The substituted adjective " +
      "carries the same meaning in each prompt; writers tracking the " +
      "concept (largeness) should agree, writers tracking surface form " +
      "(the literal token) should diverge.",
    recommendedMainPrompt: "The big house",
    prompts: [
      "The large house",
      "The huge house",
      "The enormous house",
      "The massive house",
      "The giant house",
    ],
  },
  {
    id: "numerical-succession",
    label: "Numerical succession",
    description:
      "Probes counting/arithmetic circuits. Each prompt asks for the " +
      "successor of a different small integer. Writers tracking the " +
      "+1 operation should agree on \"increment\"; writers tracking " +
      "the specific operand differ per prompt.",
    recommendedMainPrompt: "The number after 1 is",
    prompts: [
      "The number after 2 is",
      "The number after 3 is",
      "The number after 4 is",
      "The number after 5 is",
      "The number after 6 is",
    ],
  },
  {
    id: "translation",
    label: "Cross-lingual translation",
    description:
      "Probes cross-lingual transfer. Each prompt asks for the same " +
      "concept (\"hello\") in a different language. Writers carrying the " +
      "concept should agree across all 5; writers carrying the target-" +
      "language signal should diverge per prompt.",
    recommendedMainPrompt: "Hello in French is",
    prompts: [
      "Hello in Spanish is",
      "Hello in German is",
      "Hello in Italian is",
      "Hello in Japanese is",
      "Hello in Portuguese is",
    ],
  },
  {
    id: "negation-flip",
    label: "Negation flip",
    description:
      "Probes negation-handling writers. Pairs of (assertion, negated " +
      "assertion) on the same subject. Writers that track polarity " +
      "should diverge between the assertion and negation; lexical " +
      "writers (the subject's identity) agree within each pair.",
    recommendedMainPrompt: "The sky is blue",
    prompts: [
      "The sky is not blue",
      "The grass is green",
      "The grass is not green",
      "Snow is cold",
      "Snow is not cold",
    ],
  },
];

/** Lookup a preset by id; returns undefined if unknown. */
export function getPresetById(id: string): PromptSetPreset | undefined {
  return PROMPT_SET_PRESETS.find((p) => p.id === id);
}
