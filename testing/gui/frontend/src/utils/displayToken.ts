export function displayToken(text: string): string {
  if (text === "") return "<empty>";
  return text
    .replace(/\u2581/g, "\u00B7")
    .replace(/ /g, "\u00B7")
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t")
    .replace(/\r/g, "\\r");
}
