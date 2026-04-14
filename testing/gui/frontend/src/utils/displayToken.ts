export function displayToken(text: string): string {
  if (text === "") return "<empty>";
  return text
    .replace(/<0x([0-9A-Fa-f]{2})>/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    .replace(/\u2581/g, "\u00B7")
    .replace(/ /g, "\u00B7")
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t")
    .replace(/\r/g, "\\r");
}
