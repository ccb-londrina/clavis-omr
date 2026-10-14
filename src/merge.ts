// Junta as páginas reconhecidas numa partitura só: os compassos de cada página seguinte são anexados,
// parte a parte (na ordem), aos da primeira — como scripts/omr-benchmark/collect.py. Os compassos são
// renumerados em sequência (o homr começa cada página do 1).
export function mergePages(xmls: string[], title: string): string {
  const parser = new DOMParser();
  const docs = xmls.map((x) => parser.parseFromString(x, "application/xml"));
  for (const d of docs) {
    const err = d.querySelector("parsererror");
    if (err) throw new Error("MusicXML inválido: " + err.textContent);
  }
  const [first, ...rest] = docs;
  const parts = [...first.querySelectorAll("score-partwise > part")];
  for (const doc of rest) {
    const more = [...doc.querySelectorAll("score-partwise > part")];
    parts.forEach((part, i) => {
      for (const m of more[i]?.querySelectorAll(":scope > measure") ?? []) part.appendChild(first.importNode(m, true));
    });
  }
  for (const part of parts) {
    let n = 1;
    for (const m of part.querySelectorAll(":scope > measure")) m.setAttribute("number", String(n++));
  }
  if (title) setTitle(first, title);
  return new XMLSerializer().serializeToString(first);
}

function setTitle(doc: XMLDocument, title: string) {
  const root = doc.documentElement;
  let work = root.querySelector(":scope > work");
  if (!work) {
    work = doc.createElement("work");
    root.insertBefore(work, root.firstElementChild);
  }
  let wt = work.querySelector("work-title");
  if (!wt) wt = work.appendChild(doc.createElement("work-title"));
  wt.textContent = title;
  root.querySelector(":scope > movement-title")?.remove();
}
