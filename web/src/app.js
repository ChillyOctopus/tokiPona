// The page: sends what's typed to the worker (worker.js) and shows each sentence's English, its grammar notes and how
// it parses, with arrows between its readings, and a single word's dictionary entry. Everything is built with
// textContent, never innerHTML, so nothing typed can become markup.

const form = document.getElementById('form');
const input = document.getElementById('input');
const go = document.getElementById('go');
const status = document.getElementById('status');
const results = document.getElementById('results');

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let loaded = false;
let latest = 0;

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const SEVERITY = { ERROR: 'Error', WARNING: 'Warning', INFO: 'Note' };


function tree(node) {
  const item = el('li');
  const line = el('span', 'node');
  line.append(el('span', 'kind', node.kind));
  if (node.role) line.append(el('span', 'role', node.role));
  line.append(el('span', 'words', node.text));
  item.append(line);
  if (node.children.length) {
    const list = el('ul');
    for (const child of node.children) list.append(tree(child));
    item.append(list);
  }
  return item;
}

function notesList(notes) {
  // The arrows already say when there's more than one reading.
  const shown = notes.filter((note) => note.code !== 'ambiguous-parse');
  if (!shown.length) return null;
  const list = el('ul', 'notes');
  for (const note of shown) {
    const item = el('li', `note ${note.severity.toLowerCase()}`);
    item.append(el('span', 'severity', SEVERITY[note.severity] ?? note.severity), ' ', note.message);
    list.append(item);
  }
  return list;
}

// A word's meanings by part of speech, laid out as on the dictionary page (web/build.py writes that markup), and a
// link to its entry there
function entryBlock(entry) {
  const block = el('section', 'entry');
  const head = el('h3', 'entry-word', entry.word);
  head.append(' ', el('span', `usage usage-${entry.usage}`, entry.usage));
  block.append(head);
  const parts = el('div', 'parts');
  for (const { pos, help, lines } of entry.senses) {
    const details = el('details', 'help');
    const summary = el('summary');
    const q = el('span', 'q', '?');
    q.setAttribute('aria-hidden', 'true');
    summary.append(el('span', 'pos-name', pos), q, el('span', 'visually-hidden', ' (what is this?)'));
    details.append(summary, el('p', '', help));
    const group = el('div', 'pos');
    group.append(details);
    for (const { label, glosses } of lines) {
      const joiner = glosses.some((gloss) => gloss.includes(',')) ? '; ' : ', ';
      const line = el('p', 'glosses');
      if (label) line.append(el('span', 'line-label', `${label}:`), ' ');
      line.append(glosses.join(joiner));
      group.append(line);
    }
    parts.append(group);
  }
  block.append(parts);
  if (entry.often.length) {
    const often = el('p', 'often');
    often.append(el('span', 'often-label', 'Often translated as'), ` ${entry.often.join(', ')}`);
    block.append(often);
  }
  const link = el('a', 'entry-link', `${entry.word} in the dictionary`);
  link.href = `dictionary#${encodeURIComponent(entry.word)}`;
  block.append(link);
  return block;
}

function sentenceCard(sentence) {
  const card = el('article', 'sentence');
  card.append(el('p', 'tp', sentence.text));
  const { readings } = sentence;

  if (!readings.length) {
    // It doesn't parse: the English is word by word, and the error note says why.
    card.append(el('p', 'english', sentence.gloss));
    const notes = notesList(sentence.notes);
    if (notes) card.append(notes);
    if (sentence.entry) card.append(entryBlock(sentence.entry));
    return card;
  }

  const body = el('div', 'reading');
  const details = el('details', 'parse');
  let at = 0;

  // Previous and next reading, likeliest first
  let nav;
  if (readings.length > 1) {
    nav = el('div', 'readings');
    const back = el('button', 'arrow', '←');
    const next = el('button', 'arrow', '→');
    const count = el('span', 'count');
    back.type = next.type = 'button';
    back.setAttribute('aria-label', 'Previous reading');
    next.setAttribute('aria-label', 'Next reading');
    count.setAttribute('aria-live', 'polite');
    const move = (step) => {
      at = Math.min(readings.length - 1, Math.max(0, at + step));
      render();
    };
    back.addEventListener('click', () => move(-1));
    next.addEventListener('click', () => move(1));
    nav.append(back, count, next);
    nav.update = () => {
      count.textContent = `Reading ${at + 1} of ${readings.length}${at === 0 ? ', the likeliest' : ''}`;
      back.disabled = at === 0;
      next.disabled = at === readings.length - 1;
    };
  }

  // Only the reading's own parts change; whether How it parses is open stays as it was.
  function render() {
    const reading = readings[at];
    const parts = [el('p', 'english', reading.english)];
    if (nav) {
      nav.update();
      parts.push(nav);
    }
    // A word on its own gets its dictionary entry; the note that it has no verb says nothing useful there.
    const notes = notesList(sentence.entry ? reading.notes.filter((note) => note.code !== 'bare-fragment') : reading.notes);
    if (notes) parts.push(notes);
    body.replaceChildren(...parts);
    const list = el('ul', 'tree');
    list.append(tree(reading.tree));
    details.replaceChildren(el('summary', '', 'How it parses'), list);
  }
  render();

  card.append(body);
  if (sentence.entry) card.append(entryBlock(sentence.entry));
  card.append(details);
  return card;
}

function show(sentences) {
  results.replaceChildren(...sentences.map(sentenceCard));
}

function translate() {
  const text = input.value.trim();
  if (!loaded) return;
  if (!text) {
    results.replaceChildren();
    return;
  }
  latest += 1;
  worker.postMessage({ id: latest, text });
}

worker.addEventListener('message', ({ data }) => {
  if (data.stage !== undefined) {
    status.textContent = data.stage;
  } else if (data.ready) {
    loaded = true;
    go.disabled = false;
    status.textContent = '';
    translate();
  } else if (data.failed) {
    status.textContent = 'The translator didn’t load. Reload the page to try again.';
  } else if (data.id === latest) {
    if (data.error) status.textContent = 'Something went wrong with that one.';
    else {
      status.textContent = '';
      show(data.result);
    }
  }
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  translate();
});

// Translate as you type, once you pause
let timer;
input.addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(translate, 300);
});

