// Runs the built app.zip in Pyodide under Node, the way the page's worker does, and checks a few sentences. It catches
// what the Python tests can't: a module or data file missing from the bundle, or code Pyodide's Python can't run.
//   (cd web && npm ci) && .venv/bin/python web/build.py && node web/test.mjs
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { loadPyodide } from 'pyodide';

const pyodide = await loadPyodide();
const zip = await readFile(new URL('./dist/app.zip', import.meta.url));
// A Node Buffer isn't a type Pyodide knows; a plain Uint8Array over the same bytes is.
pyodide.unpackArchive(new Uint8Array(zip.buffer, zip.byteOffset, zip.byteLength), 'zip', { extractDir: '/app' });
pyodide.runPython("import sys; sys.path.insert(0, '/app')");
const api = pyodide.pyimport('api');
const analyse = (text) => JSON.parse(api.analyse(text));

const started = performance.now();
const [{ readings: [first] }] = analyse('mi moku e kili.');
console.log(`test: first sentence in ${Math.round(performance.now() - started)} ms`);
assert.equal(first.english, 'I eat the fruit.');
assert.equal(first.valid, true);
assert.equal(first.tree.kind, 'Utterance · Sentence · Clause');
assert.deepEqual(first.tree.children.map((child) => child.kind), ['Subject', 'Verb and object']);

const cases = [
  ['soweli lili li lape lon tomo mi.', 'The small animal sleeps in my building.'],
  ['o pana e telo tawa kasi.', 'Give the water to the plant.'],
  ['tenpo pini la mi wile tawa ma Kanata.', 'In the past, I want to go to Canada.'],
];
for (const [toki, english] of cases) assert.equal(analyse(toki)[0].readings[0].english, english);

const [{ readings: [comma] }] = analyse('mi moku, e moku.');
assert.equal(comma.valid, false);
assert.ok(comma.notes.some((note) => note.severity === 'ERROR'));

// Every reading, likeliest first, each with its own English and notes
const [{ readings }] = analyse('sina sona ala sona e toki pona?');
assert.ok(readings.length > 1);
assert.equal(readings[0].english, 'Do you know toki pona?');
assert.ok(readings.every((reading) => reading.english && reading.tree && Array.isArray(reading.notes)));
assert.equal(new Set(readings.map((reading) => reading.english)).size, readings.length);

// One word on its own brings up its dictionary entry
const [{ entry }] = analyse('pona');
assert.equal(entry.word, 'pona');
assert.ok(entry.senses.some((sense) => sense.pos === 'noun'));
assert.equal(analyse('mi moku.')[0].entry, null);

const two = analyse('mi moku. sina lape.');
assert.equal(two.length, 2);

const [nonsense] = analyse('li li li');
assert.equal(nonsense.readings.length, 0);
assert.ok(nonsense.gloss);
assert.ok(nonsense.notes.some((note) => note.severity === 'ERROR'));

const timed = performance.now();
analyse('jan pona mi li toki e ni: sina pona.');
console.log(`test: a later sentence in ${Math.round(performance.now() - timed)} ms`);
console.log('test: all passed');
