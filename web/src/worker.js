// Runs the Python in Pyodide, off the page's main thread so typing never stalls. It loads Pyodide and app.zip (built by
// web/build.py), imports web/api.py, and answers each {id, text} with {id, result} or {id, error}.
import { loadPyodide } from './pyodide/pyodide.mjs';

// Each stage is reported to the page, which shows it while the translator loads.
const stage = (text) => postMessage({ stage: text, at: Math.round(performance.now()) });

const ready = (async () => {
  stage('Loading Python…');
  const appZip = fetch(new URL('./app.zip', import.meta.url));
  const pyodide = await loadPyodide({ indexURL: new URL('./pyodide/', import.meta.url).href });
  stage('Loading the dictionary and grammar…');
  const response = await appZip;
  if (!response.ok) throw new Error(`app.zip: ${response.status}`);
  pyodide.unpackArchive(await response.arrayBuffer(), 'zip', { extractDir: '/app' });
  pyodide.runPython("import sys; sys.path.insert(0, '/app')");
  const api = pyodide.pyimport('api');
  // The first call builds the lexicon; do it now rather than on the first sentence.
  api.analyse('mi moku.');
  stage('');
  return api;
})();

ready.then(
  () => postMessage({ ready: true }),
  (error) => postMessage({ failed: String(error) }),
);

self.addEventListener('message', async ({ data }) => {
  try {
    const api = await ready;
    postMessage({ id: data.id, result: JSON.parse(api.analyse(data.text)) });
  } catch (error) {
    postMessage({ id: data.id, error: String(error) });
  }
});
