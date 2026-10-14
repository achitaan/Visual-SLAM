// Build the saved gallery from the canonical report plots; avoid duplicate Git assets.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const entries = JSON.parse(fs.readFileSync(path.join(root, 'data/saved-visuals.json'), 'utf8'));
const destination = path.join(root, 'public/saved-visuals');
fs.mkdirSync(destination, { recursive: true });
const escape = text => String(text).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const sections = entries.map(entry => {
  if (path.basename(entry.file) !== entry.file) throw new Error('Invalid plot filename');
  fs.copyFileSync(path.join(root, '../docs/benchmark/plots', entry.file), path.join(destination, entry.file));
  return `<section><p>${escape(entry.group)}</p><h2>${escape(entry.label)}</h2><a href="/saved-visuals/${escape(entry.file)}"><img loading="lazy" src="/saved-visuals/${escape(entry.file)}"></a></section>`;
});
fs.writeFileSync(path.join(root, 'public/benchmark-graphs.html'), `<!doctype html><html><meta charset="utf-8"><title>Visual-SLAM benchmark graphs</title><style>body{font:16px system-ui;margin:40px;background:#101720;color:#edf2f7}main{max-width:1200px;margin:auto}img{width:100%;background:white}section{margin:48px 0}a{color:#82c8ff}</style><main><h1>Visual-SLAM benchmark graphs</h1><p>Saved experimental results. Current progress is available on the dashboard. Stereo uses SE(3) alignment; monocular uses evaluation-only Sim(3) scale fitting. Baselines and rejected experiments are labeled separately.</p>${sections.join('\n')}</main></html>`);
