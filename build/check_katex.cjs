// check_katex.cjs -- reject raw '<' in markup math, then compile every $..$ / $$..$$ in index.html (outside script/style/svg/code) with KaTeX, throwOnError.
const fs = require('fs'); const katex = require('../vendor/katex/katex.min.js');
let h = fs.readFileSync(process.argv[2] || 'index.html', 'utf8');
const facts = []; // math produced at runtime by the facts template
for (const m of h.matchAll(/setMath\('[\w-]+', `([\s\S]*?)`\);/g)) facts.push(...[...m[1].replace(/\$\{[^}]*\}/g, '1.0').matchAll(/\$([^$]+)\$/g)].map(x => x[1].replace(/\\\\/g, '\\')));
h = h.replace(/<(script|style|svg|code|pre)\b[\s\S]*?<\/\1>/g, ' ');
const out = []; const re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g; let x;
while ((x = re.exec(h))) out.push([x[1] ? true : false, (x[1] || x[2])]);
let bad = 0; const all = out.concat(facts.map(f => [false, f]));
for (const [disp, t] of out) if (t.includes('<')) { bad++; console.log('FAIL raw < in markup math (the browser reads it as a tag):', JSON.stringify(t.slice(0, 60))); }
for (const [disp, t] of all) {
  const tex = t.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  try { katex.renderToString(tex, {displayMode: disp, throwOnError: true, strict: 'ignore'}); }
  catch (e) { bad++; console.log('FAIL', JSON.stringify(tex.slice(0, 90)), e.message.slice(0, 120)); }
}
console.log('math', all.length, '(static', out.length, '+ runtime', facts.length, ') failures', bad); process.exit(bad ? 1 : 0);
