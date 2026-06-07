// Post-build (wired into build:ts): pre-resolve the package-internal
// "#wasm" specifier in the emitted ESM to a relative path. Relative
// specifiers resolve natively in every context (pages, workers, node,
// bundlers); "#wasm" needs node-style resolution, which browsers only get
// from import maps - and import maps don't apply inside workers. Source
// keeps "#wasm" so typecheck works without pkg/ built (wasm-pkg.d.ts).
import { readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");
for (const name of await readdir(dist)) {
  if (!name.endsWith(".js")) {
    continue;
  }
  const path = join(dist, name);
  const src = await readFile(path, "utf8");
  const out = src.replaceAll('"#wasm"', '"../pkg/thinfer_web.js"');
  if (out !== src) {
    await writeFile(path, out);
  }
}
