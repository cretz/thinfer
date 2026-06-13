// Static server for the example: serves the thinfer-web package root so the
// page can reach /examples/gen-image/, /dist/ (tsc output), and /pkg/
// (wasm-bindgen output). No deps; Node 24 runs the TS directly. The one thing
// generic one-liner servers get wrong is the .wasm MIME type, which
// `instantiateStreaming` requires.
//
// Run: pnpm example:gen-image (PORT env to override the default 8787).
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const port = Number(process.env.PORT ?? 8787);

const MIME: Record<string, string> = {
  ".css": "text/css",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".json": "application/json",
  ".map": "application/json",
  ".png": "image/png",
  ".wasm": "application/wasm",
};

createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
    let file = resolve(root, `.${path}`);
    // resolve() collapses any `..`; reject anything that escaped the root.
    if (file !== root && !file.startsWith(root + sep)) {
      res.writeHead(403).end();
      return;
    }
    let s = await stat(file).catch(() => null);
    if (s?.isDirectory()) {
      if (!path.endsWith("/")) {
        // Redirect so the page's relative refs (./main.js) resolve under the
        // directory, not its parent.
        res.writeHead(301, { location: `${path}/` }).end();
        return;
      }
      file = join(file, "index.html");
      s = await stat(file).catch(() => null);
    }
    if (!s?.isFile()) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, {
      "content-type": MIME[extname(file)] ?? "application/octet-stream",
      "content-length": s.size,
    });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(500).end();
  }
}).listen(port, () => {
  console.log(`serving ${root}`);
  console.log(`open http://localhost:${port}/examples/gen-image/`);
});
