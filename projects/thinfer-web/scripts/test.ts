// Runs the thinfer-web wasm-bindgen-test suite against a project-local Chromium
// (installed by Playwright into its user-cache dir) driven by the npm
// `chromedriver` package. Wires the two together via a generated webdriver.json
// so wasm-bindgen-test-runner uses our Chromium and not whatever the system has.
//
// Assumes Playwright's Chromium is already installed (the `test:web` package.json
// script runs `playwright install chromium` first).
//
// Steps:
//   1. Resolve Playwright's Chromium binary and read its version.
//   2. Resolve chromedriver's binary and version (from the npm package).
//   3. Assert majors match. Mismatch is fatal with a clear bump-this-dep message.
//   4. Merge `webdriver.template.json` + `goog:chromeOptions.binary = <chromium>`
//      into `webdriver.json` (gitignored), where wasm-bindgen-test-runner reads
//      capabilities.
//   5. Spawn `cargo test --target wasm32-unknown-unknown -p thinfer-web` with
//      CHROMEDRIVER set; forward args and exit code.

import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = dirname(here);

function run(cmd: string, args: string[], env?: NodeJS.ProcessEnv): number {
  const r = spawnSync(cmd, args, {
    stdio: "inherit",
    env: env ? { ...process.env, ...env } : process.env,
    shell: false,
  });
  if (r.error) throw r.error;
  return r.status ?? 1;
}

function major(v: string): number {
  const m = v.match(/(\d+)\./);
  if (!m) throw new Error(`cannot parse major from version: ${v}`);
  return Number(m[1]);
}

// 1. Chromium binary + version. We read the version from playwright-core's
// `browsers.json` rather than invoking `<chromium> --version`: on Windows that
// invocation pops a real browser window instead of printing-and-exiting.
const chromiumBin: string = chromium.executablePath();
// playwright-core is a transitive dep of playwright; with pnpm it is not
// hoisted into our top-level node_modules. Resolve it via playwright's own
// resolution paths.
const playwrightDir = dirname(require.resolve("playwright/package.json"));
const browsersJsonPath = join(
  dirname(require.resolve("playwright-core/package.json", { paths: [playwrightDir] })),
  "browsers.json",
);
const browsersJson = JSON.parse(readFileSync(browsersJsonPath, "utf8")) as {
  browsers: { name: string; browserVersion: string }[];
};
const chromiumEntry = browsersJson.browsers.find((b) => b.name === "chromium");
if (!chromiumEntry) throw new Error("chromium entry missing from browsers.json");
const chromiumVersion = chromiumEntry.browserVersion;

// 2. Chromedriver binary + version.
const cd = require("chromedriver") as { path: string; version: string };

// 3. Major-version sanity check.
if (major(chromiumVersion) !== major(cd.version)) {
  console.error(
    `chromedriver / chromium major mismatch:\n` +
      `  chromium    ${chromiumVersion}  (${chromiumBin})\n` +
      `  chromedriver ${cd.version}  (${cd.path})\n` +
      `Bump the chromedriver devDep to ^${major(chromiumVersion)}.0.0 ` +
      `(or roll Playwright forward/back to match chromedriver).`,
  );
  process.exit(1);
}

// 4. Generate webdriver.json from template + binary path.
const template = JSON.parse(readFileSync(join(pkgDir, "webdriver.template.json"), "utf8")) as {
  "goog:chromeOptions"?: Record<string, unknown>;
};
const chromeOpts = (template["goog:chromeOptions"] ?? {}) as Record<string, unknown>;
chromeOpts.binary = chromiumBin;
// CI runners have no GPU; point Chrome's WebGPU at SwiftShader (its bundled
// software implementation) so an adapter exists at all. Local runs keep the
// real GPU.
if (process.env.CI) {
  const args = (chromeOpts.args ?? []) as string[];
  args.push("--use-webgpu-adapter=swiftshader");
  chromeOpts.args = args;
}
template["goog:chromeOptions"] = chromeOpts;
writeFileSync(join(pkgDir, "webdriver.json"), `${JSON.stringify(template, null, 2)}\n`);

// 5. cargo test. Always --nocapture: wasm-bindgen-test-runner only replays
// captured console output for tests that REPORT failure; a panic that kills
// the harness before reporting ("Failed to detect test as having been run")
// silently swallows the actual error otherwise.
const cargoArgs = [
  "test",
  "--target",
  "wasm32-unknown-unknown",
  "-p",
  "thinfer-web",
  ...process.argv.slice(2),
  "--",
  "--nocapture",
];
process.exit(run("cargo", cargoArgs, { CHROMEDRIVER: cd.path }));
