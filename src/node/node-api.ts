// The few Node.js built-ins the Node entry uses, loaded at run time (the package builds without @types/node).
// Node >= 20.

interface Fs {
  readFileSync(p: string): Uint8Array;
  existsSync(p: string): boolean;
}
interface Path { join(...p: string[]): string; resolve(...p: string[]): string; dirname(p: string): string }
interface ModuleApi { createRequire(from: string): (id: string) => unknown }
interface UrlApi { fileURLToPath(u: string | URL): string }
interface Proc { platform: string; arch: string; env: Record<string, string | undefined>; versions: { node?: string } }

export interface NodeApis { fs: Fs; path: Path; module: ModuleApi; url: UrlApi; process: Proc }

let apis: Promise<NodeApis> | null = null;

/** node:fs, node:path, node:module, node:url and process (throws outside Node). */
export function nodeApis(): Promise<NodeApis> {
  apis ??= (async () => {
    const proc = (globalThis as unknown as { process?: Proc }).process;
    if (!proc?.versions?.node) throw new Error('zen-tts/node runs on Node.js (use the "zen-tts" entry in the browser)');
    const imp = (id: string): Promise<unknown> => import(/* @vite-ignore */ id);
    const [fs, path, module, url] = await Promise.all([imp('node:fs'), imp('node:path'), imp('node:module'), imp('node:url')]) as [Fs, Path, ModuleApi, UrlApi];
    return { fs, path, module, url, process: proc };
  })();
  return apis;
}
