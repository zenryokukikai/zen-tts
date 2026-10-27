// Getting a Web Audio context ready to play inside a tap. iOS leaves a page's AudioContext 'interrupted' (not
// 'suspended') after the user comes back from another app, and it can also become 'closed'; resume() can then reject or
// never settle.

/** The part of AudioContext used here. */
export interface ResumableContext {
  readonly state: string;
  resume(): Promise<void>;
  close(): Promise<void>;
}

export interface ReadyContext<C extends ResumableContext> {
  ctx: C;
  /** A new context was made: buffers decoded by the old one must be decoded again. */
  fresh: boolean;
  /** ctx.state is 'running': false means play through an <audio> element instead. */
  running: boolean;
}

export const RESUME_TIMEOUT_MS = 1500;

/** Resume with a time limit: true when it resolved, false when it rejected or did not settle in time. */
async function tryResume(ctx: ResumableContext, timeoutMs: number): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      ctx.resume().then(() => true, () => false),
      new Promise<boolean>((r) => { timer = setTimeout(() => r(false), timeoutMs); }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/** A context that is 'running' if the browser allows it: the current one resumed when it is not running; a new one
 *  when there is none, it is 'closed', or its resume failed (the old one is closed). Call from the tap handler. */
export async function readyAudioContext<C extends ResumableContext>(current: C | undefined, make: () => C,
  timeoutMs = RESUME_TIMEOUT_MS): Promise<ReadyContext<C>> {
  let ctx = current;
  let fresh = false;
  if (!ctx || ctx.state === 'closed') { ctx = make(); fresh = true; }
  if (ctx.state === 'running') return { ctx, fresh, running: true };
  const ok = await tryResume(ctx, timeoutMs);
  if ((!ok || ctx.state === 'closed') && !fresh) {
    void ctx.close().catch(() => {});
    ctx = make();
    fresh = true;
    if (ctx.state !== 'running') await tryResume(ctx, timeoutMs);
  }
  return { ctx, fresh, running: ctx.state === 'running' };
}
