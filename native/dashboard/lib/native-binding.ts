import type { TelemetryFrame } from './types';
export type NativeRunBinding = { url: string; runId: string; sequence: string; frames: number };
export function parseNativeBinding(search: string): NativeRunBinding {
  const params = new URLSearchParams(search), runId = params.get('run') ?? '';
  if (!/^native_prototype_[a-z0-9_]+$/.test(runId)) throw new Error('Open the link for the requested native run.');
  const sequence = params.get('sequence') ?? '', rawFrames = params.get('frames') ?? '';
  const frames = /^\d+$/.test(rawFrames) ? Number(rawFrames) : NaN;
  const prepared = sequence === '01' ? [350, 1101] : sequence === '04' ? [60, 80, 271] : [];
  if (!prepared.includes(frames)) throw new Error('The link must identify a prepared native sequence and frame count.');
  let url: URL;
  try { url = new URL(params.get('ws') ?? ''); }
  catch { throw new Error('The native backend endpoint is missing or invalid.'); }
  if (url.protocol !== 'ws:' || !['127.0.0.1', 'localhost'].includes(url.hostname) ||
      !/^\d+$/.test(url.port) || Number(url.port) < 1024 || Number(url.port) > 65535 ||
      url.username || url.password || url.pathname !== '/' || url.search || url.hash)
    throw new Error('Choose the local native endpoint supplied for this run.');
  return { url: url.toString(), runId, sequence, frames };
}
type RecordValue = Record<string, unknown>;
const object = (v: unknown): v is RecordValue => v !== null && typeof v === 'object' && !Array.isArray(v);
const nonnegative = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0;
const vector = (v: unknown, width: number): v is number[] => Array.isArray(v) && v.length === width && v.every(n => typeof n === 'number' && Number.isFinite(n));
const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
/** Display provenance check, never an estimator or accuracy verifier. */
export class NativeTelemetryGate {
  private path: number[][] = [];
  private pose: number[][] | undefined;
  private lastFrame = -1;
  private failed = false;
  private originAnchor: { mapId: number; generation: string } | undefined;
  private lastOrigin: string | undefined;
  reason = '';
  constructor(private binding: NativeRunBinding) {}
  private reject(reason: string): false { this.failed = true; this.reason = reason; return false; }
  accept(value: unknown): value is TelemetryFrame {
    if (this.failed) return false;
    if (!object(value) || value.schema_version !== 1 || value.mode !== 'slam' || value.mode_locked !== true ||
        value.translation_scale !== 'metric' || value.run_id !== this.binding.runId || value.sequence !== this.binding.sequence ||
        value.total_frames !== this.binding.frames) return this.reject('This endpoint did not identify the requested native run.');
    const index = value.frame_index, pose = value.pose_T_wc, path = value.trajectory;
    if (!nonnegative(index) || index >= this.binding.frames || index < this.lastFrame ||
        typeof value.timestamp !== 'number' || !Number.isFinite(value.timestamp)) return this.reject('Native frame order or coverage is invalid.');
    if (!Array.isArray(pose) || pose.length !== 4 || !pose.every(p => vector(p, 4)) || !same(pose[3], [0, 0, 0, 1]) ||
        !Array.isArray(path) || path.length !== index + 1 || !path.every(p => vector(p, 3))) return this.reject('Native pose or trajectory data is invalid.');
    if (!same(path[index], pose.slice(0, 3).map(p => p[3])) || this.path.some((p, i) => !same(p, path[i])) ||
        (index === this.lastFrame && this.pose?.some((p, i) => !same(p, pose[i])))) return this.reject('Previously emitted native poses changed.');
    if (!object(value.native_coverage) || value.native_coverage.emitted_original_rows !== index + 1 ||
        !nonnegative(value.native_coverage.held_rows) || value.native_coverage.held_rows > index + 1 ||
        value.native_coverage.display_delivery_is_not_coverage !== true || !object(value.tracking) ||
        value.pose_graph != null || value.expected_pose_T_wc != null) return this.reject('Native provenance or online history is invalid.');
    const origin = value.native_origin;
    const token = (v: unknown): v is string => typeof v === 'string' && /^(0|[1-9][0-9]{0,19})$/.test(v) && BigInt(v) < BigInt('18446744073709551615');
    if (!object(origin) || origin.version !== 2 || typeof origin.map_id !== 'number' || !Number.isSafeInteger(origin.map_id) || origin.map_id < -1 ||
        !token(origin.map_generation) || !token(origin.anchor_generation) || !nonnegative(value.tracking.native_outcome) || value.tracking.native_outcome > 4)
      return this.reject('Native coordinate-generation provenance is missing or invalid.');
    const accepted = value.tracking.native_outcome === 0 || value.tracking.native_outcome === 1;
    const originKey = `${origin.map_id}/${origin.map_generation}/${origin.anchor_generation}`;
    if (index === this.lastFrame && originKey !== this.lastOrigin) return this.reject('Committed native coordinate identity changed.');
    if (this.originAnchor) {
      if (origin.anchor_generation !== this.originAnchor.generation ||
          (accepted && (origin.map_id !== this.originAnchor.mapId || origin.map_generation !== this.originAnchor.generation)))
        return this.reject('Native pose changed its anchored coordinate generation.');
    } else if (accepted) {
      if (value.tracking.native_outcome !== 0 || origin.map_id < 0 || origin.map_generation === '0' || origin.anchor_generation !== origin.map_generation)
        return this.reject('Native origin was not initialized consistently.');
      this.originAnchor = { mapId: origin.map_id, generation: origin.map_generation };
    } else if (origin.anchor_generation !== '0') return this.reject('Uninitialized native origin claims an anchor.');
    if (!accepted && this.pose?.some((p, i) => !same(p, pose[i]))) return this.reject('A held native pose changed.');
    if (!object(value.map) || typeof value.map.available !== 'boolean') return this.reject('Native map availability is invalid.');
    const snapshot = value.native_snapshot;
    if (value.map.available) {
      if (!nonnegative(value.map.keyframes) || !nonnegative(value.map.map_points) || !Array.isArray(value.map_points) ||
          value.map_points.length > 5000 || value.map_points.length > value.map.map_points || !value.map_points.every(p => vector(p, 3)) ||
          !object(snapshot) || !nonnegative(snapshot.original_id) || snapshot.original_id > index ||
          snapshot.original_id !== value.map.snapshot_original_id || snapshot.keyframes !== value.map.keyframes ||
          snapshot.map_points !== value.map.map_points || !nonnegative(snapshot.map_id) || !nonnegative(snapshot.request_ns) ||
          !nonnegative(snapshot.capture_ns) || !nonnegative(snapshot.completion_ns) || snapshot.request_ns > snapshot.capture_ns ||
          snapshot.capture_ns > snapshot.completion_ns) return this.reject('Native map snapshot provenance is invalid.');
    } else if (value.map.keyframes !== null || value.map.map_points !== null || snapshot != null || value.map_points != null)
      return this.reject('Unavailable native map counts must remain unavailable.');
    if (value.map.available && (!object(snapshot) || !this.originAnchor || snapshot.map_id !== this.originAnchor.mapId || snapshot.map_generation !== this.originAnchor.generation))
      return this.reject('Native map changed its anchored coordinate generation.');
    this.lastOrigin = originKey;
    this.path = path.map(p => [...p]); this.pose = pose.map(p => [...p]); this.lastFrame = index;
    return true;
  }
}
