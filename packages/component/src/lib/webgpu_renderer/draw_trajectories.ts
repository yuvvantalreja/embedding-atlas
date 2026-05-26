// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { AuxiliaryResources } from "./renderer.js";
import { gpuBuffer, gpuBufferData } from "./utils.js";

/** Per-segment GPU layout: 3 × vec4<f32> = 48 bytes.
 *  Floats: [x0, y0, x1, y1, r, g, b, _, width_px, alpha0, alpha1, is_jump]
 *  - (r, g, b) is the segment's color (pre-gamma-encoded, premultiplication
 *    happens in the fragment shader once alpha is known).
 *  - (alpha0, alpha1) are the segment's *endpoint* alphas, already including
 *    both the trajectory opacity and the tail→head modulation. The fragment
 *    shader interpolates between them along the segment.
 *  - `is_jump` is 0 or 1; jumps are rendered as a dashed, dimmed stroke. */
const SEGMENT_FLOATS = 12;
export const TRAJECTORY_SEGMENT_BYTES = SEGMENT_FLOATS * 4;

/** Default tail→head alpha modulation when the trajectory doesn't override.
 *  Segment alpha at tail end = `opacity * TAIL_ALPHA_SCALE`,
 *  at head end = `opacity * 1`. */
const TAIL_ALPHA_SCALE_DEFAULT = 0.2;
/** Default jump threshold: a segment is "long" if its length in data
 *  coordinates is more than this multiple of its trajectory's median
 *  segment length. */
const JUMP_THRESHOLD_DEFAULT = 5;

/** A trajectory ready for upload — endpoints are in data coordinates. */
export interface RendererTrajectory {
  /** Ordered points in data coordinates. */
  points: { x: number; y: number }[];
  /** Linear sRGB color components in [0, 1] (gamma_correction handles encoding). */
  color: { r: number; g: number; b: number };
  /** Optional per-segment colors (length = `points.length - 1`). */
  segmentColors?: ({ r: number; g: number; b: number } | null)[];
  /** Stroke width in CSS pixels. */
  width: number;
  /** Stroke opacity in [0, 1]. */
  opacity: number;
  /** Multiplier on opacity at the trajectory's tail. */
  tailAlphaScale?: number;
  /** Length ratio above which a segment is rendered as a "jump". */
  jumpThreshold?: number;
}

/** Compute the median of a small, possibly-zero-length number array.
 *  Returns 0 when empty. Allocates one copy; fine for the modest segment
 *  counts (hundreds–thousands) we deal with per trajectory. */
function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) * 0.5 : sorted[mid];
}

export interface TrajectoryResources {
  /** GPU storage buffer holding all segments across all trajectories. */
  segmentBuffer: Node<GPUBuffer>;
  /** Number of segments uploaded — pass to draw(4, count). */
  segmentCount: Node<number>;
  /** Bind group layout exposing only binding 3 of group 1 (the segment buffer). */
  bindGroupLayout: Node<GPUBindGroupLayout>;
  /** Bind group bound at group 1 of the trajectory pipeline. */
  bindGroup: Node<GPUBindGroup>;
}

/**
 * Build a flat Float32Array containing per-segment data ready for upload.
 *
 * Each segment is laid out as `[x0, y0, x1, y1, r, g, b, a, width_px, t0, t1, is_jump]`.
 * Colors are gamma-encoded into linear render space using `gamma`, matching
 * how `category_colors` are pre-encoded in the points pipeline.
 *
 * Trajectory-relative `t` runs from 0 at the first valid point to 1 at the
 * last; gaps (non-finite points) preserve the parametrization so the visible
 * subsegments still fade tail→head along the *full* polyline.
 *
 * Segments whose data-space length exceeds `jumpThreshold × medianLen` are
 * flagged `is_jump = 1` so the fragment shader can render them dashed.
 */
export function buildTrajectorySegmentData(
  trajectories: RendererTrajectory[] | null,
  pixelRatio: number,
  gamma: number,
): Float32Array<ArrayBuffer> {
  if (trajectories == null || trajectories.length == 0) {
    return new Float32Array(new ArrayBuffer(0));
  }

  let segmentCount = 0;
  for (const t of trajectories) {
    if (t.points.length >= 2) {
      segmentCount += t.points.length - 1;
    }
  }

  const data = new Float32Array(new ArrayBuffer(segmentCount * SEGMENT_FLOATS * 4));
  let off = 0;
  for (const t of trajectories) {
    if (t.points.length < 2) {
      continue;
    }
    const widthPx = Math.max(0.5, t.width * pixelRatio);
    const a = Math.max(0, Math.min(1, t.opacity));
    const tailScale = Math.max(0, Math.min(1, t.tailAlphaScale ?? TAIL_ALPHA_SCALE_DEFAULT));
    const baseR = Math.pow(Math.max(0, Math.min(1, t.color.r)), gamma);
    const baseG = Math.pow(Math.max(0, Math.min(1, t.color.g)), gamma);
    const baseB = Math.pow(Math.max(0, Math.min(1, t.color.b)), gamma);

    const segColors = t.segmentColors;
    const hasSegColors = segColors != null && segColors.length === t.points.length - 1;

    // Pre-compute per-segment lengths in data coordinates so we can derive a
    // robust outlier threshold (median × jumpThreshold). Non-finite endpoints
    // mark a gap and contribute no length.
    const lengths = new Float64Array(t.points.length - 1);
    const lengthsForMedian: number[] = [];
    for (let i = 1; i < t.points.length; i++) {
      const p0 = t.points[i - 1];
      const p1 = t.points[i];
      if (!Number.isFinite(p0.x) || !Number.isFinite(p0.y) || !Number.isFinite(p1.x) || !Number.isFinite(p1.y)) {
        lengths[i - 1] = NaN;
        continue;
      }
      const dx = p1.x - p0.x;
      const dy = p1.y - p0.y;
      const len = Math.hypot(dx, dy);
      lengths[i - 1] = len;
      lengthsForMedian.push(len);
    }
    const medianLen = median(lengthsForMedian);
    const jumpThreshold = t.jumpThreshold ?? JUMP_THRESHOLD_DEFAULT;
    // medianLen == 0 (e.g. all-zero-length polyline) → never flag a jump.
    const jumpAbsThreshold = medianLen > 0 ? medianLen * jumpThreshold : Infinity;

    // Parametrize t ∈ [0, 1] over the trajectory length. Index 0 is the tail,
    // index points.length - 1 is the head. We use index-based parametrization
    // (rather than arc length) because it matches the user's intuition that
    // "step 0 is the start" — long jumps shouldn't dominate the gradient.
    const denom = Math.max(1, t.points.length - 1);

    for (let i = 1; i < t.points.length; i++) {
      const p0 = t.points[i - 1];
      const p1 = t.points[i];
      // Skip segments with non-finite endpoints — they would produce NaN
      // matrix outputs and corrupt the strip. (Gaps in the trajectory.)
      if (!Number.isFinite(p0.x) || !Number.isFinite(p0.y) || !Number.isFinite(p1.x) || !Number.isFinite(p1.y)) {
        continue;
      }

      // Per-segment color override: use the destination point's color when
      // provided so segments inherit the action/state they led to.
      let r = baseR;
      let g = baseG;
      let b = baseB;
      if (hasSegColors) {
        const sc = segColors![i - 1];
        if (sc != null) {
          r = Math.pow(Math.max(0, Math.min(1, sc.r)), gamma);
          g = Math.pow(Math.max(0, Math.min(1, sc.g)), gamma);
          b = Math.pow(Math.max(0, Math.min(1, sc.b)), gamma);
        }
      }

      const u0 = (i - 1) / denom;
      const u1 = i / denom;
      // mix(tailScale, 1, u): tail = tailScale × opacity, head = full opacity.
      const alpha0 = a * (tailScale + (1 - tailScale) * u0);
      const alpha1 = a * (tailScale + (1 - tailScale) * u1);
      const isJump = lengths[i - 1] > jumpAbsThreshold ? 1 : 0;

      data[off + 0] = p0.x;
      data[off + 1] = p0.y;
      data[off + 2] = p1.x;
      data[off + 3] = p1.y;
      data[off + 4] = r;
      data[off + 5] = g;
      data[off + 6] = b;
      data[off + 7] = 0;
      data[off + 8] = widthPx;
      data[off + 9] = alpha0;
      data[off + 10] = alpha1;
      data[off + 11] = isJump;
      off += SEGMENT_FLOATS;
    }
  }

  // Trim if we skipped any malformed segments.
  if (off < data.length) {
    return data.subarray(0, off) as Float32Array<ArrayBuffer>;
  }
  return data;
}

export { TAIL_ALPHA_SCALE_DEFAULT, JUMP_THRESHOLD_DEFAULT };

/**
 * Allocates and updates the GPU storage buffer that backs trajectory rendering.
 * The buffer auto-grows when the segment count increases.
 */
export function makeTrajectoryResources(
  df: Dataflow,
  device: Node<GPUDevice>,
  segmentData: Node<Float32Array<ArrayBuffer>>,
): TrajectoryResources {
  const segmentCount = df.derive([segmentData], (data) => Math.floor(data.length / SEGMENT_FLOATS));

  // Allocate at least one segment's worth so the buffer is never zero-sized.
  const bufferByteSize = df.derive([segmentCount], (count) => Math.max(1, count) * TRAJECTORY_SEGMENT_BYTES);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

  const segmentBufferStorage = df.statefulDerive([device, bufferByteSize, usage], gpuBuffer);
  const segmentBuffer = df.statefulDerive([device, segmentBufferStorage, segmentData], gpuBufferData);

  const bindGroupLayout = df.derive([device], (device) =>
    device.createBindGroupLayout({
      entries: [{ binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
    }),
  );

  const bindGroup = df.derive([device, bindGroupLayout, segmentBuffer], (device, layout, buffer) =>
    device.createBindGroup({
      layout,
      entries: [{ binding: 3, resource: { buffer } }],
    }),
  );

  return { segmentBuffer, segmentCount, bindGroupLayout, bindGroup };
}

/**
 * Build the trajectory render pass. Trajectories are blended additively into
 * the same color/alpha textures used by points so that gamma_correction
 * composites them with the rest of the scene.
 *
 * The pass uses `loadOp: "load"` to preserve already-rendered points; it must
 * therefore run after `drawPoints` / `drawDensityMap` and before
 * `gammaCorrection`.
 */
export function makeDrawTrajectoriesCommand(
  df: Dataflow,
  device: Node<GPUDevice>,
  module: Node<GPUShaderModule>,
  group0Layout: Node<GPUBindGroupLayout>,
  trajectoryResources: TrajectoryResources,
  group0BindGroup: Node<GPUBindGroup>,
  auxiliaryResources: AuxiliaryResources,
): Node<(encoder: GPUCommandEncoder) => void> {
  const pipeline = df.derive(
    [device, module, group0Layout, trajectoryResources.bindGroupLayout],
    (device, module, group0, group1Trajectory) =>
      device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [group0, group1Trajectory] }),
        vertex: { entryPoint: "trajectory_vs", module: module },
        fragment: {
          entryPoint: "trajectory_fs",
          module: module,
          targets: [
            {
              format: auxiliaryResources.colorTextureFormat,
              blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
            },
            {
              format: auxiliaryResources.alphaTextureFormat,
              blend: { color: { srcFactor: "one", dstFactor: "one" }, alpha: { srcFactor: "one", dstFactor: "one" } },
            },
          ],
        },
        primitive: { topology: "triangle-strip" },
      }),
  );

  return df.derive(
    [
      pipeline,
      group0BindGroup,
      trajectoryResources.bindGroup,
      trajectoryResources.segmentCount,
      auxiliaryResources.colorTexture,
      auxiliaryResources.alphaTexture,
    ],
    (pipeline, group0, group1, count, colorTexture, alphaTexture) => (encoder) => {
      if (count <= 0) {
        return;
      }
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { loadOp: "load", storeOp: "store", view: colorTexture.createView() },
          { loadOp: "load", storeOp: "store", view: alphaTexture.createView() },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, group1);
      pass.draw(4, count);
      pass.end();
    },
  );
}
