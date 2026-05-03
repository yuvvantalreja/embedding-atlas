// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Dataflow, Node } from "../dataflow.js";
import type { AuxiliaryResources } from "./renderer.js";
import { gpuBuffer, gpuBufferData } from "./utils.js";

/** Per-segment GPU layout: 3 × vec4<f32> = 48 bytes. */
const SEGMENT_FLOATS = 12;
export const TRAJECTORY_SEGMENT_BYTES = SEGMENT_FLOATS * 4;

/** A trajectory ready for upload — endpoints are in data coordinates. */
export interface RendererTrajectory {
  /** Ordered points in data coordinates. */
  points: { x: number; y: number }[];
  /** Linear sRGB color components in [0, 1] (gamma_correction handles encoding). */
  color: { r: number; g: number; b: number };
  /** Stroke width in CSS pixels. */
  width: number;
  /** Stroke opacity in [0, 1]. */
  opacity: number;
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
 * Each segment is laid out as `[x0, y0, x1, y1, r, g, b, a, width_px, 0, 0, 0]`.
 * Colors are gamma-encoded into linear render space using `gamma`, matching
 * how `category_colors` are pre-encoded in the points pipeline.
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
    const r = Math.pow(Math.max(0, Math.min(1, t.color.r)), gamma);
    const g = Math.pow(Math.max(0, Math.min(1, t.color.g)), gamma);
    const b = Math.pow(Math.max(0, Math.min(1, t.color.b)), gamma);
    for (let i = 1; i < t.points.length; i++) {
      const p0 = t.points[i - 1];
      const p1 = t.points[i];
      // Skip segments with non-finite endpoints — they would produce NaN
      // matrix outputs and corrupt the strip.
      if (
        !Number.isFinite(p0.x) ||
        !Number.isFinite(p0.y) ||
        !Number.isFinite(p1.x) ||
        !Number.isFinite(p1.y)
      ) {
        continue;
      }
      data[off + 0] = p0.x;
      data[off + 1] = p0.y;
      data[off + 2] = p1.x;
      data[off + 3] = p1.y;
      data[off + 4] = r;
      data[off + 5] = g;
      data[off + 6] = b;
      data[off + 7] = a;
      data[off + 8] = widthPx;
      data[off + 9] = 0;
      data[off + 10] = 0;
      data[off + 11] = 0;
      off += SEGMENT_FLOATS;
    }
  }

  // Trim if we skipped any malformed segments.
  if (off < data.length) {
    return data.subarray(0, off) as Float32Array<ArrayBuffer>;
  }
  return data;
}

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

  const bindGroup = df.derive(
    [device, bindGroupLayout, segmentBuffer],
    (device, layout, buffer) =>
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
