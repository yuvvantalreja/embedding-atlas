// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Point, ViewportState } from "./utils.js";

export type RenderMode = "points" | "density";

/**
 * A trajectory ready for rendering: ordered points in data coordinates with
 * a resolved color in linear sRGB, stroke width in CSS pixels, and opacity.
 *
 * The view layer is responsible for resolving the user-facing `Trajectory`
 * (CSS color string, optional defaults) into this concrete renderer type.
 */
export interface RendererTrajectory {
  points: { x: number; y: number }[];
  /** Default trajectory color, used for any segment that doesn't have its
   *  own entry in `segmentColors`. */
  color: { r: number; g: number; b: number };
  /** Optional per-segment colors, length = `points.length - 1`. An entry of
   *  `null` falls back to `color`. */
  segmentColors?: ({ r: number; g: number; b: number } | null)[];
  width: number;
  opacity: number;
  /** Multiplier on `opacity` applied at the trajectory's tail (its first
   *  point); the head (last point) is drawn at full opacity. Values in
   *  `[0, 1)` produce a directional fade — the default of 0.2 makes the
   *  start of each polyline noticeably dimmer than its end. */
  tailAlphaScale?: number;
  /** Length ratio above which a segment is treated as a "jump" (rendered
   *  dashed, dimmed). Compared against each trajectory's median segment
   *  length. Default 5; pass `Infinity` to disable. */
  jumpThreshold?: number;
}

export interface EmbeddingRendererProps {
  mode: RenderMode;
  colorScheme: "light" | "dark";

  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  category: Uint8Array<ArrayBuffer> | null;

  categoryCount: number;
  categoryColors: string[] | null;

  viewportX: number;
  viewportY: number;
  viewportScale: number;

  pointSize: number;
  pointAlpha: number;
  pointsAlpha: number;

  densityScaler: number;
  densityBandwidth: number;
  densityQuantizationStep: number;
  densityAlpha: number;
  contoursAlpha: number;

  gamma: number;
  width: number;
  height: number;

  /** Approximate maximum points to render. null/Infinity = no limit. Default: 4,000,000 */
  downsampleMaxPoints: number | null;
  /** Density weight for downsampling (0-10). Default: 5 */
  downsampleDensityWeight: number;

  /** Optional polylines to overlay on top of points/density. */
  trajectories: RendererTrajectory[] | null;
  /** Pixel ratio used to convert CSS-pixel widths to framebuffer pixels. */
  pixelRatio: number;
}

export interface DensityMap {
  data: Float32Array;
  width: number;
  height: number;
  coordinateAtPixel: (x: number, y: number) => Point;
}

export interface EmbeddingRenderer {
  readonly props: EmbeddingRendererProps;

  /** Set renderer props. Returns true if a render is needed. */
  setProps(newProps: Partial<EmbeddingRendererProps>): boolean;

  /** Render */
  render(): void;

  /** Destroy the renderer and free any resource */
  destroy(): void;

  /** Produce a density map */
  densityMap(width: number, height: number, radius: number, viewportState: ViewportState): Promise<DensityMap>;
}
