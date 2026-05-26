// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type DataPointID = string | number | bigint;

export interface DataPoint {
  x: number;
  y: number;
  category?: number;
  text?: string;
  identifier?: DataPointID;
  fields?: Record<string, any>;
}

export type DataField = string | { sql: string };

export interface Cache {
  get: (key: string) => Promise<any | null>;
  set: (key: string, value: any) => Promise<void>;
}

/** The content of a label: either a text string or an image with display dimensions (and optionally x, y coordinates). */
export type LabelContent = string | { x?: number; y?: number; image: string; width: number; height: number };

export interface Label {
  /** X coordinate. */
  x: number;
  /** Y coordinate. */
  y: number;
  /** Label content: a text string or an image reference. */
  content: LabelContent;
  /** Label level. The label will be shown around 2^level zoom factor. */
  level?: number | null;
  /** Placement priority. */
  priority?: number | null;
}

export interface OverlayProxy {
  location: (x: number, y: number) => { x: number; y: number };
  width: number;
  height: number;
}

/** A trajectory: an ordered list of points in data coordinates to be connected with a polyline. */
export interface Trajectory {
  /** Ordered points in data coordinates. */
  points: { x: number; y: number }[];
  /** Optional stroke color (CSS color). Defaults to a generated color per trajectory. */
  color?: string;
  /** Optional stroke width in CSS pixels. Defaults to 1.2. */
  width?: number;
  /** Optional stroke opacity. Defaults to 0.25.
   *  The rendered alpha is also modulated tail→head: the start of a trajectory
   *  is drawn at `tailAlphaScale * opacity` and the end at full `opacity`,
   *  giving each polyline a visible sense of direction. */
  opacity?: number;
  /** Optional per-step CSS colors, parallel to `points`. When provided, each
   *  segment is colored using the destination point's color (i.e. segment
   *  `i → i+1` uses `stepColors[i + 1]`). Use this to color flow by an
   *  action/state that changes along the trajectory. Entries may be `null` at
   *  the same index as a gap (NaN) point; those segments are skipped. */
  stepColors?: (string | null | undefined)[];
  /** Optional identifier (e.g., episode id) — used for keying and default color hashing. */
  id?: string | number;
}

/** Reactive column-based trajectory spec. When provided to a Mosaic-aware view,
 *  trajectories are aggregated from the data table and re-aggregated under the
 *  active cross-filter, so filtering points in any chart also filters the
 *  polylines. Gaps in a group's filtered rows render as disconnected segments. */
export interface TrajectorySpec {
  /** Column identifying each trajectory (one polyline per distinct value). */
  group_by: string;
  /** Column determining step order within a trajectory. */
  order_by: string;
  /** Cap on the number of trajectories drawn. Defaults to 50.
   *  The largest groups (by row count) are kept. */
  max_groups?: number;
  /** Stroke width in CSS pixels. */
  width?: number;
  /** Stroke opacity in [0, 1]. */
  opacity?: number;
  /** Column whose value selects the color (looked up in `colors`).
   *  By default each segment is colored by its endpoint's value, so a polyline
   *  can change color along its length (e.g. by `action`). Set
   *  `color_per_segment` to `false` to use a single color per trajectory
   *  (taken from any row in the group). */
  color_by?: string;
  /** When `color_by` is set, controls whether each segment carries its own
   *  color (true, default) or the whole polyline uses one color (false). */
  color_per_segment?: boolean;
  /** Map from `color_by` value to CSS color. */
  colors?: Record<string, string>;
}

type CustomComponentClass<N, P> = new (node: N, props: P) => { update?: (props: P) => void; destroy?: () => void };

export type CustomComponent<N, P> =
  | {
      class: CustomComponentClass<N, P & any>;
      props?: Record<string, any>;
    }
  | CustomComponentClass<N, P>;
