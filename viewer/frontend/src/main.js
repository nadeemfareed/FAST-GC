/* FASTGC_VIEWER_CLEAN_V5 */

import "../style.css";

import {
  Box3,
  Color,
  MOUSE,
  Vector3,
} from "three";

import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import ColorMap
  from "@giro3d/giro3d/core/ColorMap.js";

import CoordinateSystem
  from "@giro3d/giro3d/core/geographic/CoordinateSystem.js";

import Instance
  from "@giro3d/giro3d/core/Instance.js";

import Extent
  from "@giro3d/giro3d/core/geographic/Extent.js";

import ElevationLayer
  from "@giro3d/giro3d/core/layer/ElevationLayer.js";

import ColorLayer
  from "@giro3d/giro3d/core/layer/ColorLayer.js";

import GiroMap
  from "@giro3d/giro3d/entities/Map.js";

import PointCloud
  from "@giro3d/giro3d/entities/PointCloud.js";

import {
  ASPRS_CLASSIFICATIONS,
  Classification,
} from "@giro3d/giro3d/renderer/PointCloudMaterial.js";

import GeoTIFFSource
  from "@giro3d/giro3d/sources/GeoTIFFSource.js";

import COPCSource
  from "@giro3d/giro3d/sources/COPCSource.js";


const $ = (id) =>
  document.getElementById(id);


const state = {
  runtime: null,
  crs: null,
  instance: null,
  controls: null,
  map: null,

  records: new Map(),
  visibleProducts: new Set(),

  activeProduct: null,

  pointBudget: 10_000_000,
  edl: true,
};


const PALETTES = {
  terrain: [
    "#2c7bb6",
    "#abd9e9",
    "#ffffbf",
    "#fdae61",
    "#8c510a",
  ],

  chm: [
    "#f7fcf5",
    "#c7e9c0",
    "#74c476",
    "#238b45",
    "#00441b",
  ],

  viridis: [
    "#440154",
    "#3b528b",
    "#21918c",
    "#5ec962",
    "#fde725",
  ],

  turbo: [
    "#30123b",
    "#4662d7",
    "#35ab8b",
    "#f9e721",
    "#a2142f",
  ],

  inferno: [
    "#000004",
    "#420a68",
    "#932667",
    "#dd513a",
    "#fcffa4",
  ],

  gray: [
    "#000000",
    "#404040",
    "#808080",
    "#bfbfbf",
    "#ffffff",
  ],
};


function setStatus(message) {
  $("status").textContent =
    String(message ?? "");
}


function colors(name) {
  return (
    PALETTES[name]
    ?? PALETTES.terrain
  ).map((value) => new Color(value));
}


function runtimeBounds() {
  const b =
    state.runtime?.bounds
    ?? state.runtime?.spatial?.bounds;

  if (!b) {
    throw new Error(
      "Runtime bounds are missing."
    );
  }

  const xmin =
    Number(b.xmin ?? b.minx ?? b.min?.[0]);

  const ymin =
    Number(b.ymin ?? b.miny ?? b.min?.[1]);

  const zmin =
    Number(b.zmin ?? b.minz ?? b.min?.[2] ?? 0);

  const xmax =
    Number(b.xmax ?? b.maxx ?? b.max?.[0]);

  const ymax =
    Number(b.ymax ?? b.maxy ?? b.max?.[1]);

  const zmax =
    Number(b.zmax ?? b.maxz ?? b.max?.[2] ?? 0);

  const values = [
    xmin, ymin, zmin,
    xmax, ymax, zmax,
  ];

  if (!values.every(Number.isFinite)) {
    throw new Error(
      "Runtime bounds are incomplete."
    );
  }

  return {
    xmin, ymin, zmin,
    xmax, ymax, zmax,
  };
}


function runtimeExtent() {
  const b = runtimeBounds();

  return new Extent(
    state.crs,
    b.xmin,
    b.xmax,
    b.ymin,
    b.ymax,
  );
}


function datasetBox() {
  const b = runtimeBounds();

  return new Box3(
    new Vector3(
      b.xmin,
      b.ymin,
      b.zmin,
    ),
    new Vector3(
      b.xmax,
      b.ymax,
      b.zmax,
    ),
  );
}


function layerSources(layer) {
  return Array.isArray(layer?.sources)
    ? layer.sources
    : [];
}


function sourceFor(layer, kind) {
  const candidates =
    layerSources(layer).filter(
      (source) =>
        source?.kind === kind
        && typeof source?.url === "string"
    );

  if (!candidates.length) {
    return null;
  }

  return (
    candidates.find(
      (source) =>
        source.preferred === true
    )
    ?? candidates.find(
      (source) =>
        source.role === "visualization"
    )
    ?? candidates.find(
      (source) =>
        /\.copc(?:\.laz)?$/i.test(
          source.filename ?? source.url
        )
    )
    ?? candidates[0]
  );
}


function productCapability(layer) {
  if (
    layer?.product === "FAST_DEM"
    && sourceFor(layer, "raster")
  ) {
    return "dem";
  }

  if (
    layer?.kind === "raster"
    && sourceFor(layer, "raster")
  ) {
    return "raster";
  }

  if (
    layer?.kind === "pointcloud"
    && sourceFor(layer, "pointcloud")
  ) {
    const source =
      sourceFor(layer, "pointcloud");

    if (
      source?.role === "authoritative"
      && !/\.copc(?:\.laz)?$/i.test(
        source.filename ?? source.url
      )
    ) {
      return null;
    }

    return "pointcloud";
  }

  return null;
}


async function ensureMap() {
  if (state.map) {
    return state.map;
  }

  const map =
    new GiroMap({
      extent: runtimeExtent(),
      backgroundColor: "#ffffff",
      lighting: true,
    });

  await state.instance.add(map);

  state.map = map;

  return map;
}


function displayStats(sourceInfo) {
  return sourceInfo?.display ?? null;
}


function rasterRange(
  sourceInfo,
  stretch,
  customMin = null,
  customMax = null,
) {
  const stats =
    displayStats(sourceInfo);

  if (!stats) {
    return null;
  }

  let min;
  let max;

  if (stretch === "full") {
    min = Number(stats.min);
    max = Number(stats.max);
  } else if (stretch === "custom") {
    min = Number(customMin);
    max = Number(customMax);
  } else {
    min = Number(stats.p02);
    max = Number(stats.p98);
  }

  if (
    !Number.isFinite(min)
    || !Number.isFinite(max)
    || max <= min
  ) {
    return null;
  }

  return { min, max };
}


function makeColorMap(
  palette,
  range,
) {
  if (!range) {
    return null;
  }

  return new ColorMap({
    min: range.min,
    max: range.max,
    colors: colors(palette),
  });
}


function defaultRasterDisplay(
  layer,
  sourceInfo,
) {
  const stats =
    displayStats(sourceInfo);

  return {
    palette:
      layer.product === "FAST_CHM"
        ? "chm"
        : "terrain",

    stretch: "auto",

    min:
      Number(stats?.p02 ?? stats?.min ?? 0),

    max:
      Number(stats?.p98 ?? stats?.max ?? 1),

    opacity: 1,
  };
}


async function createRasterRecord(
  layer,
  display = null,
) {
  const sourceInfo =
    sourceFor(layer, "raster");

  if (!sourceInfo) {
    throw new Error(
      `${layer.product}: no raster source.`
    );
  }

  const map =
    await ensureMap();

  const cfg =
    display
    ?? defaultRasterDisplay(
      layer,
      sourceInfo,
    );

  const range =
    rasterRange(
      sourceInfo,
      cfg.stretch,
      cfg.min,
      cfg.max,
    );

  const colorMap =
    makeColorMap(
      cfg.palette,
      range,
    );

  const source =
    new GeoTIFFSource({
      url: sourceInfo.url,
      crs: state.crs,
    });

  let entity;
  let family;

  if (layer.product === "FAST_DEM") {
    family = "dem";

    entity =
      new ElevationLayer({
        name: layer.product,
        extent: runtimeExtent(),
        source,
        colorMap,
        resolutionFactor: 1 / 4,
      });
  } else {
    family = "raster";

    entity =
      new ColorLayer({
        name: layer.product,
        extent: runtimeExtent(),
        source,
        colorMap,
      });
  }

  entity.opacity =
    Number(cfg.opacity ?? 1);

  await map.addLayer(entity);

  return {
    layer,
    family,
    entity,
    sourceInfo,
    display: cfg,
  };
}


function attributeName(attribute) {
  return (
    attribute?.name
    ?? attribute?.attribute?.name
    ?? null
  );
}


function attributeRange(attribute) {
  const min =
    Number(
      attribute?.min
      ?? attribute?.minimum
    );

  const max =
    Number(
      attribute?.max
      ?? attribute?.maximum
    );

  if (
    Number.isFinite(min)
    && Number.isFinite(max)
    && max > min
  ) {
    return { min, max };
  }

  return null;
}


function isClassification(name) {
  return (
    String(name ?? "")
      .toLowerCase()
      === "classification"
  );
}


async function createPointRecord(layer) {
  const sourceInfo =
    sourceFor(layer, "pointcloud");

  if (!sourceInfo) {
    throw new Error(
      `${layer.product}: no point-cloud source.`
    );
  }

  const sourceName =
    String(
      sourceInfo.filename
      ?? sourceInfo.url
      ?? ""
    );

  if (
    sourceInfo.role === "authoritative"
    && !/\.copc(?:\.laz)?$/i.test(
      sourceName
    )
  ) {
    throw new Error(
      `${layer.product}: raw authoritative LAS/LAZ `
      + "is intentionally not loaded into the browser."
    );
  }

  if (
    !/\.copc(?:\.laz)?$/i.test(
      sourceName
    )
  ) {
    throw new Error(
      `${layer.product}: scalable COPC source required.`
    );
  }

  const source =
    new COPCSource({
      url: sourceInfo.url,
    });

  const cloud =
    new PointCloud({
      source,
    });

  cloud.pointBudget =
    state.pointBudget;

  cloud.pointSize = 0;

  /*
   * FASTGC_VIEWER_V51_LOD
   *
   * Giro3D default subdivision threshold is 1.
   * Lowering the viewer-only SSE threshold requests
   * finer COPC hierarchy nodes earlier during close
   * inspection.
   */
  cloud.subdivisionThreshold =
    0.55;
  cloud.visible = false;

  await state.instance.add(cloud);

  const attributes =
    cloud.getSupportedAttributes?.()
    ?? [];

  const names =
    attributes
      .map(attributeName)
      .filter(Boolean);

  const preferred =
    names.find(isClassification)
    ?? names.find(
      (name) =>
        String(name).toLowerCase() === "z"
    )
    ?? names[0]
    ?? null;

  if (preferred) {
    cloud.setColoringMode(
      "attribute"
    );

    cloud.setActiveAttribute(
      preferred
    );
  }

  return {
    layer,
    family: "pointcloud",
    entity: cloud,
    sourceInfo,
    attributes,
    activeAttribute: preferred,
    palette: "terrain",
  };
}


async function ensureRecord(layer) {
  const existing =
    state.records.get(layer.id);

  if (existing) {
    return existing;
  }

  const capability =
    productCapability(layer);

  let record;

  if (
    capability === "dem"
    || capability === "raster"
  ) {
    record =
      await createRasterRecord(layer);
  } else if (
    capability === "pointcloud"
  ) {
    record =
      await createPointRecord(layer);
  } else {
    throw new Error(
      `${layer.product}: viewer representation unavailable.`
    );
  }

  state.records.set(
    layer.id,
    record,
  );

  return record;
}


function recordByProduct(product) {
  for (
    const record
    of state.records.values()
  ) {
    if (
      record.layer.product
      === product
    ) {
      return record;
    }
  }

  return null;
}


function runtimeLayer(product) {
  return (
    state.runtime.layers.find(
      (layer) =>
        layer.product === product
    )
    ?? null
  );
}


async function setVisible(
  layer,
  visible,
) {
  let record;

  try {
    record =
      await ensureRecord(layer);
  } catch (error) {
    console.error(error);
    setStatus(error.message);
    throw error;
  }

  record.entity.visible =
    Boolean(visible);

  if (visible) {
    state.visibleProducts.add(
      layer.product
    );

    state.activeProduct =
      layer.product;
  } else {
    state.visibleProducts.delete(
      layer.product
    );

    if (
      state.activeProduct
      === layer.product
    ) {
      state.activeProduct =
        [...state.visibleProducts]
          .at(-1)
        ?? null;
    }
  }

  state.instance.notifyChange(
    record.entity
  );

  refreshActiveLayerSelect();
  refreshDisplayPanel();
}


function refreshActiveLayerSelect() {
  const select =
    $("active-layer");

  select.replaceChildren();

  for (
    const product
    of state.visibleProducts
  ) {
    const option =
      document.createElement(
        "option"
      );

    option.value = product;
    option.textContent = product;

    select.append(option);
  }

  if (
    state.activeProduct
    && state.visibleProducts.has(
      state.activeProduct
    )
  ) {
    select.value =
      state.activeProduct;
  }
}


function activeRecord() {
  if (!state.activeProduct) {
    return null;
  }

  return recordByProduct(
    state.activeProduct
  );
}



/* FASTGC_VIEWER_V51_CLASSIFICATION */

const CLASSIFICATION_PALETTES = {
  ASPRS: null,

  "Ground emphasis": {
    0: "#bdbdbd",
    1: "#d9d9d9",
    2: "#7b4b21",
    3: "#a7d676",
    4: "#57ad45",
    5: "#167a2f",
    6: "#d64a3a",
    7: "#ff00ff",
    9: "#438bd3",
  },

  "High contrast": {
    0: "#808080",
    1: "#e0e0e0",
    2: "#ff8c00",
    3: "#b7e075",
    4: "#35b779",
    5: "#006837",
    6: "#e31a1c",
    7: "#984ea3",
    9: "#377eb8",
  },

  "Forest": {
    0: "#a0a0a0",
    1: "#d0d0d0",
    2: "#704214",
    3: "#c5e1a5",
    4: "#7cb342",
    5: "#2e7d32",
    6: "#c62828",
    7: "#ad4bc6",
    9: "#3182bd",
  },
};


function classificationTable(name) {
  const base =
    ASPRS_CLASSIFICATIONS.map(
      item => item.clone()
    );

  const overrides =
    CLASSIFICATION_PALETTES[name];

  if (!overrides) {
    return base;
  }

  for (
    const [codeText, hex]
    of Object.entries(
      overrides
    )
  ) {
    const code =
      Number(codeText);

    if (
      Number.isInteger(code)
      && code >= 0
      && code < base.length
    ) {
      base[code] =
        new Classification(
          new Color(hex)
        );
    }
  }

  return base;
}


function applyClassificationPalette() {
  const record =
    activeRecord();

  if (
    !record
    || record.family !== "pointcloud"
    || !isClassification(
      record.activeAttribute
    )
  ) {
    return;
  }

  const selector =
    $("classification-palette");

  if (!selector) {
    return;
  }

  const table =
    classificationTable(
      selector.value
    );

  /*
   * Giro3D stores classifications per attribute.
   * PointCloud exposes the material to the entity
   * traversal/update path.
   */
  record.entity.traverseMaterials?.(
    material => {
      if (!material) {
        return;
      }

      if (
        material
          .attributesState
          ?.classifications
      ) {
        for (
          const slot
          of material
            .attributesState
            .classifications
        ) {
          slot.classifications =
            table;
        }
      }

      material.needsUpdate =
        true;
    }
  );

  state.instance.notifyChange(
    record.entity
  );

  setStatus(
    `${record.layer.product}: Classification / `
    + selector.value
  );
}


function refreshClassificationPalette() {
  const row =
    $("classification-palette-row");

  if (!row) {
    return;
  }

  const record =
    activeRecord();

  const show =
    record
    && record.family === "pointcloud"
    && isClassification(
      record.activeAttribute
    );

  row.hidden =
    !show;
}

function refreshPointPanel(record) {
  $("point-controls").hidden = false;
  $("raster-controls").hidden = true;

  const select =
    $("point-attribute");

  select.replaceChildren();

  /*
   * Read the loaded entity again rather than
   * relying on stale UI state.
   */
  const attributes =
    record.entity
      .getSupportedAttributes?.()
    ?? record.attributes
    ?? [];

  record.attributes =
    attributes;

  for (
    const attribute
    of attributes
  ) {
    const name =
      attributeName(attribute);

    if (!name) {
      continue;
    }

    const option =
      document.createElement(
        "option"
      );

    option.value = name;
    option.textContent = name;

    select.append(option);
  }

  if (
    record.activeAttribute
    && [...select.options].some(
      (option) =>
        option.value
        === record.activeAttribute
    )
  ) {
    select.value =
      record.activeAttribute;
  } else if (
    select.options.length
  ) {
    record.activeAttribute =
      select.options[0].value;

    select.value =
      record.activeAttribute;
  }

  const selected =
    attributes.find(
      (attribute) =>
        attributeName(attribute)
        === select.value
    );

  const range =
    attributeRange(selected);

  $("attribute-info").textContent =
    range
      ? `${range.min} to ${range.max}`
      : "Categorical or source-defined attribute.";

  const classification =
    isClassification(
      select.value
    );

  $("classification-note").hidden =
    !classification;

  $("scalar-controls").hidden =
    classification;

  $("point-palette").disabled =
    classification;

  refreshClassificationPalette();

  const size =
    Number(
      record.entity.pointSize ?? 0
    );

  $("point-size").value =
    String(size);

  $("point-size-value").textContent =
    size === 0
      ? "Auto"
      : `${size.toFixed(2)} px`;
}


function refreshRasterPanel(record) {
  $("point-controls").hidden = true;
  $("raster-controls").hidden = false;

  const cfg =
    record.display;

  $("raster-palette").value =
    cfg.palette;

  $("raster-stretch").value =
    cfg.stretch;

  $("raster-min").value =
    cfg.min;

  $("raster-max").value =
    cfg.max;

  $("raster-opacity").value =
    cfg.opacity;

  $("raster-opacity-value")
    .textContent =
      `${Math.round(
        cfg.opacity * 100
      )}%`;

  $("raster-custom").hidden =
    cfg.stretch !== "custom";

  const stats =
    displayStats(
      record.sourceInfo
    );

  if (stats) {
    $("raster-info").textContent =
      `2–98%: ${Number(stats.p02).toFixed(2)}–`
      + `${Number(stats.p98).toFixed(2)} | `
      + `full: ${Number(stats.min).toFixed(2)}–`
      + `${Number(stats.max).toFixed(2)}`;
  } else {
    $("raster-info").textContent =
      "Raster statistics unavailable.";
  }
}


function refreshDisplayPanel() {
  const record =
    activeRecord();

  if (!record) {
    $("point-controls").hidden = true;
    $("raster-controls").hidden = true;
    return;
  }

  if (
    record.family
    === "pointcloud"
  ) {
    refreshPointPanel(record);
  } else {
    refreshRasterPanel(record);
  }
}


function applyPointAttribute() {
  const record =
    activeRecord();

  if (
    !record
    || record.family !== "pointcloud"
  ) {
    return;
  }

  const name =
    $("point-attribute").value;

  if (!name) {
    return;
  }

  record.entity.setColoringMode(
    "attribute"
  );

  record.entity.setActiveAttribute(
    name
  );

  record.activeAttribute = name;

  if (!isClassification(name)) {
    applyPointPalette();
  }

  state.instance.notifyChange(
    record.entity
  );

  refreshPointPanel(record);
}


function applyPointPalette() {
  const record =
    activeRecord();

  if (
    !record
    || record.family !== "pointcloud"
  ) {
    return;
  }

  const name =
    record.activeAttribute;

  if (
    !name
    || isClassification(name)
  ) {
    return;
  }

  const attribute =
    record.attributes.find(
      (item) =>
        attributeName(item)
        === name
    );

  const range =
    attributeRange(attribute);

  if (!range) {
    setStatus(
      `${name}: scalar range unavailable; `
      + "using source-defined rendering."
    );
    return;
  }

  const palette =
    $("point-palette").value;

  const colorMap =
    makeColorMap(
      palette,
      range,
    );

  record.entity
    .setAttributeColorMap(
      name,
      colorMap,
    );

  record.entity.setColoringMode(
    "attribute"
  );

  record.entity.setActiveAttribute(
    name
  );

  record.palette = palette;

  state.instance.notifyChange(
    record.entity
  );

  setStatus(
    `${record.layer.product}: `
    + `${name} / ${palette}`
  );
}


function applyPointSize() {
  const record =
    activeRecord();

  if (
    !record
    || record.family !== "pointcloud"
  ) {
    return;
  }

  const value =
    Number(
      $("point-size").value
    );

  record.entity.pointSize =
    value;

  $("point-size-value").textContent =
    value === 0
      ? "Auto"
      : `${value.toFixed(2)} px`;

  state.instance.notifyChange(
    record.entity
  );
}


async function rebuildRaster() {
  const old =
    activeRecord();

  if (
    !old
    || old.family === "pointcloud"
  ) {
    return;
  }

  const palette =
    $("raster-palette").value;

  const stretch =
    $("raster-stretch").value;

  const min =
    Number(
      $("raster-min").value
    );

  const max =
    Number(
      $("raster-max").value
    );

  const opacity =
    Number(
      $("raster-opacity").value
    );

  const cfg = {
    palette,
    stretch,
    min,
    max,
    opacity,
  };

  const range =
    rasterRange(
      old.sourceInfo,
      stretch,
      min,
      max,
    );

  if (!range) {
    setStatus(
      `${old.layer.product}: invalid raster range.`
    );
    return;
  }

  cfg.min = range.min;
  cfg.max = range.max;

  const wasVisible =
    old.entity.visible !== false;

  /*
   * CLEAN V5:
   * do not mutate an already-composed raster ColorMap.
   * Remove and recreate only this visualization layer.
   */
  old.entity.visible = false;

  if (
    typeof state.map.removeLayer
    === "function"
  ) {
    await state.map.removeLayer(
      old.entity
    );
  } else if (
    typeof old.entity.dispose
    === "function"
  ) {
    old.entity.dispose();
  }

  state.records.delete(
    old.layer.id
  );

  const replacement =
    await createRasterRecord(
      old.layer,
      cfg,
    );

  replacement.entity.visible =
    wasVisible;

  state.records.set(
    old.layer.id,
    replacement,
  );

  state.instance.notifyChange(
    replacement.entity
  );

  refreshDisplayPanel();

  setStatus(
    `${old.layer.product}: `
    + `${palette}, `
    + `${range.min.toFixed(2)}–`
    + `${range.max.toFixed(2)}`
  );
}


function applyRasterOpacity() {
  const record =
    activeRecord();

  if (
    !record
    || record.family === "pointcloud"
  ) {
    return;
  }

  const value =
    Number(
      $("raster-opacity").value
    );

  record.display.opacity = value;
  record.entity.opacity = value;

  $("raster-opacity-value")
    .textContent =
      `${Math.round(
        value * 100
      )}%`;

  state.instance.notifyChange(
    record.entity
  );
}


function applyPointBudget() {
  state.pointBudget =
    Number(
      $("point-budget").value
    );

  $("point-budget-value")
    .textContent =
      state.pointBudget
        .toLocaleString();

  for (
    const record
    of state.records.values()
  ) {
    if (
      record.family
      === "pointcloud"
    ) {
      record.entity.pointBudget =
        state.pointBudget;

      state.instance.notifyChange(
        record.entity
      );
    }
  }
}


function applyEdl() {
  state.edl =
    $("edl").checked;

  state.instance.renderingOptions
    .enableEDL =
      state.edl;

  state.instance.notifyChange();
}


function cameraGeometry() {
  const box =
    datasetBox();

  const center =
    box.getCenter(
      new Vector3()
    );

  const size =
    box.getSize(
      new Vector3()
    );

  const radius =
    Math.max(
      size.length() / 2,
      1,
    );

  return {
    box,
    center,
    size,
    radius,
  };
}



/* FASTGC_VIEWER_V51_CAMERA */

function fitDirection(
  direction,
  requestedUp
) {
  const box = datasetBox();

  if (!box) {
    fitDataset();
    return;
  }

  const camera =
    state.instance.view.camera;

  const controls =
    state.controls;

  const center =
    box.getCenter(
      new Vector3()
    );

  const dir =
    direction
      .clone()
      .normalize();

  let up =
    requestedUp
      .clone()
      .normalize();

  let right =
    new Vector3()
      .crossVectors(
        dir,
        up
      );

  if (right.lengthSq() < 1e-10) {
    up =
      Math.abs(dir.z) > 0.9
        ? new Vector3(0, 1, 0)
        : new Vector3(0, 0, 1);

    right.crossVectors(
      dir,
      up
    );
  }

  right.normalize();

  const trueUp =
    new Vector3()
      .crossVectors(
        right,
        dir
      )
      .normalize();

  const corners = [];

  for (
    const x of [
      box.min.x,
      box.max.x
    ]
  ) {
    for (
      const y of [
        box.min.y,
        box.max.y
      ]
    ) {
      for (
        const z of [
          box.min.z,
          box.max.z
        ]
      ) {
        corners.push(
          new Vector3(
            x,
            y,
            z
          ).sub(center)
        );
      }
    }
  }

  let halfWidth = 0;
  let halfHeight = 0;
  let halfDepth = 0;

  for (const p of corners) {
    halfWidth =
      Math.max(
        halfWidth,
        Math.abs(
          p.dot(right)
        )
      );

    halfHeight =
      Math.max(
        halfHeight,
        Math.abs(
          p.dot(trueUp)
        )
      );

    halfDepth =
      Math.max(
        halfDepth,
        Math.abs(
          p.dot(dir)
        )
      );
  }

  const dom =
    state.instance.domElement;

  const width =
    Math.max(
      dom?.clientWidth ?? 1,
      1
    );

  const height =
    Math.max(
      dom?.clientHeight ?? 1,
      1
    );

  const aspect =
    width / height;

  camera.aspect = aspect;

  const vFov =
    camera.fov
    * Math.PI
    / 180;

  const hFov =
    2 * Math.atan(
      Math.tan(vFov / 2)
      * aspect
    );

  const distV =
    halfHeight
    / Math.max(
      Math.tan(vFov / 2),
      1e-6
    );

  const distH =
    halfWidth
    / Math.max(
      Math.tan(hFov / 2),
      1e-6
    );

  const distance =
    (
      Math.max(
        distV,
        distH
      )
      + halfDepth
    )
    * 1.10;

  camera.position.copy(
    center
      .clone()
      .add(
        dir
          .clone()
          .multiplyScalar(
            distance
          )
      )
  );

  camera.up.copy(
    trueUp
  );

  camera.lookAt(
    center
  );

  controls.target.copy(
    center
  );

  /*
   * Important for preset views:
   * discard damping movement inherited
   * from the previous camera orientation.
   */
  const damping =
    controls.enableDamping;

  controls.enableDamping =
    false;

  controls.update();

  controls.enableDamping =
    damping;

  /*
   * Prevent close-range clipping.
   */
  const radius =
    Math.max(
      box.getSize(
        new Vector3()
      ).length() / 2,
      1
    );

  camera.near =
    Math.max(
      0.01,
      radius * 0.0001
    );

  camera.far =
    Math.max(
      distance
        + radius * 30,
      10000
    );

  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);

  state.instance.notifyChange(
    camera
  );
}

function fitDataset() {
  const g =
    cameraGeometry();

  const camera =
    state.instance.view.camera;

  const distance =
    Math.max(
      g.radius * 2.2,
      10,
    );

  camera.up.set(0, 0, 1);

  camera.position.set(
    g.center.x + distance,
    g.center.y - distance,
    g.center.z + distance,
  );

  camera.lookAt(
    g.center
  );

  state.controls.target.copy(
    g.center
  );

  state.controls.update();

  camera.updateMatrixWorld(true);

  if (
    typeof camera
      .updateProjectionMatrix
    === "function"
  ) {
    camera.updateProjectionMatrix();
  }

  state.instance.notifyChange(
    camera
  );
}


function setView(name) {
  const key =
    String(name)
      .toLowerCase();

  if (key === "fit") {
    fitDataset();
    setStatus(
      "FIT view"
    );
    return;
  }

  const views = {
    top: {
      direction:
        new Vector3(
          0, 0, 1
        ),
      up:
        new Vector3(
          0, 1, 0
        ),
    },

    bottom: {
      direction:
        new Vector3(
          0, 0, -1
        ),
      up:
        new Vector3(
          0, 1, 0
        ),
    },

    front: {
      direction:
        new Vector3(
          0, -1, 0
        ),
      up:
        new Vector3(
          0, 0, 1
        ),
    },

    back: {
      direction:
        new Vector3(
          0, 1, 0
        ),
      up:
        new Vector3(
          0, 0, 1
        ),
    },

    left: {
      direction:
        new Vector3(
          -1, 0, 0
        ),
      up:
        new Vector3(
          0, 0, 1
        ),
    },

    right: {
      direction:
        new Vector3(
          1, 0, 0
        ),
      up:
        new Vector3(
          0, 0, 1
        ),
    },

    isometric: {
      direction:
        new Vector3(
          1, -1, 1
        ),
      up:
        new Vector3(
          0, 0, 1
        ),
    },
  };

  const view =
    views[key];

  if (!view) {
    return;
  }

  fitDirection(
    view.direction,
    view.up
  );

  setStatus(
    `${key.toUpperCase()} view — centered + fitted`
  );
}


async function inspectScene(event) {
  event.preventDefault();

  try {
    const hits =
      await state.instance
        .pickObjectsAt(
          event,
          {
            radius: 3,
            limit: 8,
          },
        );

    if (!hits?.length) {
      $("inspect").textContent =
        "No object at cursor.";
      return;
    }

    const clean =
      hits.map(
        (hit, index) => ({
          index: index + 1,
          entity:
            hit.entity?.name
            ?? hit.object?.name
            ?? hit.entity?.type
            ?? "object",
          point:
            hit.point
              ? {
                  x: hit.point.x,
                  y: hit.point.y,
                  z: hit.point.z,
                }
              : undefined,
          distance:
            hit.distance,
        })
      );

    $("inspect").textContent =
      JSON.stringify(
        clean,
        null,
        2,
      );
  } catch (error) {
    console.error(error);

    $("inspect").textContent =
      `Inspect error: ${error.message}`;
  }
}


function populateProducts() {
  const host =
    $("products");

  host.replaceChildren();

  for (
    const layer
    of state.runtime.layers
  ) {
    const capability =
      productCapability(layer);

    const label =
      document.createElement(
        "label"
      );

    const input =
      document.createElement(
        "input"
      );

    input.type = "checkbox";
    input.disabled = !capability;

    const text =
      document.createElement(
        "span"
      );

    text.textContent =
      capability
        ? layer.product
        : `${layer.product} (unavailable)`;

    label.append(
      input,
      text,
    );

    host.append(label);

    if (!capability) {
      continue;
    }

    input.addEventListener(
      "change",
      async () => {
        input.disabled = true;

        try {
          await setVisible(
            layer,
            input.checked,
          );
        } catch {
          input.checked = false;
        } finally {
          input.disabled = false;
        }
      },
    );
  }
}


function bindUi() {
  $("active-layer")
    .addEventListener(
      "change",
      () => {
        state.activeProduct =
          $("active-layer").value;

        refreshDisplayPanel();
      },
    );

  $("point-attribute")
    .addEventListener(
      "change",
      applyPointAttribute,
    );

  $("classification-palette")
    .addEventListener(
      "change",
      applyClassificationPalette,
    );

  $("point-palette")
    .addEventListener(
      "change",
      applyPointPalette,
    );

  $("point-size")
    .addEventListener(
      "input",
      applyPointSize,
    );

  $("point-budget")
    .addEventListener(
      "input",
      applyPointBudget,
    );

  $("edl")
    .addEventListener(
      "change",
      applyEdl,
    );

  $("raster-palette")
    .addEventListener(
      "change",
      rebuildRaster,
    );

  $("raster-stretch")
    .addEventListener(
      "change",
      async () => {
        $("raster-custom").hidden =
          $("raster-stretch").value
          !== "custom";

        if (
          $("raster-stretch").value
          !== "custom"
        ) {
          await rebuildRaster();
        }
      },
    );

  $("raster-min")
    .addEventListener(
      "change",
      rebuildRaster,
    );

  $("raster-max")
    .addEventListener(
      "change",
      rebuildRaster,
    );

  $("raster-opacity")
    .addEventListener(
      "input",
      applyRasterOpacity,
    );

  for (
    const button
    of document.querySelectorAll(
      "[data-view]"
    )
  ) {
    button.addEventListener(
      "click",
      () => {
        setView(
          button.dataset.view
        );
      },
    );
  }

  state.instance.domElement
    .addEventListener(
      "contextmenu",
      inspectScene,
    );
}


async function enableInitialLayer() {
  const preferred =
    state.runtime.layers.find(
      (layer) =>
        layer.product === "FAST_GC"
        && productCapability(layer)
    )
    ?? state.runtime.layers.find(
      (layer) =>
        productCapability(layer)
    );

  if (!preferred) {
    return;
  }

  const labels =
    [...$("products")
      .querySelectorAll("label")];

  for (const label of labels) {
    const span =
      label.querySelector("span");

    const input =
      label.querySelector("input");

    if (
      span?.textContent
      === preferred.product
    ) {
      input.checked = true;

      await setVisible(
        preferred,
        true,
      );

      break;
    }
  }
}


async function initialize() {
  try {
    setStatus(
      "Loading FAST-GC runtime..."
    );

    const response =
      await fetch(
        "/runtime.json",
        {
          cache: "no-store",
        },
      );

    if (!response.ok) {
      throw new Error(
        `runtime.json HTTP ${response.status}`
      );
    }

    const runtime =
      await response.json();

    if (
      runtime.schema_version !== 2
    ) {
      throw new Error(
        "FAST-GC runtime schema V2 required."
      );
    }

    if (
      !runtime.crs?.id
      || !runtime.crs?.wkt
    ) {
      throw new Error(
        "Runtime CRS id/WKT missing."
      );
    }

    if (
      !Array.isArray(runtime.layers)
    ) {
      throw new Error(
        "Runtime layers missing."
      );
    }

    state.runtime = runtime;

    state.crs =
      CoordinateSystem.register(
        runtime.crs.id,
        runtime.crs.wkt,
      );

    const instance =
      new Instance({
        target: "giro3d-view",
        crs: state.crs,
        backgroundColor: "#ffffff",
      });

    state.instance = instance;

    instance.renderingOptions
      .enableEDL =
        state.edl;

    const controls =
      new OrbitControls(
        instance.view.camera,
        instance.domElement,
      );

    /*
     * CloudCompare-like mouse contract.
     */
    controls.mouseButtons.LEFT =
      MOUSE.ROTATE;

    controls.mouseButtons.MIDDLE =
      MOUSE.PAN;

    /*
     * Right mouse belongs to FAST-GC inspection.
     */
    controls.mouseButtons.RIGHT =
      null;

    controls.enableDamping = true;

    // FASTGC_VIEWER_V51_ACTUAL
    // Free 3-D point-cloud navigation.
    controls.enableRotate = true;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.screenSpacePanning = true;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;

    controls.dampingFactor = 0.12;

    controls.addEventListener(
      "change",
      () => {
        instance.notifyChange(
          instance.view.camera
        );
      },
    );

    state.controls = controls;

    if (
      typeof instance.view
        .setControls === "function"
    ) {
      instance.view.setControls(
        controls
      );
    }

    $("dataset").textContent =
      runtime.dataset
      ?? "FAST-GC dataset";

    const pointCount =
      runtime.points
      ?? runtime.spatial?.points
      ?? runtime.point_count
      ?? null;

    $("metadata").textContent =
      [
        runtime.sensor
          ? `Sensor: ${runtime.sensor}`
          : null,

        runtime.crs?.id
          ? `CRS: ${runtime.crs.id}`
          : null,

        Number.isFinite(
          Number(pointCount)
        )
          ? `Points: ${Number(pointCount).toLocaleString()}`
          : null,
      ]
        .filter(Boolean)
        .join(" · ");

    populateProducts();
    bindUi();

    applyPointBudget();
    applyEdl();

    await enableInitialLayer();

    fitDataset();

    setStatus(
      "FAST-GC Viewer V5 ready."
    );

  } catch (error) {
    console.error(error);

    setStatus(
      `Initialization failed:\n${error.stack ?? error}`
    );
  }
}


initialize();
