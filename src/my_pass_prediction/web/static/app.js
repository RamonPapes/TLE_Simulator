"use strict";

/* Interface do simulador de passagens.
 *
 * O servidor entrega a trajetória já amostrada; este arquivo só desenha
 * e anima. Nenhuma conta de órbita acontece aqui -- se algum número
 * precisar ser calculado, ele pertence ao Python.
 */

const map = L.map("map", { worldCopyJump: true }).setView([-12.98, -38.5], 4);

/* Camadas de fundo, todas sem credencial.
 *
 * Fica como escolha do usuário porque política de tile muda: a CARTO,
 * que estava aqui antes, passou a exigir key e carimba "API KEY
 * REQUIRED" *dentro* do PNG -- responde HTTP 200, então nenhuma
 * checagem de status pega isso. Com três provedores, um mudar de ideia
 * não deixa o mapa inutilizável.
 *
 * Atenção à ordem dos eixos: o ArcGIS serve {z}/{y}/{x}, ao contrário
 * do {z}/{x}/{y} que quase todo mundo usa.
 */
const ESRI_ATTR =
  'Tiles &copy; <a href="https://www.esri.com/">Esri</a>';

const esri = (service) =>
  L.tileLayer(
    `https://server.arcgisonline.com/ArcGIS/rest/services/${service}/MapServer/tile/{z}/{y}/{x}`,
    { attribution: ESRI_ATTR, maxZoom: 19 }
  );

const basemaps = {
  // Base sem rótulos mais uma camada só de nomes por cima.
  Escuro: L.layerGroup([
    esri("Canvas/World_Dark_Gray_Base"),
    esri("Canvas/World_Dark_Gray_Reference"),
  ]),
  Satélite: esri("World_Imagery"),
  Mapa: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
  }),
};

basemaps.Escuro.addTo(map);
L.control.layers(basemaps, null, { position: "topright" }).addTo(map);

/* Ícones servidos do próprio `static/`, e não de um CDN: a página
 * continua funcionando offline e não depende da política de terceiros.
 *
 * As âncoras diferem porque o que marca a posição é diferente em cada
 * um. O satélite é um ponto projetado do céu, então ancora pelo centro
 * do desenho -- que está centrado no viewBox de propósito. A estação é
 * uma antena apoiada no chão: ancora pela base do pedestal, a 97.5% da
 * altura, senão o prato é que ficaria sobre a coordenada.
 */
const SAT_SIZE = 38;
const STATION_SIZE = 34;

const satIcon = L.icon({
  iconUrl: "satellite.svg",
  iconSize: [SAT_SIZE, SAT_SIZE],
  iconAnchor: [SAT_SIZE / 2, SAT_SIZE / 2],
  popupAnchor: [0, -SAT_SIZE / 2 - 2],
  className: "sat-icon",
});

const stationIcon = L.icon({
  iconUrl: "ground-station.svg",
  iconSize: [STATION_SIZE, STATION_SIZE],
  iconAnchor: [STATION_SIZE / 2, STATION_SIZE * 0.975],
  popupAnchor: [0, -STATION_SIZE + 4],
  className: "station-icon",
});

const layers = L.layerGroup().addTo(map);

/** Estado da simulação atual. `null` enquanto nada foi calculado. */
let sim = null;

/* ------------------------------------------------------------------ *
 * Helpers de geometria de tela
 * ------------------------------------------------------------------ */

/** Diferença de longitude no intervalo (-180, 180]. */
function lonDelta(a, b) {
  return ((b - a + 540) % 360) - 180;
}

/**
 * Quebra a ground track nos cruzamentos do antimeridiano.
 *
 * Sem isso o Leaflet liga o ponto em +179 ao ponto em -179 com um traço
 * atravessando o mapa inteiro -- a órbita aparenta voltar no tempo. A
 * emenda estende cada segmento até a borda (+/-180) para que a linha
 * saia de um lado exatamente na latitude em que entra no outro.
 */
function splitAtAntimeridian(points) {
  const segments = [];
  let current = [points[0]];

  for (let i = 1; i < points.length; i += 1) {
    const [prevLat, prevLon] = points[i - 1];
    const [lat, lon] = points[i];
    const delta = lonDelta(prevLon, lon);

    if (Math.abs(lon - prevLon) > 180) {
      // Latitude no instante da travessia, por interpolação linear.
      const edge = delta > 0 ? 180 : -180;
      const fraction = Math.abs(lonDelta(prevLon, edge) / delta);
      const edgeLat = prevLat + (lat - prevLat) * fraction;

      current.push([edgeLat, edge]);
      segments.push(current);
      current = [[edgeLat, -edge], [lat, lon]];
    } else {
      current.push([lat, lon]);
    }
  }
  segments.push(current);
  return segments.filter((segment) => segment.length > 1);
}

/** Interpola a amostra no instante `t` (segundos desde a época). */
function sampleAt(samples, t) {
  if (t <= samples[0].t) return samples[0];
  if (t >= samples[samples.length - 1].t) return samples[samples.length - 1];

  let lo = 0;
  let hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) lo = mid;
    else hi = mid;
  }

  const a = samples[lo];
  const b = samples[hi];
  const span = b.t - a.t;
  const f = span === 0 ? 0 : (t - a.t) / span;
  const mix = (x, y) => x + (y - x) * f;

  return {
    t,
    // Pela diferença, e não pelos valores brutos: entre +179 e -179 a
    // média direta daria 0, jogando o satélite no meridiano oposto.
    lat: mix(a.lat, b.lat),
    lon: a.lon + lonDelta(a.lon, b.lon) * f,
    alt_km: mix(a.alt_km, b.alt_km),
    el: mix(a.el, b.el),
    az: mix(a.az, b.az),
    range_km: mix(a.range_km, b.range_km),
  };
}

const pad = (n) => String(Math.floor(Math.abs(n))).padStart(2, "0");

/** Segundos desde a época como "+mm:ss" / "-mm:ss". */
function signedClock(seconds) {
  const sign = seconds < 0 ? "-" : "+";
  return `${sign}${pad(seconds / 60)}:${pad(seconds % 60)}`;
}

function formatDuration(seconds) {
  return `${Math.floor(seconds / 60)}m ${pad(seconds % 60)}s`;
}

function timeOf(iso) {
  return new Date(iso).toISOString().slice(11, 19);
}

/* ------------------------------------------------------------------ *
 * Desenho
 * ------------------------------------------------------------------ */

function render(data) {
  layers.clearLayers();

  const { samples, aos, los } = data.track;
  const points = samples.map((s) => [s.lat, s.lon]);

  // Trilha completa, incluindo o trecho abaixo do horizonte.
  splitAtAntimeridian(points).forEach((segment) => {
    L.polyline(segment, {
      color: "#ff5a5f",
      weight: 2,
      opacity: 0.55,
      dashArray: "5 6",
    }).addTo(layers);
  });

  // Trecho visível da estação, em destaque: é a passagem propriamente
  // dita, entre AOS e LOS.
  const visible = samples.filter((s) => s.el > 0).map((s) => [s.lat, s.lon]);
  if (visible.length > 1) {
    splitAtAntimeridian(visible).forEach((segment) => {
      L.polyline(segment, { color: "#4dff9d", weight: 3.5, opacity: 0.95 }).addTo(
        layers
      );
    });
  }

  const station = L.marker([data.station.lat, data.station.lon], {
    icon: stationIcon,
    zIndexOffset: 100,
  }).addTo(layers);
  station.bindPopup(
    `<b>${data.station.name}</b><br>${data.station.lat.toFixed(4)}, ${data.station.lon.toFixed(4)}`
  );

  const satellite = L.marker(points[0], {
    icon: satIcon,
    zIndexOffset: 200,
  }).addTo(layers);

  const bounds = L.latLngBounds(points).extend([
    data.station.lat,
    data.station.lon,
  ]);
  map.fitBounds(bounds, { padding: [70, 70] });

  sim = {
    data,
    samples,
    satellite,
    t0: samples[0].t,
    t1: samples[samples.length - 1].t,
    t: samples[0].t,
    playing: false,
    lastFrame: null,
  };

  slider.min = String(samples[0].t);
  slider.max = String(samples[samples.length - 1].t);
  slider.step = "0.5";
  playback.hidden = false;

  renderSummary(data);
  seek(sim.t0);

  if (aos && los) {
    satellite.bindPopup(
      `<b>${data.tle.name}</b><br>AOS ${timeOf(aos.t)} · LOS ${timeOf(los.t)} UTC`
    );
  }
}

function renderSummary(data) {
  const { orbit, track } = data;
  const rows = [
    ["Elev. máxima", `${(track.tca ? track.tca.el : track.max_elevation_deg).toFixed(2)}°`],
    ["Duração", track.pass_duration_s === null ? "—" : formatDuration(track.pass_duration_s)],
    ["AOS", track.aos ? `${timeOf(track.aos.t)} UTC` : "—"],
    ["TCA", track.tca ? `${timeOf(track.tca.t)} UTC` : "—"],
    ["LOS", track.los ? `${timeOf(track.los.t)} UTC` : "—"],
    ["Inclinação", `${orbit.inclination_deg.toFixed(2)}°`],
    ["Período", `${orbit.period_min.toFixed(2)} min`],
    ["Amostras", String(track.samples.length)],
  ];

  summary.replaceChildren(
    ...rows.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    })
  );

  tleBox.textContent = `${data.tle.name}\n${data.tle.line1}\n${data.tle.line2}`;
  results.hidden = false;
}

/* ------------------------------------------------------------------ *
 * Reprodução
 * ------------------------------------------------------------------ */

function seek(t) {
  if (!sim) return;
  sim.t = Math.min(sim.t1, Math.max(sim.t0, t));

  const s = sampleAt(sim.samples, sim.t);
  sim.satellite.setLatLng([s.lat, s.lon]);
  slider.value = String(sim.t);

  const horizon = s.el > 0 ? "" : " (abaixo do horizonte)";
  readout.textContent =
    `${signedClock(sim.t)}  el ${s.el.toFixed(1)}°  ${Math.round(s.range_km)} km${horizon}`;
}

function frame(now) {
  if (!sim || !sim.playing) return;

  const elapsed = sim.lastFrame === null ? 0 : (now - sim.lastFrame) / 1000;
  sim.lastFrame = now;
  seek(sim.t + elapsed * Number(speed.value));

  if (sim.t >= sim.t1) {
    pause();
    return;
  }
  requestAnimationFrame(frame);
}

function play() {
  if (!sim || sim.playing) return;
  if (sim.t >= sim.t1) sim.t = sim.t0;
  sim.playing = true;
  sim.lastFrame = null;
  playButton.textContent = "⏸";
  playButton.title = "Pausar";
  requestAnimationFrame(frame);
}

function pause() {
  if (!sim) return;
  sim.playing = false;
  playButton.textContent = "▶";
  playButton.title = "Reproduzir";
}

/* ------------------------------------------------------------------ *
 * Ligações com o DOM
 * ------------------------------------------------------------------ */

const form = document.getElementById("params");
const errorBox = document.getElementById("error");
const results = document.getElementById("results");
const summary = document.getElementById("summary");
const tleBox = document.getElementById("tle");
const playback = document.getElementById("playback");
const playButton = document.getElementById("play");
const slider = document.getElementById("slider");
const readout = document.getElementById("readout");
const speed = document.getElementById("speed");
const simulateButton = document.getElementById("simulate");
const panel = document.getElementById("panel");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

async function simulate() {
  pause();
  errorBox.hidden = true;
  simulateButton.disabled = true;
  simulateButton.textContent = "Calculando…";

  const payload = Object.fromEntries(new FormData(form).entries());

  try {
    const response = await fetch("api/pass", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      showError(body.error || `erro ${response.status}`);
      return;
    }
    render(body);
  } catch (err) {
    showError(`falha ao falar com o servidor: ${err.message}`);
  } finally {
    simulateButton.disabled = false;
    simulateButton.textContent = "Simular passagem";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  simulate();
});

playButton.addEventListener("click", () => (sim && sim.playing ? pause() : play()));

slider.addEventListener("input", (event) => {
  pause();
  seek(Number(event.target.value));
});

map.on("click", (event) => {
  form.lat.value = event.latlng.lat.toFixed(4);
  form.lon.value = L.Util.wrapNum(event.latlng.lng, [-180, 180], true).toFixed(4);
});

document.getElementById("copy-tle").addEventListener("click", async (event) => {
  try {
    await navigator.clipboard.writeText(tleBox.textContent);
    event.target.textContent = "Copiado";
    setTimeout(() => (event.target.textContent = "Copiar"), 1400);
  } catch {
    showError("o navegador bloqueou o acesso à área de transferência");
  }
});

document.getElementById("toggle-panel").addEventListener("click", (event) => {
  const collapsed = panel.classList.toggle("collapsed");
  event.target.textContent = collapsed ? "+" : "−";
});

// Espaço alterna play/pause, desde que o foco não esteja num campo.
document.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.target.matches("input, select, button")) return;
  event.preventDefault();
  sim && sim.playing ? pause() : play();
});

simulate();
