async function loadForecastData() {
  try {
    const response = await fetch(`/api/forecast`);
    const data = await response.json();
    applyChartUpdate(data);
  } catch (error) {
    console.error("Помилка отримання прогнозу:", error);
  }
}

function applyChartUpdate(currentData) {
  if (!currentData) return;

  console.log("forecast", currentData.forecast);
  console.log("forecast length", currentData.forecast?.length);

  // Окреме вирівнювання для SMA/EMA
  const alignIndicators = (series) => {
    if (!series) return [];
    const offset = currentData.real.length - series.length;
    return series.map((y, i) => ({ x: offset + i, y }));
  };

  const indicatorsToDraw = {};

  if (document.getElementById("toggle-sma").checked) {
    indicatorsToDraw.SMA_10 = alignIndicators(currentData.indicators?.SMA_10);
    indicatorsToDraw.SMA_30 = alignIndicators(currentData.indicators?.SMA_30);
  }

  if (document.getElementById("toggle-ema").checked) {
    indicatorsToDraw.EMA_10 = alignIndicators(currentData.indicators?.EMA_10);
    indicatorsToDraw.EMA_30 = alignIndicators(currentData.indicators?.EMA_30);
  }

  if (document.getElementById("toggle-rsi").checked) {
  document.getElementById("rsi-container").style.display = "flex";
  drawRSIChart(currentData.indicators?.RSI);
} else {
  document.getElementById("rsi-container").style.display = "none";
}

if (document.getElementById("toggle-macd").checked) {
  document.getElementById("macd-container").style.display = "flex";
  drawMACDChart(currentData.indicators?.MACD, currentData.indicators?.Signal_Line);
} else {
  document.getElementById("macd-container").style.display = "none";
}


  updateChart(currentData.real, currentData.forecast, "binance", indicatorsToDraw);
}

// --- RSI ---
function drawRSIChart(rsiData) {
  const svg = d3.select("#rsi-chart").html("").append("svg").attr("width", 800).attr("height", 180);
  const margin = { top: 30, right: 20, bottom: 30, left: 50 };
  const width = 800 - margin.left - margin.right;
  const height = 180 - margin.top - margin.bottom;

  const xScale = d3.scaleLinear().domain([0, rsiData.length - 1]).range([0, width]);
  const yScale = d3.scaleLinear().domain([0, 100]).range([height, 0]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  // Назва графіка
  svg.append("text")
    .attr("x", margin.left)
    .attr("y", 20)
    .text("RSI (Relative Strength Index)")
    .attr("font-size", "14px")
    .attr("font-weight", "bold");

  // Осі
  g.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6));
  g.append("g").call(d3.axisLeft(yScale).ticks(5));

  // Сітка
  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(yScale).ticks(5).tickSize(-width).tickFormat(""));

  // Лінія RSI
  const line = d3.line().x((d, i) => xScale(i)).y(d => yScale(d));
  g.append("path")
    .datum(rsiData)
    .attr("fill", "none")
    .attr("stroke", "#DC143C")
    .attr("stroke-width", 1.5)
    .attr("d", line);

  // Границі 30 / 70
  [30, 70].forEach(y => {
    g.append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", yScale(y))
      .attr("y2", yScale(y))
      .attr("stroke", "#aaa")
      .attr("stroke-dasharray", "4,2");
  });

  // Легенда
  svg.append("text")
    .attr("x", width + margin.left - 100)
    .attr("y", 20)
    .attr("text-anchor", "start")
    .attr("fill", "#DC143C")
    .style("font-size", "12px")
    .text("RSI");

  // Курсор
  const focusLine = g.append("line").attr("stroke", "#999").attr("stroke-dasharray", "3,3").style("display", "none");

  svg.on("mousemove", function(event) {
    const [x] = d3.pointer(event);
    const xVal = Math.round(xScale.invert(x - margin.left));
    if (xVal >= 0 && xVal < rsiData.length) {
      focusLine
        .style("display", null)
        .attr("x1", xScale(xVal))
        .attr("x2", xScale(xVal))
        .attr("y1", 0)
        .attr("y2", height);
    }
  }).on("mouseleave", () => {
    focusLine.style("display", "none");
  });
}


// --- MACD ---
function drawMACDChart(macd, signal) {
  const svg = d3.select("#macd-chart").html("").append("svg").attr("width", 800).attr("height", 180);
  const margin = { top: 30, right: 20, bottom: 30, left: 50 };
  const width = 800 - margin.left - margin.right;
  const height = 180 - margin.top - margin.bottom;

  const all = macd.concat(signal);
  const xScale = d3.scaleLinear().domain([0, macd.length - 1]).range([0, width]);
  const yScale = d3.scaleLinear().domain([d3.min(all), d3.max(all)]).range([height, 0]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  // Назва графіка
  svg.append("text")
    .attr("x", margin.left)
    .attr("y", 20)
    .text("MACD (Moving Average Convergence Divergence)")
    .attr("font-size", "14px")
    .attr("font-weight", "bold");

  // Осі
  g.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(xScale).ticks(6));
  g.append("g").call(d3.axisLeft(yScale).ticks(5));

  // Сітка
  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(yScale).ticks(5).tickSize(-width).tickFormat(""));

  const line = d3.line().x((d, i) => xScale(i)).y(d => yScale(d));

  g.append("path")
    .datum(macd)
    .attr("fill", "none")
    .attr("stroke", "#1E90FF")
    .attr("stroke-width", 1.5)
    .attr("d", line);

  g.append("path")
    .datum(signal)
    .attr("fill", "none")
    .attr("stroke", "#808080")
    .attr("stroke-dasharray", "2,2")
    .attr("stroke-width", 1.2)
    .attr("d", line);

  // Лінія 0
  g.append("line")
    .attr("x1", 0)
    .attr("x2", width)
    .attr("y1", yScale(0))
    .attr("y2", yScale(0))
    .attr("stroke", "#999")
    .attr("stroke-dasharray", "3,3");

  // Легенда
  svg.append("text")
    .attr("x", width + margin.left - 160)
    .attr("y", 20)
    .attr("fill", "#1E90FF")
    .attr("font-size", "12px")
    .text("MACD");

  svg.append("text")
    .attr("x", width + margin.left - 100)
    .attr("y", 20)
    .attr("fill", "#808080")
    .attr("font-size", "12px")
    .text("Signal");

  // Курсор
  const focusLine = g.append("line").attr("stroke", "#999").attr("stroke-dasharray", "3,3").style("display", "none");

  svg.on("mousemove", function(event) {
    const [x] = d3.pointer(event);
    const xVal = Math.round(xScale.invert(x - margin.left));
    if (xVal >= 0 && xVal < macd.length) {
      focusLine
        .style("display", null)
        .attr("x1", xScale(xVal))
        .attr("x2", xScale(xVal))
        .attr("y1", 0)
        .attr("y2", height);
    }
  }).on("mouseleave", () => {
    focusLine.style("display", "none");
  });
}


// --- Init ---
document.addEventListener("DOMContentLoaded", () => {
  ["toggle-sma", "toggle-ema", "toggle-rsi", "toggle-macd"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => loadForecastData());
  });

  loadForecastData();
});
