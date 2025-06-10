// mainChart.js

// Об’єкт із кольорами для різних бірж
const exchangeConfig = {
  binance: {
    realColor: "steelblue",
    forecastColor: "orange",
    animation: 800
  },
};

/**
 * Малює основний графік (реальні + прогноз) і, за потреби, індикатори.
 * @param {number[]} real       – масив останніх 60 реальних цін
 * @param {number[]} forecast   – масив прогнозу (7 точок)
 * @param {string}  exchange    – 'binance' або 'kucoin'
 * @param {Object}  indicators  – об’єкт виду { SMA_10: [...], EMA_10: [...], RSI: [...], MACD: [...], Signal_Line: [...] }
 */
function updateChart(real, forecast, exchange = "binance", indicators = {}) {
    // Спочатку видаляємо попереднє <svg> (якщо існує)
    d3.select("#chart svg").remove();

    const cfg = exchangeConfig[exchange] || exchangeConfig["binance"];
    const allData = real.concat(forecast);

    const width  = 800,
          height = 400,
          margin = { top: 20, right: 20, bottom: 50, left: 60 };

    // Додаємо новий <svg> до контейнера #chart
    const svg = d3.select("#chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Шкали
    const xScale = d3.scaleLinear()
        .domain([0, allData.length - 1])
        .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
        .domain([d3.min(allData), d3.max(allData)])
        .range([height - margin.bottom, margin.top]);

    // Вісі
    const xAxis = d3.axisBottom(xScale).ticks(10);
    const yAxis = d3.axisLeft(yScale).ticks(8);

    svg.append("g")
       .attr("transform", `translate(0, ${height - margin.bottom})`)
       .call(xAxis);

    svg.append("g")
       .attr("transform", `translate(${margin.left}, 0)`)
       .call(yAxis);

    // Лінійний генератор
    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d));

    // 1) Малюємо реальні ціни
    svg.append("path")
       .datum(real)
       .attr("fill", "none")
       .attr("stroke", cfg.realColor)
       .attr("stroke-width", 2)
       .attr("d", line);

    // 2) Малюємо прогноз (зсунутий вправо на довжину real)
    svg.append("path")
       .datum(forecast)
       .attr("fill", "none")
       .attr("stroke", cfg.forecastColor)
       .attr("stroke-width", 2)
       .attr("d", line.x((d, i) => xScale(i + real.length)));

    // === Тепер додаємо індикатори (якщо увімкнені чекбокси) ===

    // Колірна палітра для індикаторів
    const indicatorColors = {
        "SMA_10":      "#2E8B57",
        "SMA_30":      "#006400",
        "EMA_10":      "#8A2BE2",
        "EMA_30":      "#4B0082",
        "RSI":         "#DC143C",
        "MACD":        "#1E90FF",
        "Signal_Line": "#808080"
    };

    // Лінійний генератор для індикаторів з координатами {x, y}
    const indicatorLine = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y));

    // Якщо увімкнено SMA (toggle-sma), малюємо обидві лінії SMA_10 та SMA_30
    if (document.getElementById("toggle-sma").checked) {
        if (indicators.SMA_10?.length) {
            svg.append("path")
                .datum(indicators.SMA_10)
                .attr("fill", "none")
                .attr("stroke", indicatorColors["SMA_10"])
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4,2")
                .attr("d", indicatorLine);
        }
        if (indicators.SMA_30?.length) {
            svg.append("path")
                .datum(indicators.SMA_30)
                .attr("fill", "none")
                .attr("stroke", indicatorColors["SMA_30"])
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4,2")
                .attr("d", indicatorLine);
        }
    }

    // Якщо увімкнено EMA (toggle-ema), малюємо EMA_10 та EMA_30
    if (document.getElementById("toggle-ema").checked) {
        if (indicators.EMA_10?.length) {
            svg.append("path")
                .datum(indicators.EMA_10)
                .attr("fill", "none")
                .attr("stroke", indicatorColors["EMA_10"])
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4,2")
                .attr("d", indicatorLine);
        }
        if (indicators.EMA_30?.length) {
            svg.append("path")
                .datum(indicators.EMA_30)
                .attr("fill", "none")
                .attr("stroke", indicatorColors["EMA_30"])
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4,2")
                .attr("d", indicatorLine);
        }
    }

    // === Легенда ===
    const legend = svg.append("g")
        .attr("transform", `translate(${margin.left + 10}, ${margin.top + 10})`)
        .attr("class", "chart-legend");

    const legendItems = [
      { label: "Реальні ціни", color: cfg.realColor, dash: null },
      { label: "Прогноз", color: cfg.forecastColor, dash: null },
    ];

    if (document.getElementById("toggle-sma").checked) {
      legendItems.push({ label: "SMA 10/30", color: indicatorColors["SMA_10"], dash: "4,2" });
    }
    if (document.getElementById("toggle-ema").checked) {
      legendItems.push({ label: "EMA 10/30", color: indicatorColors["EMA_10"], dash: "4,2" });
    }

    legend.selectAll("g").data(legendItems).enter().append("g")
      .attr("transform", (_, i) => `translate(0, ${i * 20})`)
      .each(function (d) {
        d3.select(this)
          .append("line")
          .attr("x1", 0).attr("x2", 30).attr("y1", 5).attr("y2", 5)
          .attr("stroke", d.color)
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", d.dash ?? null);

        d3.select(this)
          .append("text")
          .attr("x", 40).attr("y", 9)
          .text(d.label)
          .style("font-size", "12px")
          .style("fill", "#333");
      });



    // ==== Додаємо курсор-перехрестя ====
    const focus = svg.append("g").style("display", "none");

    focus.append("line")
         .attr("class", "x-hover-line")
         .attr("y1", margin.top)
         .attr("y2", height - margin.bottom)
         .attr("stroke", "#999")
         .attr("stroke-dasharray", "3,3");

    focus.append("line")
         .attr("class", "y-hover-line")
         .attr("x1", margin.left)
         .attr("x2", width - margin.right)
         .attr("stroke", "#999")
         .attr("stroke-dasharray", "3,3");

    svg.append("rect")
       .attr("width", width - margin.left - margin.right)
       .attr("height", height - margin.top - margin.bottom)
       .attr("transform", `translate(${margin.left}, ${margin.top})`)
       .style("fill", "none")
       .style("pointer-events", "all")
       .on("mouseover", () => focus.style("display", null))
       .on("mouseout", () => focus.style("display", "none"))
       .on("mousemove", function (event) {
           const [xMouse] = d3.pointer(event, this);
           const x0 = Math.round(xScale.invert(xMouse + margin.left));
           if (x0 >= 0 && x0 < allData.length) {
               const x = xScale(x0);
               const y = yScale(allData[x0]);
               focus.select(".x-hover-line").attr("transform", `translate(${x},0)`);
               focus.select(".y-hover-line").attr("transform", `translate(0,${y})`);
           }
       });
} // end of updateChart


window.updateChart = updateChart;

function getSelectedIndicators() {
    return Array.from(document.querySelectorAll(".indicator-toggle:checked"))
        .map(el => el.value);
}