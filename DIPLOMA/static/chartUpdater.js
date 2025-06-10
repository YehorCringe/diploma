// chartUpdater.js

function drawMiniChart(data, selector, color) {
    const svg = d3.select(selector);
    svg.selectAll("*").remove();

    const width = +svg.attr("width");
    const height = +svg.attr("height");
    const margin = { top: 20, right: 20, bottom: 30, left: 70 };

    const xScale = d3.scaleLinear()
        .domain([0, data.length - 1])
        .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
        .domain([d3.min(data), d3.max(data)])
        .range([height - margin.bottom, margin.top]);

    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d));

    const yAxis = d3.axisLeft(yScale)
        .ticks(4)
        .tickFormat(d => d.toLocaleString("uk-UA", { minimumFractionDigits: 0 }));

    svg.append("g")
        .attr("transform", `translate(${margin.left}, 0)`)
        .call(yAxis)
        .selectAll("text")
        .style("font-size", "10px");

    svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("d", line);

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

    focus.append("circle")
        .attr("r", 4)
        .attr("fill", color)
        .style("display", "none");

    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("background", "#fff")
        .style("border", "1px solid #ccc")
        .style("padding", "4px 8px")
        .style("font-size", "12px")
        .style("pointer-events", "none")
        .style("display", "none");

    svg.append("rect")
        .attr("width", width - margin.left - margin.right)
        .attr("height", height - margin.top - margin.bottom)
        .attr("transform", `translate(${margin.left}, ${margin.top})`)
        .style("fill", "none")
        .style("pointer-events", "all")
        .on("mouseover", () => {
            focus.style("display", null);
            focus.select("circle").style("display", null);
            tooltip.style("display", null);
        })
        .on("mouseout", () => {
            focus.style("display", "none");
            tooltip.style("display", "none");
        })
        .on("mousemove", function (event) {
            const [mouseX, mouseY] = d3.pointer(event, this);
            const i = Math.round(xScale.invert(mouseX + margin.left));

            if (i >= 0 && i < data.length) {
                const x = xScale(i);
                const y = yScale(data[i]);

                focus.select(".x-hover-line")
                    .attr("transform", `translate(${x}, 0)`);

                focus.select(".y-hover-line")
                    .attr("transform", `translate(0, ${y})`);

                focus.select("circle")
                    .attr("cx", x)
                    .attr("cy", y)
                    .style("display", "block");

                tooltip
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY - 10}px`)
                    .style("display", "inline-block")
                    .html(`${data[i].toLocaleString("uk-UA", { maximumFractionDigits: 1 })} USD`);
            }
        });
}

async function fetchMiniCharts() {
    try {
        const res = await fetch('/api/mini-charts');
        const data = await res.json();

        if (data.binance.length) drawMiniChart(data.binance, "#chart-binance", "green");
        if (data.bybit.length) drawMiniChart(data.bybit, "#chart-bybit", "purple");
    } catch (e) {
        console.error("Помилка завантаження міні-графіків:", e);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    fetchMiniCharts();
    setInterval(fetchMiniCharts, 5 * 60 * 1000);
});
