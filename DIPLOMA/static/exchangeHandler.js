document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("exchangeSelect");

    select.addEventListener("change", async () => {
        const exchange = select.value;

        const res = await fetch(`/api/forecast/${exchange}`);
        const data = await res.json();

        renderMainChart(data.real, data.forecast);  // ця функція повинна перебудовувати SVG графік
    });
});
