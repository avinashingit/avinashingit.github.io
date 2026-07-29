const root = document.documentElement;
const toggle = document.querySelector(".theme-toggle");

if (toggle) {
  toggle.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    localStorage.setItem("portfolio-theme", next);
  });
}
