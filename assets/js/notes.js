(() => {
  const browser = document.querySelector("[data-notes-browser]");
  if (!browser) return;

  const results = browser.querySelector("[data-notes-results]");
  const label = browser.querySelector("[data-results-label]");
  const count = browser.querySelector("[data-results-count]");
  const reset = browser.querySelector("[data-notes-reset]");
  const keywordButtons = [...browser.querySelectorAll("[data-keyword]")];
  const noteLinks = [...browser.querySelectorAll("[data-note-view]")];
  const listMarkup = results.innerHTML;

  const clearSelection = () => {
    keywordButtons.forEach((button) => {
      button.classList.remove("is-active");
      button.setAttribute("aria-pressed", "false");
    });
    noteLinks.forEach((link) => link.classList.remove("is-active"));
  };

  const showList = (keyword = null, updateHistory = true) => {
    results.innerHTML = listMarkup;
    clearSelection();
    reset.classList.toggle("is-active", !keyword);

    const cards = [...results.querySelectorAll("[data-note-card]")];
    let visible = cards.length;

    if (keyword) {
      visible = 0;
      label.textContent = keyword;
      cards.forEach((card) => {
        const matches = card.dataset.keywords.split("|").includes(keyword);
        card.hidden = !matches;
        if (matches) visible += 1;
      });
      const activeButton = keywordButtons.find((button) => button.dataset.keyword === keyword);
      if (activeButton) {
        activeButton.classList.add("is-active");
        activeButton.setAttribute("aria-pressed", "true");
        label.textContent = activeButton.textContent.trim();
      }
    } else {
      label.textContent = "All notes · Recent first";
    }

    count.textContent = visible;
    if (updateHistory) {
      const nextUrl = keyword ? `?keyword=${encodeURIComponent(keyword)}` : window.location.pathname;
      history.pushState({ keyword }, "", nextUrl);
    }
  };

  const showNote = async (link, updateHistory = true) => {
    clearSelection();
    reset.classList.remove("is-active");
    link.classList.add("is-active");
    label.textContent = "Loading note…";
    count.textContent = "";

    try {
      const response = await fetch(link.href);
      if (!response.ok) throw new Error(`Unable to load note (${response.status})`);
      const page = new DOMParser().parseFromString(await response.text(), "text/html");
      const sourceHeader = page.querySelector(".note-header");
      const sourceBody = page.querySelector(".note-prose");
      if (!sourceHeader || !sourceBody) throw new Error("Note content was not found");

      sourceHeader.querySelector(".back-link")?.remove();
      sourceBody.querySelectorAll("[src]").forEach((node) => {
        node.setAttribute("src", new URL(node.getAttribute("src"), response.url).href);
      });

      const viewer = document.createElement("article");
      viewer.className = "embedded-note";
      const viewerHeader = document.createElement("header");
      viewerHeader.className = "embedded-note-header";
      viewerHeader.append(...sourceHeader.childNodes);
      viewer.append(viewerHeader, sourceBody);
      results.replaceChildren(viewer);

      const title = viewerHeader.querySelector("h1")?.textContent.trim() || link.textContent.trim();
      label.textContent = title;

      if (window.MathJax?.typesetPromise) {
        window.MathJax.typesetPromise([viewer]).catch(() => {});
      }
      if (window.siteMermaid && viewer.querySelector(".mermaid")) {
        window.siteMermaid.run({ nodes: viewer.querySelectorAll(".mermaid"), suppressErrors: true });
      }

      if (updateHistory) {
        history.pushState({ noteHref: link.href }, "", `?note=${encodeURIComponent(link.getAttribute("href"))}`);
      }
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch {
      window.location.href = link.href;
    }
  };

  reset.addEventListener("click", () => showList());
  keywordButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const keyword = button.dataset.keyword;
      const isActive = button.classList.contains("is-active");
      showList(isActive ? null : keyword);
    });
  });
  noteLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showNote(link);
    });
  });

  window.addEventListener("popstate", (event) => {
    if (event.state?.noteHref) {
      const link = noteLinks.find((item) => item.href === event.state.noteHref);
      if (link) showNote(link, false);
    } else {
      showList(event.state?.keyword || null, false);
    }
  });

  const params = new URLSearchParams(window.location.search);
  const initialKeyword = params.get("keyword");
  const initialNote = params.get("note");
  if (initialNote) {
    const noteUrl = new URL(initialNote, window.location.origin).href;
    const link = noteLinks.find((item) => item.href === noteUrl);
    if (link) showNote(link, false);
  } else if (initialKeyword) {
    showList(initialKeyword, false);
  }
})();
