// MiLB -> MLB projections: top 100 table, player cards and player finder.
(function () {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const plain = (s) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();

  // Build version (from index.html) so browsers never mix old and new files.
  const V = (document.currentScript && new URL(document.currentScript.src).searchParams.get("v")) || "";
  const dataUrl = (f) => `data/${f}?v=${V}`;
  let TOP = [], ALL = null, filter = "all";

  // ---------- outcome tiers (WAR over first 6 MLB seasons; same cut-offs as src/export.py) ----------
  const CUTS = [1.5, 5, 10, 17, 25];
  const TIERS = {
    H: [
      ["Depth / up-and-down", "an up-and-down depth player"],
      ["Role player", "a role player"],
      ["Everyday regular", "an everyday regular"],
      ["Above-average regular", "an above-average regular"],
      ["All-Star", "an All-Star"],
      ["Superstar", "a superstar"],
    ],
    P: [
      ["Depth / up-and-down", "an up-and-down depth arm"],
      ["Back-end SP / reliever", "a back-end starter or reliever"],
      ["Mid-rotation / high-leverage RP", "a mid-rotation starter or high-leverage reliever"],
      ["No. 2-3 starter", "a No. 2-3 starter"],
      ["All-Star arm", "an All-Star arm"],
      ["Ace", "an ace"],
    ],
  };
  const LEVEL_TXT = ["any role", "1.5+ WAR", "5+ WAR", "10+ WAR", "17+ WAR", "25+ WAR"];
  const tierOf = (war) => CUTS.filter((c) => war >= c).length; // 0..5

  // Approximate chance of becoming AT LEAST each tier: a smooth distribution through
  // the model's floor (7th pct), expected (median) and ceiling (93rd pct), times the MLB chance.
  function atLeast(p) {
    const [lo0, m, hi0] = p.war6;
    const lo = Math.min(lo0, m - 0.05), hi = Math.max(hi0, m + 0.05);
    const pts = [[lo - (m - lo) * 0.4, 0], [lo, 0.07], [m, 0.5], [hi, 0.93], [hi + (hi - m) * 0.4, 1]];
    const cdf = (x) => {
      if (x <= pts[0][0]) return 0;
      for (let i = 1; i < pts.length; i++) {
        if (x <= pts[i][0]) {
          const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
          return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
        }
      }
      return 1;
    };
    return [p.p_mlb, ...CUTS.map((c) => (1 - cdf(c)) * p.p_mlb)];
  }

  // ---------- formatting ----------
  function fmt(stat, v) {
    if (v === null || v === undefined) return "–";
    if (["AVG", "OBP", "SLG", "OPS"].includes(stat)) return v < 1 ? v.toFixed(3).replace(/^0/, "") : v.toFixed(3);
    if (["ERA", "FIP", "WHIP"].includes(stat)) return v.toFixed(2);
    if (["HR/600", "SB/600", "wRC+"].includes(stat)) return v.toFixed(0);
    return v.toFixed(1);
  }
  const pct = (x) => (x < 0.005 ? "<1%" : Math.round(x * 100) + "%");
  const lastName = (n) => n.split(" ").filter((w) => !/^(jr\.?|sr\.?|ii|iii|iv)$/i.test(w)).pop();
  const initials = (n) => n.split(" ").map((w) => w[0]).filter((c) => /[A-ZÀ-Ý]/.test(c)).slice(0, 2).join("");
  const headshot = (p, size) =>
    `<img class="face" width="${size}" height="${size}" loading="lazy" alt="" data-ini="${esc(initials(p.name))}" ` +
    `src="https://img.mlbstatic.com/mlb-photos/image/upload/w_${size * 2},q_auto:best,d_people:generic:headshot:silo:current.png/v1/people/${p.id}/headshot/milb/current">`;

  function expLine(p) {
    const x = p.proj, m = (k) => fmt(k, x[k][1]);
    return p.type === "H"
      ? `${m("AVG")}/${m("OBP")}/${m("SLG")} · ${m("HR/600")} HR/600 · ${m("wRC+")} wRC+`
      : `${m("ERA")} ERA · ${m("K/9")} K/9 · ${m("BB/9")} BB/9`;
  }
  const tierDot = (t) => `<i class="dot t${t}" aria-hidden="true"></i>`;

  // ---------- career outlook ladder ----------
  function ladder(p) {
    const ch = atLeast(p), names = TIERS[p.type];
    const [tf, tm, tc] = p.war6.map(tierOf);
    let html = "";
    for (let t = 5; t >= 0; t--) {
      const tags = [t === tc && "Ceiling", t === tm && "Expected", t === tf && "Floor"].filter(Boolean)
        .map((s) => `<span class="tagx ${s.toLowerCase()}">${s}</span>`).join("");
      const label = t === 0 ? "Reaches MLB" : names[t][0];
      html += `
      <div class="lrow t${t}${t === tm ? " exp" : ""}">
        <div class="lname"><span>${label} ${tags}</span><small>${LEVEL_TXT[t]}</small></div>
        <div class="ltrack"><div class="lbar" style="width:${Math.max(ch[t] * 100, ch[t] > 0.004 ? 1.5 : 0)}%"></div></div>
        <div class="lpct">${pct(ch[t])}</div>
      </div>`;
    }
    return `<div class="ladder" role="img" aria-label="Chance of becoming at least each level of player">${html}</div>`;
  }

  function summary(p) {
    const n = TIERS[p.type], [tf, tm, tc] = p.war6.map(tierOf), ln = lastName(p.name);
    const perSeason = p.war6[1] / 6;
    const when = p.debuted ? `already debuted in ${p.eta}` : `has a <b>${pct(p.p_mlb)}</b> chance to reach the majors, most likely in <b>${p.eta}</b>`;
    const pace = perSeason < 0.5 ? "less than 0.5 WAR a season" : `about ${perSeason.toFixed(1)} WAR a season`;
    return `<p class="summary"><b>${esc(p.name)}</b> (${p.age}, ${p.level}) ${when}. ` +
      `If ${esc(ln)} makes it, the expected outcome is <b>${n[tm][1]}</b> (${pace}). ` +
      `Best case: ${n[tc][1]}. Worst case: ${n[tf][1]}.</p>`;
  }

  // ---------- player card ----------
  function card(p) {
    const H = p.type === "H";
    const order = H ? ["AVG", "OBP", "SLG", "OPS", "HR/600", "SB/600", "K%", "BB%", "wRC+", "WAR/yr"]
                    : ["ERA", "FIP", "WHIP", "K/9", "BB/9", "WAR/yr"];
    const rows = order.map((s) => {
      const [lo, mid, hi] = p.proj[s];
      return `<tr><td>${s}</td><td>${fmt(s, lo)}</td><td class="exp">${fmt(s, mid)}</td><td>${fmt(s, hi)}</td></tr>`;
    }).join("");
    const comps = p.comps.map((c) => {
      const t = tierOf(c.war6);
      return `<li>${tierDot(t)}<div><b>${esc(c.name)}</b><small>MLB debut ${c.debut} · became ${TIERS[p.type][t][1]} (${c.war6.toFixed(1)} WAR in 6 yrs)</small></div></li>`;
    }).join("");
    const histHead = H
      ? ["Year", "Lvl", "Team", "Age", "PA", "AVG", "OBP", "SLG", "HR", "SB", "K%", "BB%"]
      : ["Year", "Lvl", "Team", "Age", "IP", "G", "GS", "ERA", "WHIP", "K/9", "BB/9"];
    const hist = p.milb.map((r) => `<tr>${r.map((v, i) => {
      const h = histHead[i];
      const txt = typeof v === "number" && ["AVG", "OBP", "SLG", "ERA", "WHIP", "K/9", "BB/9", "K%", "BB%"].includes(h) ? fmt(h, v) : v;
      return `<td${h === "Team" ? ' class="t"' : ""}>${esc(txt)}</td>`;
    }).join("")}</tr>`).join("");

    return `<article class="card">
      <header class="card-head">
        <div class="who">
          ${headshot(p, 64)}
          <div>
            <h3>${esc(p.name)}</h3>
            <div class="pmeta">${esc(p.pos)} · ${esc(p.org)} · ${p.level} · age ${p.age}</div>
          </div>
        </div>
        <div class="chips">
          <div class="chip"><div class="v">${p.rank ? "#" + p.rank : "—"}</div><div class="l">Prospect rank</div></div>
          <div class="chip"><div class="v">${p.eta}</div><div class="l">${p.debuted ? "Debuted" : "ETA"}</div></div>
          <div class="chip"><div class="v">${pct(p.p_mlb)}</div><div class="l">MLB chance</div></div>
        </div>
      </header>
      ${summary(p)}
      <div class="grid">
        <section>
          <h4>Career outlook</h4>
          <p class="explain">Chance to become <b>at least</b> each level of player. Levels use <b>WAR</b> (wins above replacement: a player's total value, in wins) over the first 6 MLB seasons. For scale, about 2 WAR a season is a solid starter and 5+ is an All-Star.</p>
          ${ladder(p)}
          <p class="note">Floor, expected and ceiling are the model's bad-case (7th percentile), median and best-case (93rd percentile) projections if the player reaches MLB: ${p.war6.map((w) => w.toFixed(1)).join(" / ")} WAR. Chances are approximate and include the risk of never reaching MLB.</p>
        </section>
        <section>
          <h4>Expected MLB stats <span class="muted">· avg. of first 3 seasons</span></h4>
          <table class="proj"><thead><tr><th>Stat</th><th>Floor</th><th>Expected</th><th>Ceiling</th></tr></thead><tbody>${rows}</tbody></table>
          <p class="note">Debut year + next 2 seasons combined, not just the rookie year.${H ? " HR and SB per 600 PA (about one full season)." : ""} WAR/yr = total WAR over those 3 seasons ÷ 3.</p>
        </section>
      </div>
      <section class="comps-wrap">
        <h4>Players with a similar minor-league profile</h4>
        <ul class="comps">${comps || '<li class="muted">No close comparisons.</li>'}</ul>
      </section>
      <section>
        <h4>Minor league stats</h4>
        <div class="hist-wrap"><table class="hist"><thead><tr>${histHead.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${hist}</tbody></table></div>
      </section>
    </article>`;
  }

  // ---------- top 100 table ----------
  function renderTop() {
    const list = TOP.filter((p) => filter === "all" || p.type === filter);
    $("#top-table tbody").innerHTML = list.map((p) => {
      const [, tm, tc] = p.war6.map(tierOf), n = TIERS[p.type];
      return `
      <tr class="row" tabindex="0" data-id="${p.id}${p.type}" aria-haspopup="dialog">
        <td class="num rank">${p.rank}</td>
        <td class="pl"><div class="who sm">${headshot(p, 36)}<div><div class="pname">${esc(p.name)}</div>
          <div class="pmeta">${esc(p.pos)} · ${esc(p.org)} · ${p.level} · ${p.age}</div></div></div></td>
        <td class="num eta">${p.eta}${p.debuted ? '<small>debuted</small>' : ""}</td>
        <td class="ch"><div class="meter" title="${pct(p.p_mlb)} chance to reach MLB"><div class="mtrack"><div class="mfill" style="width:${p.p_mlb * 100}%"></div></div><span>${pct(p.p_mlb)}</span></div></td>
        <td class="hide-md line">${expLine(p)}</td>
        <td class="ol"><div class="outlook">${tierDot(tm)}<span>${n[tm][0]}</span></div><small class="ceil">Ceiling: ${n[tc][0]}</small></td>
      </tr>`;
    }).join("");
  }

  function openSheet(tr) {
    const p = TOP.find((x) => x.id + x.type === tr.dataset.id), dlg = $("#sheet");
    $("#sheet-body").innerHTML = card(p);
    dlg.showModal();
    dlg.scrollTop = 0;
  }

  // ---------- search ----------
  let matches = [], active = 0;
  function loadAll() {
    if (!ALL) ALL = fetch(dataUrl("players.json")).then(okJson).then((d) => d.map((p) => ((p.q = plain(p.name)), p)));
    return ALL;
  }
  async function suggest() {
    const box = $("#suggest"), words = plain($("#q").value.trim()).split(/\s+/).filter(Boolean);
    if (!words.length) { box.classList.remove("show"); return; }
    const all = await loadAll();
    matches = all.filter((p) => words.every((w) => p.q.includes(w))).sort((a, b) => b.score - a.score).slice(0, 8);
    active = 0;
    box.innerHTML = matches.length
      ? matches.map((p, i) => `<li role="option" data-i="${i}" class="${i === 0 ? "active" : ""}">${headshot(p, 28)}<span class="sname">${esc(p.name)}</span><span class="pmeta">${esc(p.pos)} · ${esc(p.org)} · ${p.level}${p.rank ? " · #" + p.rank : ""}</span></li>`).join("")
      : `<li class="muted">No current minor leaguer matches.</li>`;
    box.classList.add("show");
  }
  function choose(p) {
    if (!p) return;
    $("#suggest").classList.remove("show");
    $("#q").value = p.name;
    $("#card-slot").innerHTML = card(p);
    history.replaceState(null, "", "#p" + p.id + p.type);
  }

  // ---------- boot ----------
  function okJson(r) {
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${r.url}`);
    return r.json();
  }
  function showError(e) {
    let el = $("#loading");
    if (!el) { el = document.createElement("p"); el.className = "loading"; $("#top100").appendChild(el); }
    el.textContent = location.protocol === "file:"
      ? "Open this page through its website link: browsers block loading the data from a local file."
      : `Could not load the data (${e.message}). Try a hard refresh (Ctrl/Cmd + Shift + R).`;
  }

  async function boot() {
    const [meta, top] = await Promise.all([
      fetch(dataUrl("meta.json")).then(okJson),
      fetch(dataUrl("top100.json")).then(okJson),
    ]);
    TOP = top;
    document.querySelectorAll("[data-meta]").forEach((el) => (el.textContent = meta[el.dataset.meta]));
    $("#loading").remove();
    renderTop();
    loadAll();
    if (location.hash.startsWith("#p")) {
      const key = location.hash.slice(2), all = await loadAll();
      const p = all.find((x) => x.id + x.type === key);
      if (p) { choose(p); $("#finder").scrollIntoView(); }
    }
  }

  // Missing headshot -> initials badge
  document.addEventListener("error", (e) => {
    const img = e.target;
    if (img.tagName === "IMG" && img.classList.contains("face")) {
      const s = document.createElement("span");
      s.className = "face ini";
      s.style.width = s.style.height = img.getAttribute("width") + "px";
      s.textContent = img.dataset.ini;
      img.replaceWith(s);
    }
  }, true);

  document.addEventListener("DOMContentLoaded", () => {
    boot().catch(showError);
    document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
      filter = b.dataset.filter; renderTop();
    }));
    const tbody = $("#top-table tbody");
    tbody.addEventListener("click", (e) => { const tr = e.target.closest("tr.row"); if (tr) openSheet(tr); });
    tbody.addEventListener("keydown", (e) => { const tr = e.target.closest("tr.row"); if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openSheet(tr); } });
    const dlg = $("#sheet");
    $("#sheet-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    const q = $("#q"), box = $("#suggest");
    q.addEventListener("input", suggest);
    q.addEventListener("focus", loadAll);
    q.addEventListener("keydown", (e) => {
      const items = box.querySelectorAll("li[data-i]");
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!items.length) return;
        active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        items.forEach((li, i) => li.classList.toggle("active", i === active));
      } else if (e.key === "Enter") { e.preventDefault(); choose(matches[active]); }
      else if (e.key === "Escape") box.classList.remove("show");
    });
    box.addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-i]"); if (li) { e.preventDefault(); choose(matches[+li.dataset.i]); } });
    document.addEventListener("click", (e) => { if (!e.target.closest(".search")) box.classList.remove("show"); });
  });
})();
