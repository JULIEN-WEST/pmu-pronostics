"""
Page HTML autonome — la vue « visuelle » servie par l'API.

Pourquoi une page plutôt que des cartes Home Assistant : une carte
markdown ne sait pas dessiner. Elle rend du texte, et une jauge en
caractères `█░░░` reste une jauge en caractères. Ici on dispose de vraies
barres, d'un podium, de couleurs qui portent une information, et surtout
de la place nécessaire pour AFFICHER LE POURQUOI à côté du combien.

Elle s'intègre dans Home Assistant par une carte `iframe`, sans aucune
dépendance HACS.

Contraintes tenues :
  - un seul fichier, aucune ressource externe (ni police, ni script, ni
    image) : la page doit s'afficher sur un réseau sans Internet ;
  - thème clair et sombre, suivis automatiquement, parce qu'une iframe
    hérite du thème du système et non de celui de Home Assistant ;
  - toutes les données du jour sont embarquées à la génération : la
    navigation entre courses ne provoque aucune requête.
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime

# ---------------------------------------------------------------------

_DISCIPLINES = {
    "ATTELE": "Attelé", "MONTE": "Monté", "PLAT": "Plat",
    "HAIES": "Haies", "STEEPLECHASE": "Steeple-chase", "CROSS": "Cross",
}

_PARTICULES = ((" Du ", " du "), (" De ", " de "), (" Des ", " des "),
               (" Le ", " le "), (" La ", " la "), (" Les ", " les "),
               (" Et ", " et "), (" Au ", " au "), (" Aux ", " aux "),
               (" D'", " d'"), (" L'", " l'"))


def joli(texte) -> str:
    """« JULIE DU NORD » → « Julie du Nord ». Les majuscules fatiguent."""
    if not texte:
        return ""
    t = str(texte).strip().lower().title()
    for avant, apres in _PARTICULES:
        t = t.replace(avant, apres)
    return t


def lieu(texte) -> str:
    if not texte:
        return ""
    t = str(texte).upper()
    for prefixe in ("HIPPODROME DE ", "HIPPODROME D'", "HIPPODROME "):
        if t.startswith(prefixe):
            t = t[len(prefixe):]
            break
    return joli(t)


def _preparer(courses: list[dict]) -> list[dict]:
    """Allège et met en forme, pour que le JavaScript n'ait rien à décider."""
    out = []
    for c in courses:
        sel = []
        for s in c.get("selection") or []:
            sel.append({
                "num": s.get("num"),
                "cheval": joli(s.get("cheval")) or "?",
                "driver": joli(s.get("driver")),
                "entraineur": joli(s.get("entraineur")),
                "pere": joli(s.get("pere")),
                "pere_mere": joli(s.get("pere_mere")),
                "musique": (s.get("musique") or "").strip(),
                "age": s.get("age"), "sexe": (s.get("sexe") or "").lower(),
                "corde": s.get("corde"),
                "nb_courses": s.get("nb_courses"),
                "nb_victoires": s.get("nb_victoires"),
                "gains": s.get("gains"),
                "proba": s.get("proba"), "rang": s.get("rang"),
                "cote": s.get("cote"), "valeur": s.get("valeur"),
                "arrivee": s.get("arrivee"),
                "motifs": s.get("motifs") or [],
                "faits": s.get("faits") or {},
            })
        depart = c.get("depart")
        heure = ""
        if depart:
            try:
                heure = datetime.fromisoformat(depart).strftime("%H:%M")
            except ValueError:
                heure = ""
        out.append({
            "code": c.get("code"), "libelle": joli(c.get("libelle")),
            "hippodrome": lieu(c.get("hippodrome")),
            "discipline": _DISCIPLINES.get(c.get("discipline"), c.get("discipline") or ""),
            "distance": c.get("distance"), "terrain": joli(c.get("terrain")),
            "allocation": c.get("allocation"),
            "partants": c.get("partants") or len(sel),
            "heure": heure, "confiance": c.get("confiance") or 0,
            "arrivee_connue": bool(c.get("arrivee_connue")),
            # Faux = le modèle n'a rien vu de net. On affiche quand même
            # la course, mais on le dit : masquer reviendrait à cacher
            # ses échecs, et le taux de réussite ne serait plus lisible.
            "publiable": bool(c.get("publiable", True)),
            "selection": sel,
        })
    return out


# ---------------------------------------------------------------------

_CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --fond:#f4f5f7; --carte:#fff; --texte:#16181d; --doux:#5d6470;
  --trait:#e2e5ea; --accent:#2d6df6; --vert:#0f9d58; --rouge:#d93025;
  --or:#c8961e; --barre:#dfe3e9;
  --r:14px; --ombre:0 1px 2px rgba(0,0,0,.06),0 4px 14px rgba(0,0,0,.05);
}
@media (prefers-color-scheme:dark){:root{
  --fond:#111418; --carte:#191d23; --texte:#e8eaed; --doux:#9aa2ad;
  --trait:#272c34; --accent:#7aa5ff; --vert:#4cc38a; --rouge:#ff6b60;
  --or:#e0b64d; --barre:#252a32;
  --ombre:0 1px 2px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.3);
}}
html,body{margin:0;padding:0;height:100%}
/* ── LA PAGE NE DÉFILE PAS ────────────────────────────────────────
   Elle est affichée dans une iframe de hauteur fixe. Si son contenu
   dépasse, un ascenseur interne apparaît, et un ascenseur dans un
   cadre lui-même dans une page qui défile est pénible à utiliser.
   La page occupe donc exactement la hauteur disponible, et c'est le
   JavaScript qui décide combien de partants tiennent — les autres
   passent derrière un bouton. Voir `ajuster()`. */
body{background:var(--fond);color:var(--texte);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:10px;-webkit-font-smoothing:antialiased;
  display:flex;flex-direction:column;overflow:hidden}
body.libre{overflow:auto;height:auto}
h1,h2,h3{margin:0;font-weight:650;letter-spacing:-.01em}
.carte{background:var(--carte);border:1px solid var(--trait);
  border-radius:var(--r);box-shadow:var(--ombre);padding:13px 15px}
#course{flex:1 1 auto;min-height:0;overflow:hidden}
#course>.carte{height:100%;overflow:hidden;display:flex;flex-direction:column}
#classement{min-height:0}

/* ── bandeau ─────────────────────────────────────────── */
.entete{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;
  margin-bottom:9px;flex:0 0 auto}
.entete h1{font-size:17px}
.puce{font-size:12px;color:var(--doux);background:var(--barre);
  padding:3px 9px;border-radius:999px;white-space:nowrap}
.puce.alerte{background:rgba(217,48,37,.14);color:var(--rouge)}
/* Classe distincte de `.alerte` : une course sous le seuil n'est pas
   une panne, et confondre les deux rendrait l'alerte de fraîcheur
   indétectable — un test l'a montré. */
.puce.seuil{background:rgba(200,150,30,.16);color:var(--or)}

/* ── sélecteur de courses ────────────────────────────── */
.rail{display:flex;gap:6px;overflow-x:auto;padding:2px 2px 8px;
  scrollbar-width:thin;flex:0 0 auto}
.rail button{flex:0 0 auto;background:var(--carte);color:var(--texte);
  border:1px solid var(--trait);border-radius:11px;padding:7px 11px;
  font:inherit;font-size:13px;cursor:pointer;text-align:left;line-height:1.25;
  transition:border-color .12s,transform .12s}
.rail button:hover{border-color:var(--accent)}
.rail button.actif{border-color:var(--accent);
  box-shadow:inset 0 0 0 1px var(--accent)}
.rail b{display:block;font-size:12px;letter-spacing:.02em}
/* `block` est indispensable : sans lui les deux libellés se collent et
   on lit « La Capellefavori battu ». */
.rail span{display:block;color:var(--doux);font-size:11px}
.rail .ok{color:var(--vert)} .rail .ko{color:var(--rouge)}
/* Troisième état : arrivée connue mais place du favori absente. Ce
   n'est pas un échec du modèle, c'est une donnée manquante. */
.rail .inconnu{color:var(--or)}
/* Course sous le seuil de confiance : visible, mais en retrait. */
.rail button.muette{opacity:.5}

/* ── en-tête de course ───────────────────────────────── */
.titre{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.titre h2{font-size:17px}
.meta{color:var(--doux);font-size:13px}

/* ── podium ──────────────────────────────────────────── */
.podium{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:8px;margin:10px 0 2px;flex:0 0 auto}
.marche{border:1px solid var(--trait);border-radius:11px;padding:8px 11px;
  background:linear-gradient(180deg,var(--barre) 0,transparent 70%)}
.marche .place{font-size:10.5px;color:var(--doux);letter-spacing:.08em;
  text-transform:uppercase}
.marche .nom{font-weight:650;margin:2px 0 0;font-size:14.5px}
.marche .pc{font-size:23px;font-weight:700;letter-spacing:-.02em}
.marche.or{border-color:var(--or)} .marche.or .pc{color:var(--or)}

/* ── tableau du classement ───────────────────────────── */
.ligne{display:grid;grid-template-columns:28px 1fr 92px;gap:10px;
  align-items:center;padding:7px 4px;border-top:1px solid var(--trait);
  cursor:pointer}
.ligne.masquee,.podium.masquee{display:none}
.reste{width:100%;margin-top:8px;background:var(--barre);color:var(--texte);
  border:1px solid var(--trait);border-radius:10px;padding:7px;font:inherit;
  font-size:12.5px;cursor:pointer;flex:0 0 auto}
.reste:hover{border-color:var(--accent)}
.ligne:hover{background:var(--barre)}
.ligne .num{font-variant-numeric:tabular-nums;font-weight:650;
  text-align:center;color:var(--doux)}
.ligne .nom{font-weight:600}
.ligne .sous{color:var(--doux);font-size:12px;margin-top:1px}
.jauge{height:7px;border-radius:4px;background:var(--barre);overflow:hidden;
  margin-top:6px}
.jauge i{display:block;height:100%;border-radius:4px;background:var(--accent)}
.ligne.top .jauge i{background:var(--or)}
.chiffres{text-align:right;font-variant-numeric:tabular-nums}
.chiffres .pc{font-weight:700;font-size:15px}
.chiffres .cote{color:var(--doux);font-size:12px}
.val{font-size:11px;padding:1px 6px;border-radius:999px;
  background:rgba(15,157,88,.15);color:var(--vert);display:inline-block}
.val.neg{background:var(--barre);color:var(--doux)}

/* ── justification, en panneau superposé ──────────────────────────
   Dépliée en ligne, elle rallongeait la page de 400 px et ramenait
   l'ascenseur qu'on cherche à supprimer. Elle recouvre donc le cadre :
   la hauteur totale ne bouge plus, et le détail dispose de toute la
   place. */
.feuille{position:fixed;inset:0;background:var(--fond);z-index:9;
  padding:12px;overflow:auto;display:flex;flex-direction:column;gap:10px}
.feuille .barre{display:flex;align-items:flex-start;gap:12px}
.feuille .barre div{flex:1}
.feuille h3{font-size:17px}
.feuille .gros{font-size:23px;font-weight:700;letter-spacing:-.02em}
.feuille .fermer{background:var(--barre);border:1px solid var(--trait);
  color:var(--texte);border-radius:9px;width:32px;height:32px;font:inherit;
  font-size:15px;cursor:pointer;flex:0 0 auto}
.feuille .fermer:hover{border-color:var(--accent)}
.detail{padding:0}
.motifs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.motif{font-size:12px;border-radius:9px;padding:4px 9px;
  border:1px solid var(--trait);background:var(--barre)}
.motif.plus{border-color:rgba(15,157,88,.45)}
.motif.moins{border-color:rgba(217,48,37,.35)}
.motif b{font-weight:650}
.faits{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:8px 18px}
.faits section{font-size:12.5px}
.faits h4{margin:0 0 2px;font-size:11px;color:var(--doux);
  text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.faits li{margin:1px 0}
.faits ul{margin:0;padding-left:16px}
.vide{color:var(--doux);font-style:italic}
.note{color:var(--doux);font-size:12px;margin-top:10px;line-height:1.45}
"""

_JS = r"""
const F = (x, n=1) => (x==null || isNaN(x)) ? "—" : (x*100).toFixed(n)+" %";
const E = s => String(s??"").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

let ouvert = new Set();

function motif(m){
  const plus = m.sens === "+";
  const det = (m.details||[]).slice(0,2).join(" · ");
  return `<span class="motif ${plus?"plus":"moins"}">${plus?"▲":"▼"}
    <b>${E(m.titre)}</b>${det?" — "+E(det):""}</span>`;
}

function faits(f){
  const ordre = ["forme","vitesse","historique","aptitude","lignee",
                 "entourage","palmares","marge","conditions","marche"];
  const titres = {forme:"Forme",vitesse:"Chrono",historique:"Historique",
    aptitude:"Aptitude",lignee:"Lignée",entourage:"Entourage",
    palmares:"Palmarès",marge:"Arrivées",conditions:"Conditions",marche:"Marché"};
  // Trois faits par famille suffisent : au-delà on ne lit plus, on scrolle.
  const blocs = ordre.filter(k => (f[k]||[]).length).map(k =>
    `<section><h4>${titres[k]}</h4><ul>${
      f[k].slice(0,3).map(t=>`<li>${E(t)}</li>`).join("")}</ul></section>`);
  return blocs.length ? `<div class="faits">${blocs.join("")}</div>`
    : `<div class="vide">Aucune donnée exploitable pour ce partant — le modèle
       l'a noté sans historique.</div>`;
}

function rendreCourse(c){
  const sel = c.selection;
  const maxi = Math.max(...sel.map(s=>s.proba||0), 0.0001);
  const podium = sel.slice(0,3).map((s,i)=>`
    <div class="marche ${i===0?"or":""}">
      <div class="place">${["Favori du modèle","2ᵉ choix","3ᵉ choix"][i]}</div>
      <div class="nom">${s.num} · ${E(s.cheval)}</div>
      <div class="pc">${F(s.proba)}</div>
      <div class="meta">${E(s.driver||"")}</div>
    </div>`).join("");

  const lignes = sel.map((s,i)=>{
    const val = s.valeur;
    const badge = (val==null) ? "" :
      `<span class="val ${val>0.15?"":"neg"}">${val>0?"+":""}${(val*100).toFixed(0)} %</span>`;
    const arrivee = c.arrivee_connue && s.arrivee
      ? `<span class="puce">arrivé ${s.arrivee}${s.arrivee===1?"ᵉʳ":"ᵉ"}</span>` : "";
    return `<div class="ligne ${i===0?"top":""}" data-i="${i}">
        <div class="num">${s.num}</div>
        <div>
          <div class="nom">${E(s.cheval)} ${arrivee}</div>
          <div class="sous">${E(s.driver||"—")}${s.musique?" · "+E(s.musique):""}
            ${s.pere?" · par "+E(s.pere):""}</div>
          <div class="jauge"><i style="width:${Math.max(2,(s.proba/maxi)*100)}%"></i></div>
        </div>
        <div class="chiffres">
          <div class="pc">${F(s.proba)}</div>
          <div class="cote">${s.cote!=null?"cote "+s.cote.toFixed(1):"cote —"}</div>
          ${badge}
        </div>
      </div>`;
  }).join("");

  const conf = c.confiance*100;
  const niveau = conf>10?"élevée":(conf>5?"moyenne":"faible");
  return `<div class="carte">
      <div class="titre">
        <h2>${E(c.code)} · ${E(c.hippodrome)}</h2>
        <span class="puce">${E(c.heure)}</span>
        <span class="puce">confiance ${niveau}</span>
        ${c.publiable===false?'<span class="puce seuil">sous le seuil de confiance</span>':""}
      </div>
      <div class="meta">${E(c.libelle)} — ${E(c.discipline)} ·
        ${c.distance} m · ${c.partants} partants${
        c.terrain?" · terrain "+E(c.terrain):""}${
        c.allocation?" · "+Math.round(c.allocation).toLocaleString("fr-FR")+" €":""}</div>
      <div class="podium">${podium}</div>
      <div id="classement">${lignes}</div>
    </div>`;
}

// ── La justification, en panneau superposé ─────────────────────────
function ouvrirFeuille(i){
  const c = DONNEES.courses[courant], s = c.selection[i];
  const f = document.createElement("div");
  f.className = "feuille";
  f.innerHTML = `
    <div class="barre">
      <div>
        <h3>${s.num} · ${E(s.cheval)}</h3>
        <div class="meta">${E(c.code)} · ${E(c.hippodrome)} — ${E(s.driver||"")}${
          s.pere?" · par "+E(s.pere):""}</div>
      </div>
      <div style="text-align:right;flex:0 0 auto">
        <div class="gros">${F(s.proba)}</div>
        <div class="meta">${s.cote!=null?"cote "+s.cote.toFixed(1):"cote —"}</div>
      </div>
      <button class="fermer" title="Fermer">✕</button>
    </div>
    <div class="motifs">${(s.motifs||[]).map(motif).join("") ||
      '<span class="vide">Aucun facteur ne se détache nettement dans ce lot.</span>'}</div>
    ${faits(s.faits||{})}
    <div class="note">Ces motifs disent ce qui a pesé sur la note dans ce lot
      précis. Ce sont des associations mesurées, pas des causes.</div>`;
  f.querySelector(".fermer").onclick = () => f.remove();
  f.onclick = e => { if (e.target === f) f.remove(); };
  document.addEventListener("keydown", function esc(e){
    if (e.key === "Escape") { f.remove(); document.removeEventListener("keydown", esc); }
  });
  document.body.appendChild(f);
}

// ── Ajustement à la hauteur réelle du cadre ────────────────────────
//
// On ne peut pas savoir d'avance combien de partants tiennent : la
// hauteur de l'iframe dépend de l'`aspect_ratio` choisi, de la largeur
// de la colonne et de l'écran. On mesure donc APRÈS le rendu, et on
// masque les lignes qui débordent. Le bouton lui-même occupe de la
// place : il est ajouté avant la mesure finale, sinon il déclencherait
// à son tour un dépassement d'un cran.
function ajuster(){
  if (toutAfficher) return;
  const zone = document.getElementById("course");
  const carte = zone && zone.firstElementChild;
  if (!carte) return;
  const lignes = [...carte.querySelectorAll(".ligne")];
  lignes.forEach(l => l.classList.remove("masquee"));
  const podium = carte.querySelector(".podium");
  if (podium) podium.classList.remove("masquee");
  if (carte.scrollHeight <= carte.clientHeight + 2) return;

  // Le podium avant les partants : il REDIT les trois premières lignes
  // du classement. Dans un cadre serré, il coûte deux partants pour ne
  // rien apprendre de neuf — c'est donc lui qui saute en premier.
  if (podium){
    podium.classList.add("masquee");
    if (carte.scrollHeight <= carte.clientHeight + 2) return;
  }

  const bouton = document.createElement("button");
  bouton.className = "reste";
  carte.appendChild(bouton);

  let i = lignes.length - 1;
  // On garde toujours trois partants : en dessous, la carte ne dit plus
  // rien et mieux vaut un cadre trop court qu'un cadre inutile.
  while (carte.scrollHeight > carte.clientHeight + 2 && i >= 3){
    lignes[i].classList.add("masquee"); i--;
  }
  const caches = lignes.length - 1 - i;
  if (caches <= 0){ bouton.remove(); return; }
  bouton.textContent = `Afficher les ${caches} autres partant`
    + (caches > 1 ? "s" : "");
  bouton.onclick = () => { toutAfficher = true; document.body.classList.add("libre"); rendre(); };
}

function rendre(){
  const c = DONNEES.courses[courant];
  document.getElementById("course").innerHTML = c ? rendreCourse(c)
    : '<div class="carte vide">Aucune course à afficher.</div>';
  document.querySelectorAll(".rail button").forEach((b,i)=>
    b.classList.toggle("actif", i===courant));
  document.querySelectorAll(".ligne").forEach(l =>
    l.onclick = () => ouvrirFeuille(+l.dataset.i));
  ajuster();
}

let courant = 0, toutAfficher = false;
document.querySelectorAll(".rail button").forEach((b,i)=>
  b.onclick=()=>{
    courant = i; toutAfficher = false;
    document.body.classList.remove("libre");
    document.querySelectorAll(".feuille").forEach(f=>f.remove());
    rendre();
  });
rendre();
// La largeur du cadre change quand la fenêtre change : le nombre de
// partants qui tiennent change avec elle.
addEventListener("resize", () => { if (!toutAfficher) rendre(); });

// Rechargement périodique AVEC HORODATAGE dans l'URL.
//
// Le paramètre n'est pas décoratif : quand la page est déposée dans le
// dossier `www` de Home Assistant, elle est servie comme un fichier
// statique et peut rester en cache dans le navigateur. Recharger la
// même URL ne garantit donc rien ; changer l'URL, si. On ne recharge
// pas non plus si une justification est dépliée — ce serait la refermer
// sous les yeux de celui qui la lit.
if (RAFRAICHIR > 0) setInterval(() => {
  if (ouvert.size === 0 && !document.hidden)
    location.replace(location.pathname + "?t=" + Math.floor(Date.now()/1000));
}, RAFRAICHIR * 1000);
"""


def _verdict(c: dict) -> str:
    """
    Verdict d'une course passée, affiché sous son code dans le rail.

    ⚠️ TROIS ÉTATS, PAS DEUX. Une arrivée connue au niveau de la course
    ne veut pas dire que la place du favori l'est : quand la propagation
    vers les partants n'a pas eu lieu, `arrivee` vaut None. Le rendre
    « favori battu » a fait croire pendant deux jours que le modèle se
    trompait sur absolument tout — alors qu'il manquait une donnée.

    Et quand la place est connue, l'annoncer : « favori 4ᵉ » se lit
    beaucoup mieux qu'un « battu » qui met la deuxième place et la
    douzième dans le même sac.
    """
    if not c.get("arrivee_connue"):
        return ""
    sel = c.get("selection") or []
    place = sel[0].get("arrivee") if sel else None
    if place is None:
        return '<span class="inconnu">arrivée non renseignée</span>'
    if place == 1:
        return '<span class="ok">favori gagnant</span>'
    return f'<span class="ko">favori {place}ᵉ</span>'


def page(courses: list[dict], *, jour: date, meta: dict | None = None,
         rafraichir: int = 300) -> str:
    """
    Rend la page complète, prête à servir.

    `rafraichir` : période d'auto-rechargement en secondes, 0 pour aucun.
    Utile parce que la page finit souvent figée dans une iframe qu'on ne
    recharge jamais à la main.
    """
    meta = meta or {}
    data = _preparer(courses)
    # ⚠️ `json.dumps` n'échappe NI « < » NI « > ». Un cheval nommé
    # « </script>… » — ou n'importe quelle chaîne venue de l'API PMU —
    # refermerait donc la balise et casserait la page, voire y ferait
    # exécuter du code. Les séquences \uXXXX sont du JSON valide et du
    # JavaScript valide : le contenu est identique après analyse, mais
    # le navigateur ne voit plus de balise.
    charge = (json.dumps({"courses": data}, ensure_ascii=False)
              .replace("<", "\\u003c")
              .replace(">", "\\u003e")
              .replace("&", "\\u0026")
              .replace("\u2028", "\\u2028")   # séparateurs de ligne Unicode :
              .replace("\u2029", "\\u2029"))  # littéraux illégaux en JS

    if data:
        rail = "".join(
            f'<button class="{"" if c["publiable"] else "muette"}">'
            f'<b>{html.escape(c["code"] or "")}</b>'
            f'<span>{html.escape(c["heure"])} · {html.escape(c["hippodrome"])}</span>'
            + _verdict(c)
            + "</button>"
            for c in data
        )
    else:
        rail = ""

    age = meta.get("age_heures")
    frais = meta.get("frais", True)
    puces = [
        f'<span class="puce">{len(data)} courses</span>',
        f'<span class="puce">modèle {html.escape(str(meta.get("modele", "—")))}</span>',
    ]
    if age is not None:
        classe = "puce" if frais else "puce alerte"
        puces.append(f'<span class="{classe}">calculé il y a {age} h</span>')

    corps = (f'<div class="rail">{rail}</div><div id="course"></div>'
             if data else
             '<div class="carte"><h2>Aucun pronostic pour cette date</h2>'
             '<p class="note">Soit le programme est vide, soit la collecte '
             'n\'a pas encore tourné. La page se remplira d\'elle-même.</p></div>')

    return f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Pronostics PMU — {jour.strftime('%d/%m/%Y')}</title>
<style>{_CSS}</style>
</head><body>
<div class="entete">
  <h1>Pronostics du {jour.strftime('%d/%m/%Y')}</h1>
  {''.join(puces)}
</div>
{corps}
<div class="note carte">
  Les probabilités sont <b>calibrées</b> : quand le modèle annonce 20 %,
  l'événement se produit environ une fois sur cinq. La <b>valeur</b> vaut
  probabilité × cote − 1 ; positive, elle signale un écart avec le marché,
  pas un pari gagnant. Le PMU est un pari mutuel : le prélèvement est
  retiré avant répartition, donc un écart doit le dépasser pour valoir
  quelque chose, et se vérifier sur plusieurs centaines de courses.
  Les motifs indiquent ce qui a pesé sur la note dans ce lot précis —
  ce sont des associations mesurées, pas des causes.
</div>
<script>const DONNEES={charge};const RAFRAICHIR={int(rafraichir)};{_JS}</script>
</body></html>"""
